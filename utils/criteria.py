import torch
from torch.nn import functional as F

from utils.dist import SyncFunction


def info_nce_loss(feat1, feat2, temp, eps=1e-6):
    feat1 = F.normalize(feat1)
    feat2 = F.normalize(feat2)
    # feat{1,2}: [batch_size, proj_dim]
    feat = torch.stack((feat1, feat2), dim=1)
    # feat: [batch_size, 2, proj_dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        feat = SyncFunction.apply(feat)
    # feat: [batch_size (* world_size), 2, proj_dim]
    feat1, feat2 = feat.unbind(1)
    # feat{1,2}: [batch_size (* world_size), proj_dim]
    feat = torch.cat((feat1, feat2))
    # feat: [batch_size*2 (* world_size), proj_dim]

    # All samples, filling diagonal to remove identity similarity ((2N)^2 - 2N)
    all_sim = (feat @ feat.T).fill_diagonal_(0)
    # all_sim: [batch_size*2 (* world_size), batch_size*2 (* world_size)]
    all_ = torch.exp(all_sim / temp).sum(-1)
    # all_: [batch_size*2 (* world_size)]

    # Positive samples (2N)
    pos_sim = (feat1 * feat2).sum(-1)
    # pos_sim: [batch_size (* world_size)]
    pos = torch.exp(pos_sim / temp)
    # Following all samples, compute positive similarity twice
    pos = torch.cat((pos, pos))
    # pos: [batch_size*2 (* world_size)]

    loss = -torch.log(pos / (all_ + eps)).mean()

    return loss


def multi_view_info_nce_loss(proj, temp):
    batch_size, num_crops, proj_dim = proj.size()
    if num_crops == 2:
        info_nce_loss(proj[:, 0, :], proj[:, 1, :], temp)

    # InfoNCE with multiple positive pairs
    proj = F.normalize(proj, dim=-1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        proj = SyncFunction.apply(proj)
        batch_size = proj.size(0)
    # proj: [batch_size (* world_size), num_crops, proj_dim]

    pos_sim = proj @ proj.transpose(1, 2)
    pos_sim = pos_sim.masked_select(~torch.eye(
        num_crops, dtype=torch.bool, device=proj.device
    ))
    pos_sim = pos_sim.view(batch_size * num_crops, -1)
    pos = torch.exp(pos_sim / temp)

    neg_proj = torch.stack([
        proj[neg_idx].view(-1, proj_dim)
        for neg_idx in ~torch.eye(batch_size, dtype=torch.bool)
    ])
    neg_sim = (proj @ neg_proj.transpose(1, 2)).view(batch_size * num_crops, -1)
    neg = torch.exp(neg_sim / temp).sum(-1)

    all_ = pos + neg.unsqueeze(-1)

    loss = -torch.log(pos / all_).mean()

    return loss


def cov_reg_loss(proj, norm=False):
    _, proj_dim = proj.size()
    # proj: [batch_size, proj_dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        proj = SyncFunction.apply(proj)
    # proj: [batch_size (* world_size), proj_dim]
    proj_cov = proj.T.corrcoef() if norm else proj.T.cov()
    off_diag_mask = ~torch.eye(
        proj_dim, dtype=torch.bool, device=proj.device
    )
    proj_cov_off_diag = proj_cov.masked_select(off_diag_mask)

    return (proj_cov_off_diag ** 2).sum() / (proj_dim * (proj_dim - 1))


def multi_view_cov_reg_loss(proj, norm=False):
    batch_size, num_crops, proj_dim = proj.size()
    # proj: [batch_size, num_crops, proj_dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        proj = SyncFunction.apply(proj)

    loss_all = 0.
    # proj: [batch_size (* world_size), num_crops, proj_dim]
    for proj_ in proj.transpose(0, 1):
        proj_cov = proj_.T.corrcoef() if norm else proj_.T.cov()
        off_diag_mask = ~torch.eye(
            proj_dim, dtype=torch.bool, device=proj_.device
        )
        proj_cov_off_diag = proj_cov.masked_select(off_diag_mask)
        loss = (proj_cov_off_diag ** 2).sum() / (proj_dim * (proj_dim - 1))
        loss_all += loss

    return loss_all / num_crops
