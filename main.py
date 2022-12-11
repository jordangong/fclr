import copy
from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
import torch_optimizer as optim
from pl_bolts.callbacks.knn_online import KNNOnlineEvaluator
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.optimizers import linear_warmup_decay
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torchmetrics import Accuracy

from models import SimCLRMaskedViT, PosReconHead
from utils.criteria import multi_view_info_nce_loss, multi_view_cov_reg_loss
from utils.datamodules import FewShotImagenetDataModule
from utils.lr_wt_decay import param_groups_lrd, exclude_from_wt_decay
from utils.transforms import SimCLRPretrainPostTransform, imagenet_normalization, MultiCropPretrainPreTransform


class RandMaskedSimCLR(LightningModule):
    def __init__(
            self,
            num_samples: int,
            num_nodes: int = 1,
            gpus: int = 1,
            batch_size: int = 128,
            max_epochs: int = 100,
            dataset: str = "imagenet",
            num_classes: int = 1000,
            size_crops: tuple[int, ...] = (224,),
            num_crops: tuple[int, ...] = (2,),
            gaussian_blur: bool = True,
            jitter_strength: float = 1.,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 384,
            depth: int = 12,
            num_heads: int = 6,
            mlp_ratio: int = 4,
            mlp_drop_rate: float = 0.,
            attention_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_decay: float = 1.,
            weight_sharing: Optional[str] = None,
            position: bool = True,
            shuffle: bool = False,
            mask_ratio: float = 0.75,
            ema_momentum: float = 0.,
            mask_ratio_sg: float = 0.75,
            proj_dim: int = 128,
            temperature: float = 0.1,
            cov_reg_norm: bool = False,
            cov_reg_coeff: float = 0.,
            pos_recon_depth: int = 0,
            pos_recon_num_heads: int = 6,
            pos_recon_coeff: float = 0.,
            optimizer: str = "adamw",
            learning_rate: float = 1e-3,
            weight_decay: float = 0.05,
            exclude_bn_bias: bool = False,
            warmup_epochs: int = 10,
            **kwargs
    ):
        super(RandMaskedSimCLR, self).__init__()
        self.save_hyperparameters()

        # Training
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.gpus = gpus
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        # compute iters per epoch
        global_batch_size = num_nodes * gpus * batch_size if gpus > 0 else batch_size
        self.train_iters_per_epoch = num_samples // global_batch_size

        # Data
        self.dataset = dataset
        self.num_classes = num_classes
        self.size_crops = size_crops
        self.num_crops = num_crops
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength

        # ViT params
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.weight_sharing = weight_sharing
        # Regularization
        self.mlp_drop_rate = mlp_drop_rate
        self.attention_drop_rate = attention_drop_rate
        self.drop_path_rate = drop_path_rate
        self.layer_decay = layer_decay

        # FCLR params
        self.position = position
        self.shuffle = shuffle
        self.mask_ratio = mask_ratio
        self.ema_momentum = ema_momentum
        self.mask_ratio_sg = mask_ratio_sg
        self.proj_dim = proj_dim
        self.temperature = temperature

        # CovReg
        self.cov_reg_norm = cov_reg_norm
        self.cov_reg_coeff = cov_reg_coeff

        # PosRecon
        self.pos_recon_depth = pos_recon_depth
        self.pos_recon_num_heads = pos_recon_num_heads
        self.pos_recon_coeff = pos_recon_coeff

        # Optimizer
        self.optim = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bn_bias = exclude_bn_bias
        self.warmup_epochs = warmup_epochs

        # Transforms
        if dataset == "imagenet":
            normalization = imagenet_normalization()
        self.normalization = normalization
        self.transform = SimCLRPretrainPostTransform(
            img_size=img_size,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength,
            normalize=normalization,
        )

        # Modules
        self.siamese_net = SimCLRMaskedViT(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            proj_dim,
            mlp_drop_rate,
            attention_drop_rate,
            drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            weight_sharing=weight_sharing,
        )
        if ema_momentum > 0:
            self.siamese_net_sg = copy.deepcopy(self.siamese_net)
        if pos_recon_coeff > 0:
            self.pos_recon_head = PosReconHead(
                embed_dim,
                pos_recon_depth,
                pos_recon_num_heads
            )
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Online metrics
        self.train_acc_top_1 = Accuracy(top_k=1)
        self.train_acc_top_5 = Accuracy(top_k=5)
        self.val_acc_top_1 = Accuracy(top_k=1, compute_on_step=False)
        self.val_acc_top_5 = Accuracy(top_k=5, compute_on_step=False)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_momentum > 0:
            for online_p, target_p in zip(self.siamese_net.parameters(),
                                          self.siamese_net_sg.parameters()):
                em = self.ema_momentum
                target_p.data = target_p.data * em + online_p.data * (1.0 - em)

    def forward(self, x):
        x = self.normalization(x)
        x, _ = self.siamese_net(x)

        return x

    def forward_multi_crop(self, img):
        # Concatenate crop with the same size, augment, and forward
        img_sizes = torch.tensor([i.size(-1) for i in img])
        end_indices = img_sizes.unique_consecutive(return_counts=True)[-1].cumsum(0)
        start_indices = torch.cat((torch.tensor([0]), end_indices[:-1]))
        for i, (start_index, end_index) in enumerate(zip(start_indices, end_indices)):
            # Assume global crop at first
            mask_ratio = self.mask_ratio if i == 0 else 0.
            img_ = torch.cat(img[start_index:end_index])
            img_ = self.transform(img_)
            yield self.siamese_net(img_, self.position, self.shuffle, mask_ratio)

    def shared_step(self, img):
        num_crops = len(img)
        batch_size, *_ = img[0].size()
        patch_embed, visible_idx, proj = [], [], []
        for patch_embed_, visible_idx_, proj_ in self.forward_multi_crop(img):
            patch_embed.append(patch_embed_)
            visible_idx.append(visible_idx_)
            proj.append(proj_.view(-1, batch_size, self.proj_dim))
        reps = torch.cat([pe[:, 0, :] for pe in patch_embed])
        patch_embed = patch_embed[0][:, 1:, :]
        visible_idx = visible_idx[0]
        proj = torch.cat(proj).transpose(0, 1)

        if self.ema_momentum > 0:
            proj_sg = []
            with torch.no_grad():
                for _, _, proj_ in self.forward_multi_crop(img):
                    proj_sg.append(proj_.view(-1, batch_size, self.proj_dim))
            proj_sg = torch.cat(proj_sg).transpose(0, 1)
            proj = torch.cat((proj, proj_sg))

        loss_clr = multi_view_info_nce_loss(proj, self.temperature)

        loss_cov_reg = multi_view_cov_reg_loss(proj, self.cov_reg_norm)
        loss = loss_clr + self.cov_reg_coeff * loss_cov_reg

        loss_pos_recon = 0.
        if self.pos_recon_coeff > 0:
            pos_embed = self.siamese_net.pos_embed.expand(patch_embed.size(0), -1, -1)
            if visible_idx is not None:
                pos_embed = pos_embed.gather(1, visible_idx)
            pos_embed_pred = self.pos_recon_head(patch_embed)
            loss_pos_recon = F.mse_loss(pos_embed_pred, pos_embed)
            loss += self.pos_recon_coeff * loss_pos_recon

        return reps, proj, {
            "clr": loss_clr,
            "cov_reg": loss_cov_reg,
            "pos_recon": loss_pos_recon,
            "all": loss,
        }

    def linear_probe(self, reps, labels, acc_top_1_fn, acc_top_5_fn):
        labels = labels.repeat(reps.size(0) // labels.size(0))
        logits = self.classifier(reps)
        loss = F.cross_entropy(logits, labels)
        acc_top_1 = acc_top_1_fn(logits.softmax(-1), labels)
        acc_top_5 = acc_top_5_fn(logits.softmax(-1), labels)

        return loss, acc_top_1, acc_top_5

    def training_step(self, batch, batch_idx):
        img, labels = batch
        reps, proj, losses = self.shared_step(img)

        loss_xent, acc_top_1, acc_top_5 = self.linear_probe(
            reps.detach(), labels, self.train_acc_top_1, self.train_acc_top_5
        )

        self.log_dict({
            'loss/clr/train': losses["clr"],
            'loss/cov_reg/train': losses["cov_reg"],
            'loss/pos_recon/train': losses["pos_recon"],
            'loss/pretrain/train': losses["all"],
            'loss/linear_probe/train': loss_xent,
            'acc/linear_probe_top_1/train': acc_top_1,
            'acc/linear_probe_top_5/train': acc_top_5,
            'norm/reps': reps.norm(dim=-1).mean(),
            'norm/proj': proj.norm(dim=-1).mean(),
        }, sync_dist=True)
        return losses["all"] + loss_xent

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        reps, proj, losses = self.shared_step(img)

        loss_xent, acc_top_1, acc_top_5 = self.linear_probe(
            reps.detach(), labels, self.val_acc_top_1, self.val_acc_top_5
        )

        self.log_dict({
            'loss/clr/val': losses["clr"],
            'loss/cov_reg/val': losses["cov_reg"],
            'loss/pos_recon/val': losses["pos_recon"],
            'loss/pretrain/val': losses["all"],
            'loss/linear_probe/val': loss_xent,
            'acc/linear_probe_top_1/val': acc_top_1,
            'acc/linear_probe_top_5/val': acc_top_5,
        }, sync_dist=True)
        return losses["all"] + loss_xent

    def configure_optimizers(self):
        param_groups = param_groups_lrd(
            self.siamese_net,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            exclude_1d_params=self.exclude_bn_bias,
            no_weight_decay_list=("pos_embed", "cls_token"),
            layer_decay=self.layer_decay
        )
        if self.exclude_bn_bias:
            param_groups += exclude_from_wt_decay(self.classifier, self.weight_decay)
            if self.pos_recon_coeff > 0:
                param_groups += exclude_from_wt_decay(self.pos_recon_head, self.weight_decay)
        else:
            param_groups += [
                {"params": self.classifier.parameters(), "weight_decay": self.weight_decay}
            ]
            if self.pos_recon_coeff > 0:
                param_groups += [
                    {"params": self.pos_recon_head.parameters(), "weight_decay": self.weight_decay}
                ]

        if self.optim == "lars":
            optimizer = optim.LARS(param_groups, lr=self.learning_rate, momentum=0.9)
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=self.learning_rate)
        elif self.optim == "lamb":
            optimizer = optim.Lamb(param_groups, lr=self.learning_rate)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training params
        parser.add_argument("--num_nodes", default=1, type=int,
                            help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int,
                            help="number of gpus to train on")
        parser.add_argument("--max_epochs", default=100, type=int,
                            help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="max steps")
        parser.add_argument("--batch_size", default=128, type=int,
                            help="batch size per gpu")
        parser.add_argument("--num_workers", default=8, type=int,
                            help="num of workers per GPU")
        parser.add_argument("--fast_dev_run", default=False, type=int)
        parser.add_argument("--fp16", default=False, action='store_true',
                            help="use fp16")

        # transform params
        parser.add_argument("--dataset", type=str, default="imagenet",
                            help="dataset")
        parser.add_argument("--data_dir", type=str, default="dataset",
                            help="path to dataset")
        parser.add_argument("--sample_pct", type=int, default=100,
                            help="%% of samples for training (only for ablation)")
        parser.add_argument("--size_crops", default=(224,), type=int, nargs="+",
                            help="crop resolution")
        parser.add_argument("--num_crops", default=(2,), type=int, nargs="+",
                            help="number of crops")
        parser.add_argument("--min_scale_crops", default=(0.08,), type=float, nargs="+",
                            help="RandResizeCrop minimum scales")
        parser.add_argument("--max_scale_crops", default=(1.,), type=float, nargs="+",
                            help="RandResizeCrop maximum scales")
        parser.add_argument("--gaussian_blur", default=True,
                            action=BooleanOptionalAction, help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0,
                            help="jitter strength")

        # model params
        parser.add_argument("--img_size", default=224, type=int,
                            help="input image size")
        parser.add_argument("--patch_size", default=16, type=int,
                            help="patch size")
        parser.add_argument("--in_chans", default=3, type=int,
                            help="number of in channels")
        parser.add_argument("--embed_dim", default=384, type=int,
                            help="embedding dimension")
        parser.add_argument("--depth", default=12, type=int,
                            help="number of Transformer blocks")
        parser.add_argument("--num_heads", default=6, type=int,
                            help="number of self-attention heads")
        parser.add_argument("--mlp_ratio", default=4, type=int,
                            help="Ratio of embedding dim to MLP dim")
        parser.add_argument("--weight_sharing", default=None, type=str,
                            help="ALBERT-style weight sharing, "
                                 "choose from None, attn, ffn, or all")
        # regularization
        parser.add_argument("--layer_decay", default=1., type=float,
                            help="layer-wise decay")
        parser.add_argument("--mlp_drop_rate", default=0.0, type=float,
                            help="mlp dropout rate")
        parser.add_argument("--attention_drop_rate", default=0.0, type=float,
                            help="attention dropout rate")
        parser.add_argument("--drop_path_rate", default=0.0, type=float,
                            help="path dropout rate")
        # FCLR
        parser.add_argument("--position", default=True, action=BooleanOptionalAction,
                            help="add positional embedding or not")
        parser.add_argument("--shuffle", default=False, action='store_true',
                            help="shuffle positional embedding or not")
        parser.add_argument("--mask_ratio", default=0.75, type=float,
                            help="mask ratio of patches")
        parser.add_argument("--ema_momentum", default=0., type=float,
                            help="ema momentum")
        parser.add_argument("--mask_ratio_sg", default=0.75, type=float,
                            help="mask ratio of patches on target branch")
        parser.add_argument("--proj_dim", default=128, type=int,
                            help="projection head output dimension")
        parser.add_argument("--temperature", default=0.1, type=float,
                            help="temperature parameter in InfoNCE loss")
        # CovReg
        parser.add_argument("--cov_reg_norm", default=True, action=BooleanOptionalAction,
                            help="use correlation instead of covariance")
        parser.add_argument("--cov_reg_coeff", default=0., type=float,
                            help="coefficient on covariance regularization loss")
        # PosRecon
        parser.add_argument("--pos_recon_depth", default=0, type=int,
                            help="depth of PosRecon head, 0 for linear layer")
        parser.add_argument("--pos_recon_num_heads", default=6, type=int,
                            help="number of attention heads in PosRecon head")
        parser.add_argument("--pos_recon_coeff", default=0., type=float,
                            help="coefficient on position reconstruction loss")

        # Optimizer
        parser.add_argument("--optimizer", default="adamw", type=str,
                            help="choose between adamw/lars")
        parser.add_argument("--learning_rate", default=1e-3, type=float,
                            help="base learning rate")
        parser.add_argument("--weight_decay", default=0.05, type=float,
                            help="weight decay")
        parser.add_argument("--exclude_bn_bias", default=True,
                            action=BooleanOptionalAction,
                            help="exclude bn/ln/bias from weight decay")
        parser.add_argument("--warmup_epochs", default=10, type=int,
                            help="number of warmup epochs")

        return parser


if __name__ == '__main__':
    seed_everything(0)
    parser = ArgumentParser()
    parser.add_argument("--version", default=None, type=str)
    parser.add_argument("--log_path", default="lightning_logs", type=str)
    parser.add_argument("--resume_ckpt_path", default=None, type=str)
    parser.add_argument("--track_grad", default=True, action=BooleanOptionalAction)
    parser.add_argument("--knn_probe", default=False, action='store_true')
    parser = RandMaskedSimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == "imagenet":
        if args.sample_pct < 100:
            dm = FewShotImagenetDataModule(args.data_dir,
                                           label_pct=args.sample_pct,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers)
        else:
            dm = ImagenetDataModule(data_dir=args.data_dir,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
        args.num_samples = dm.num_samples
        args.num_classes = 1000
    else:
        raise NotImplementedError(f"Unimplemented dataset: {args.dataset}")

    dm.train_transforms = dm.val_transforms = MultiCropPretrainPreTransform(
        args.size_crops, args.num_crops, args.min_scale_crops, args.max_scale_crops
    )

    model = RandMaskedSimCLR(**args.__dict__)

    logger = TensorBoardLogger(args.log_path, name="pretrain", version=args.version)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, monitor="loss/pretrain/val")

    callbacks = [model_checkpoint]
    if args.knn_probe:
        callbacks.append(KNNOnlineEvaluator())
    callbacks.append(lr_monitor)

    if args.gpus > 1:
        if args.ema_momentum > 0:
            strategy = "ddp"
        else:
            strategy = "ddp_find_unused_parameters_false"
    else:
        strategy = None

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        devices=args.gpus if args.gpus > 0 else None,
        num_nodes=args.num_nodes,
        accelerator="gpu" if args.gpus > 0 else None,
        strategy=strategy,
        sync_batchnorm=True if args.gpus > 1 else False,
        track_grad_norm=2 if args.track_grad else -1,
        precision=16 if args.fp16 else 32,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt_path)
