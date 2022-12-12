import copy
from argparse import ArgumentParser, BooleanOptionalAction

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

from models import SimCLRResNet
from utils.criteria import multi_view_info_nce_loss, multi_view_cov_reg_loss
from utils.datamodules import FewShotImagenetDataModule, FewShotImagenetLMDBDataModule, \
    Imagenet25pctLMDBDataModule, ImagenetLMDBDataModule
from utils.lr_wt_decay import exclude_from_wt_decay
from utils.transforms import SimCLRPretrainPostTransform, imagenet_normalization, MultiCropPretrainPreTransform


class SimCLR(LightningModule):
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
            layers: tuple = (3, 4, 6, 3),
            embed_dim: int = 2048,
            ema_momentum: float = 0.,
            proj_dim: int = 128,
            temperature: float = 0.1,
            cov_reg_norm: bool = False,
            cov_reg_coeff: float = 0.,
            optimizer: str = "adamw",
            learning_rate: float = 1e-3,
            weight_decay: float = 0.05,
            exclude_bn_bias: bool = False,
            warmup_epochs: int = 10,
            **kwargs
    ):
        super(SimCLR, self).__init__()
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

        # ResNet params
        self.layers = layers

        # SimCLR params
        self.ema_momentum = ema_momentum
        self.proj_dim = proj_dim
        self.temperature = temperature

        # CovReg
        self.cov_reg_norm = cov_reg_norm
        self.cov_reg_coeff = cov_reg_coeff

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
        self.siamese_net = SimCLRResNet(layers=layers, proj_dim=proj_dim)
        if ema_momentum > 0:
            self.siamese_net_sg = copy.deepcopy(self.siamese_net)
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
            img_ = torch.cat([
                self.transform(i) for i in img[start_index:end_index]
            ])
            yield self.siamese_net(img_)

    def shared_step(self, img):
        batch_size, *_ = img[0].size()
        reps, proj = [], []
        for reps_, proj_ in self.forward_multi_crop(img):
            reps.append(reps_)
            proj.append(proj_.view(-1, batch_size, self.proj_dim))
        reps = torch.cat(reps)
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

        return reps, proj, {
            "clr": loss_clr,
            "cov_reg": loss_cov_reg,
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
            'loss/pretrain/val': losses["all"],
            'loss/linear_probe/val': loss_xent,
            'acc/linear_probe_top_1/val': acc_top_1,
            'acc/linear_probe_top_5/val': acc_top_5,
        }, sync_dist=True)
        return losses["all"] + loss_xent

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            param_groups = exclude_from_wt_decay(self, self.weight_decay)
        else:
            param_groups = [
                {"params": self.parameters(), "weight_decay": self.weight_decay}
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
        parser.add_argument("--lmdb_25pct", default=False, action='store_true',
                            help="use dedicated 25%% LMDB dataset")
        parser.add_argument("--lmdb", default=False, action='store_true',
                            help="use LMDB dataset")
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
        parser.add_argument("--layers", default=(3, 4, 6, 3), type=int, nargs="+",
                            help="ResNet layers")
        parser.add_argument("--embed_dim", default=2048, type=int,
                            help="embedding dimension")
        # SimCLR
        parser.add_argument("--ema_momentum", default=0., type=float,
                            help="ema momentum")
        parser.add_argument("--proj_dim", default=128, type=int,
                            help="projection head output dimension")
        parser.add_argument("--temperature", default=0.1, type=float,
                            help="temperature parameter in InfoNCE loss")
        # CovReg
        parser.add_argument("--cov_reg_norm", default=True, action=BooleanOptionalAction,
                            help="use correlation instead of covariance")
        parser.add_argument("--cov_reg_coeff", default=0., type=float,
                            help="coefficient on covariance regularization loss")

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
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == "imagenet":
        if args.sample_pct < 100:
            if args.lmdb_25pct:
                dm = Imagenet25pctLMDBDataModule(
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers
                )
            elif args.lmdb:
                dm = FewShotImagenetLMDBDataModule(
                    args.data_dir,
                    label_pct=args.sample_pct,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers
                )
            else:
                dm = FewShotImagenetDataModule(
                    args.data_dir,
                    label_pct=args.sample_pct,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers
                )
        else:
            if args.lmdb:
                dm = ImagenetLMDBDataModule
            else:
                dm = ImagenetDataModule
            dm = dm(data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)
    else:
        raise NotImplementedError(f"Unimplemented dataset: {args.dataset}")
    args.num_samples = dm.num_samples
    args.num_classes = dm.num_classes

    dm.train_transforms = dm.val_transforms = MultiCropPretrainPreTransform(
        args.size_crops, args.num_crops, args.min_scale_crops, args.max_scale_crops
    )

    model = SimCLR(**args.__dict__)

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
