from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial

import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised import SSLFineTuner
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from timm.data.mixup import Mixup
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from torch import nn
from torchmetrics import Accuracy
from torchvision import transforms

from main import RandMaskedSimCLR
from models import SimCLRViT
from utils.datamodules import CIFAR100DataModule, \
    Flowers102DataModule, OxfordIIITPetDataModule, build_imagenet
from utils.lr_wt_decay import param_groups_lrd, exclude_from_wt_decay
from utils.transforms import SimCLRFinetuneTransform, imagenet_normalization, \
    cifar10_normalization, cifar100_normalization, flower102_normalization, \
    oxford_iiit_pet_normalization


class CLREvaluator(SSLFineTuner):
    def __init__(
            self,
            protocol: str = 'linear',
            dataset: str = 'imagenet',
            img_size: int = 224,
            optim: str = 'sgd',
            warmup_epochs: int = 10,
            start_lr: float = 1e-6,
            exclude_bn_bias: bool = True,
            mixup_alpha: float = 0.,
            cutmix_alpha: float = 0.,
            label_smoothing: float = 0.,
            layer_decay: float = 1.,
            **kwargs
    ):
        """
        Args:
            protocol: evaluation protocol, including `linear` and `finetune`
            dataset: name of dataset for evaluation
            img_size: input image size
            optim: optimizer (SGD, Adam, or AdamW)
            warmup_epochs: linear warmup epochs
            start_lr: start learning rate
            exclude_bn_bias: exclude weight decay on 1d params (e.g. bn/ln and bias)
            mixup_alpha: mixup alpha, active if > 0.
            cutmix_alpha: cutmix alpha, active if > 0.
            label_smoothing: label smoothing regularization
            layer_decay: layer-wise learning rate decay
        """
        assert protocol in {'linear', 'finetune'}, f"unknown protocol: {protocol}"
        assert optim in {'sgd', 'adam', 'adamw'}, f"unknown optimizer: {optim}"

        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['backbone'])

        self.protocol = protocol
        self.dataset = dataset
        self.optim = optim
        self.exclude_bn_bias = exclude_bn_bias
        self.mixup = None
        if mixup_alpha > 0. or cutmix_alpha > 0.:
            self.mixup = Mixup(mixup_alpha, cutmix_alpha,
                               label_smoothing=label_smoothing,
                               num_classes=self.linear_layer.n_classes)
        self.label_smoothing = label_smoothing
        self.layer_decay = layer_decay
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr

        if self.mixup is None:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = SoftTargetCrossEntropy()

        if dataset == "imagenet":
            normalization = imagenet_normalization()
        elif dataset == "cifar10":
            normalization = cifar10_normalization()
        elif dataset == "cifar100":
            normalization = cifar100_normalization()
        elif dataset == "flowers102":
            normalization = flower102_normalization()
        elif dataset == "oxford_iiit_pet":
            normalization = oxford_iiit_pet_normalization()
        self.normalization = normalization
        self.train_transform = SimCLRFinetuneTransform(
            img_size=img_size,
            normalize=normalization,
            eval_transform=False,
        )
        self.eval_transform = SimCLRFinetuneTransform(
            img_size=img_size,
            normalize=normalization,
            eval_transform=True,
        )

        self.train_acc5 = Accuracy(top_k=5)
        self.val_acc5 = Accuracy(top_k=5, compute_on_step=False)
        self.test_acc5 = Accuracy(top_k=5, compute_on_step=False)

    def on_train_epoch_start(self) -> None:
        if self.protocol == 'linear':
            self.backbone.eval()
        elif self.protocol == 'finetune':
            self.backbone.train()

    def forward(self, x, position=True):
        x = self.normalization(x)
        x, *_ = self.backbone(x, position)

        return x

    def shared_step(self, batch):
        x, y = batch
        if self.protocol == 'linear':
            with torch.no_grad():
                feats, *_ = self.backbone(x)
        elif self.protocol == 'finetune':
            feats, *_ = self.backbone(x)

        logits = self.linear_layer(feats[:, 0, :])
        loss = self.criterion(logits, y)

        return loss, logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.train_transform(x)
        target = y.clone()
        if self.mixup is not None:
            x, y = self.mixup(x, y)
        loss, logits = self.shared_step((x, y))
        acc = self.train_acc(logits.softmax(-1), target)
        acc5 = self.train_acc5(logits.softmax(-1), target)

        self.log(f"loss/xent_{self.protocol}/train", loss, prog_bar=True)
        self.log(f"acc/{self.protocol}_top_1/train_step", acc, prog_bar=True)
        self.log(f"acc/{self.protocol}_top_5/train_step", acc5, prog_bar=True)
        self.log(f"acc/{self.protocol}_top_1/train_epoch", self.train_acc,
                 on_step=False, on_epoch=True)
        self.log(f"acc/{self.protocol}_top_5/train_epoch", self.train_acc5,
                 on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.eval_transform(x)
        loss, logits = self.shared_step((x, y))
        self.val_acc(logits.softmax(-1), y)
        self.val_acc5(logits.softmax(-1), y)

        self.log(f"loss/xent_{self.protocol}/val",
                 loss, prog_bar=True, sync_dist=True)
        self.log(f"acc/{self.protocol}_top_1/val_epoch", self.val_acc)
        self.log(f"acc/{self.protocol}_top_5/val_epoch", self.val_acc5)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.eval_transform(x)
        loss, logits = self.shared_step((x, y))
        self.test_acc(logits.softmax(-1), y)
        self.test_acc5(logits.softmax(-1), y)

        self.log(f"loss/xent_{self.protocol}/test", loss, sync_dist=True)
        self.log(f"acc/{self.protocol}_top_1/test_epoch", self.test_acc)
        self.log(f"acc/{self.protocol}_top_5/test_epoch", self.test_acc5)

        return loss

    def configure_optimizers(self):
        param_groups = []
        # add backbone to params_groups while finetuning
        if self.protocol == "finetune":
            param_groups += param_groups_lrd(
                self.backbone,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                exclude_1d_params=self.exclude_bn_bias,
                no_weight_decay_list=("pos_embed", "cls_token"),
                layer_decay=self.layer_decay
            )

        # add linear head
        if self.exclude_bn_bias:
            linear_param_groups = exclude_from_wt_decay(
                self.linear_layer,
                self.weight_decay,
            )
        else:
            linear_param_groups = [{
                "params": self.linear_layer.parameters(),
                "weight_decay": self.weight_decay
            }]
        param_groups += linear_param_groups

        if self.optim == "sdg":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.learning_rate,
                nesterov=self.nesterov,
                momentum=0.9,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=self.learning_rate)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             self.decay_epochs,
                                                             gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )
        elif self.scheduler_type == "warmup-anneal":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, self.warmup_epochs, max_epochs=self.epochs,
                warmup_start_lr=self.start_lr, eta_min=self.final_lr
            )

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
        parser.add_argument("--batch_size", default=256, type=int,
                            help="batch size per gpu")
        parser.add_argument("--num_workers", default=4, type=int,
                            help="num of workers per GPU")
        parser.add_argument("--fast_dev_run", default=False, type=int)
        parser.add_argument("--fp16", default=False, action='store_true',
                            help="use fp16")

        # data params
        parser.add_argument("--dataset", default="imagenet", type=str,
                            help="dataset")
        parser.add_argument("--lmdb_25pct", default=False, action='store_true',
                            help="use dedicated 25%% LMDB dataset")
        parser.add_argument("--lmdb", default=False, action='store_true',
                            help="use LMDB dataset")
        parser.add_argument("--data_dir", default="dataset", type=str,
                            help="path to dataset")
        parser.add_argument("--protocol", default="linear", type=str,
                            choices=("linear", "finetune"),
                            help="evalution protocol")
        parser.add_argument("--sample_pct", default=100, type=int,
                            help="%% of labels for training")
        parser.add_argument("--ckpt_path", required=True, type=str,
                            help="path to ckpt")

        # regularization
        parser.add_argument("--layer_decay", default=1., type=float,
                            help="layer-wise decay")
        parser.add_argument("--mlp_drop_rate", default=0., type=float,
                            help="mlp dropout rate")
        parser.add_argument("--attention_drop_rate", default=0., type=float,
                            help="attention dropout rate")
        parser.add_argument("--drop_path_rate", default=0., type=float,
                            help="path dropout rate")
        parser.add_argument("--linear_head_drop_rate", default=0., type=float,
                            help="linear head dropout rate")
        parser.add_argument("--mixup_alpha", default=0., type=float,
                            help="mixup alpha")
        parser.add_argument("--cutmix_alpha", default=0., type=float,
                            help="cutmix alpha")
        parser.add_argument("--label_smoothing", default=0., type=float,
                            help="label smoothing ratio")

        # optimizer
        parser.add_argument("--optimizer", default="adamw", type=str,
                            help="choose between SGD, Adam, and AdamW")
        parser.add_argument("--learning_rate", default=1e-3, type=float,
                            help="base learning rate")
        parser.add_argument("--nesterov", default=False, type=bool,
                            help="SGD Nesterov flag")
        parser.add_argument("--weight_decay", default=0.05, type=float,
                            help="weight decay")
        parser.add_argument("--exclude_bn_bias", default=True,
                            action=BooleanOptionalAction,
                            help="exclude bn/ln/bias from weight decay")
        parser.add_argument("--scheduler_type", default="warmup-anneal", type=str,
                            help="choose between step, cosine, and warmup-anneal")
        parser.add_argument("--decay_epochs", default=(60, 80), type=float,
                            help="multi-step decay epochs")
        parser.add_argument("--gamma", default=0.1, type=float,
                            help="multi-step decay rate")
        parser.add_argument("--warmup_epochs", type=int, default=10,
                            help="number of warmup epochs")
        parser.add_argument("--start_lr", default=1e-6, type=float,
                            help="start learning rate")
        parser.add_argument("--final_lr", default=1e-6, type=float,
                            help="final learning rate")

        return parser


if __name__ == "__main__":
    seed_everything(0)
    parser = ArgumentParser()
    parser.add_argument("--version", default=None, type=str)
    parser.add_argument("--log_path", default="lightning_logs", type=str)
    parser.add_argument("--log_steps", default=50, type=int)
    parser.add_argument("--resume_ckpt_path", default=None, type=str)
    parser.add_argument("--track_grad", default=True, type=BooleanOptionalAction)
    parser = CLREvaluator.add_model_specific_args(parser)
    args = parser.parse_args()

    pretrained = RandMaskedSimCLR.load_from_checkpoint(args.ckpt_path, strict=False)
    # a bit hacky here, replace ViT with dropout rate
    pretained_state_dict = pretrained.state_dict()
    pretrained.siamese_net = SimCLRViT(
        pretrained.img_size,
        pretrained.patch_size,
        pretrained.in_chans,
        pretrained.embed_dim,
        pretrained.depth,
        pretrained.num_heads,
        pretrained.mlp_ratio,
        pretrained.proj_dim,
        drop_rate=args.mlp_drop_rate,
        attention_drop_rate=args.attention_drop_rate,
        drop_path_rate=args.drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        weight_sharing=pretrained.weight_sharing,
    )
    pretrained.load_state_dict(pretained_state_dict)

    if args.dataset == "imagenet":
        dm = build_imagenet(args.sample_pct, args.lmdb, args.lmdb_25pct)
    elif args.dataset == "cifar10":
        dm = CIFAR10DataModule
    elif args.dataset == "cifar100":
        dm = CIFAR100DataModule
    elif args.dataset == "flowers102":
        dm = Flowers102DataModule
    elif args.dataset == "oxford_iiit_pet":
        dm = OxfordIIITPetDataModule
    else:
        raise NotImplementedError(f"Unimplemented dataset: {args.dataset}")
    dm = dm(data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers)
    args.num_classes = dm.num_classes

    dm.train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(pretrained.img_size),
        transforms.ToTensor(),
    ])
    dm.val_transforms = dm.test_transforms = transforms.Compose([
        transforms.Resize(int(pretrained.img_size + 0.1 * pretrained.img_size)),
        transforms.CenterCrop(pretrained.img_size),
        transforms.ToTensor(),
    ])

    evaluator = CLREvaluator(
        protocol=args.protocol,
        dataset=args.dataset,
        img_size=pretrained.img_size,
        backbone=pretrained.siamese_net,
        in_features=pretrained.embed_dim,
        num_classes=args.num_classes,
        epochs=args.max_epochs,
        dropout=args.linear_head_drop_rate,
        layer_decay=args.layer_decay,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        label_smoothing=args.label_smoothing,
        optim=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        exclude_bn_bias=args.exclude_bn_bias,
        scheduler_type=args.scheduler_type,
        decay_epochs=args.decay_epochs,
        gamma=args.gamma,
        warmup_epochs=args.warmup_epochs,
        start_lr=args.start_lr,
        final_lr=args.final_lr,
    )

    logger = TensorBoardLogger(args.log_path, name="evaluation", version=args.version)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        save_last=True,
        monitor=f"loss/xent_{args.protocol}/val"
    )
    callbacks = [model_checkpoint, lr_monitor]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        devices=args.gpus if args.gpus > 0 else None,
        num_nodes=args.num_nodes,
        accelerator="gpu" if args.gpus > 0 else None,
        strategy="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        track_grad_norm=2 if args.track_grad else -1,
        precision=16 if args.fp16 else 32,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_steps,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(evaluator, datamodule=dm, ckpt_path=args.resume_ckpt_path)
    trainer.test(dataloaders=dm, ckpt_path='last')
