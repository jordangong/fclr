import io
import os
from typing import Any, Callable, Optional

import lmdb
import numpy as np
import pyarrow as pa
from PIL import Image
from pl_bolts.datamodules import ImagenetDataModule, CIFAR10DataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datasets import UnlabeledImagenet
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR100, Flowers102, OxfordIIITPet


def default_train_transform(
        image_size: int,
        normalization: Callable,
) -> Callable:
    return transform_lib.Compose([
        transform_lib.RandomResizedCrop(image_size),
        transform_lib.RandomHorizontalFlip(),
        transform_lib.ToTensor(),
        normalization(),
    ])


def default_val_transform(
        image_size: int,
        normalization: Callable,
) -> Callable:
    return transform_lib.Compose([
        transform_lib.Resize(int(image_size + 0.1 * image_size)),
        transform_lib.CenterCrop(image_size),
        transform_lib.ToTensor(),
        normalization(),
    ])


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.imgs: list = pa.deserialize(txn.get(b'__imgs__'))
            self.classes: list = pa.deserialize(txn.get(b'__classes__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        key, target = self.imgs[index]
        with self.env.begin(write=False) as txn:
            img_bytes, _ = pa.deserialize(txn.get(key))

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.db_path})"


class ImageNetLMDB(ImageFolderLMDB):
    def __init__(self, root: str, split: str = "train", **kwargs: Any):
        self.root = os.path.expanduser(root)
        self.split = split

        super().__init__(self.split_folder, **kwargs)

    @property
    def split_path(self) -> str:
        return os.path.join(self.root, f"{self.split}.lmdb")


class UnlabeledImagenetLMDB(ImageNetLMDB):
    def __init__(
            self,
            root,
            split: str = "train",
            num_classes: int = -1,
            num_imgs_per_class: int = -1,
            num_imgs_per_class_val_split: int = 50,
            **kwargs,
    ):
        """
        Args:
            root: path of dataset
            split:
            num_classes: Sets the limit of classes
            num_imgs_per_class: Limits the number of images per class
            num_imgs_per_class_val_split: How many images per class to generate the val split
            kwargs:
        """
        root = self.root = os.path.expanduser(root)

        # [train], [val] --> [train, val], [test]
        original_split = split
        if split == "train" or split == "val":
            split = "train"

        if split == "test":
            split = "val"

        self.split = split

        super(ImageNetLMDB, self).__init__(self.split_path, **kwargs)

        # shuffle images first
        np.random.seed(1234)
        np.random.shuffle(self.imgs)

        # partition train set into [train, val]
        if split == "train":
            train, val = self.partition_train_set(self.imgs, num_imgs_per_class_val_split)
            if original_split == "train":
                self.imgs = train
            if original_split == "val":
                self.imgs = val

        # limit the number of images in train or test set since the limit was already applied to the val set
        if split in ["train", "test"]:
            if num_imgs_per_class != -1:
                clean_imgs = []
                cts = {x: 0 for x in range(len(self.classes))}
                for img_name, idx in self.imgs:
                    if cts[idx] < num_imgs_per_class:
                        clean_imgs.append((img_name, idx))
                        cts[idx] += 1

                self.imgs = clean_imgs

        # limit the number of classes
        if num_classes != -1:
            # choose the classes at random (but deterministic)
            ok_classes = list(range(num_classes))
            np.random.seed(1234)
            np.random.shuffle(ok_classes)
            ok_classes = ok_classes[:num_classes]
            ok_classes = set(ok_classes)

            clean_imgs = []
            for img_name, idx in self.imgs:
                if idx in ok_classes:
                    clean_imgs.append((img_name, idx))

            self.imgs = clean_imgs

        # shuffle again for final exit
        np.random.seed(1234)
        np.random.shuffle(self.imgs)

    def partition_train_set(self, imgs, nb_imgs_in_val):
        val = []
        train = []

        cts = {x: 0 for x in range(len(self.classes))}
        for img_name, idx in imgs:
            if cts[idx] < nb_imgs_in_val:
                val.append((img_name, idx))
                cts[idx] += 1
            else:
                train.append((img_name, idx))

        return train, val


class ImagenetLMDBDataModule(ImagenetDataModule):
    def prepare_data(self) -> None:
        files = os.listdir(self.data_dir)
        for split in ["train", "val"]:
            if file := f"{split}.lmdb" not in files:
                raise FileNotFoundError(f"{file} file not found")

    def train_dataloader(self) -> DataLoader:
        """Uses the train split of imagenet2012 and puts away a portion of it for the validation split."""
        transforms = self.train_transform() if self.train_transforms is None else self.train_transforms

        dataset = UnlabeledImagenetLMDB(
            self.data_dir,
            num_imgs_per_class=-1,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            split="train",
            transform=transforms,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        """Uses the part of the train split of imagenet2012  that was not used for training via
        `num_imgs_per_val_class`
        """
        transforms = self.val_transform() if self.val_transforms is None else self.val_transforms

        dataset = UnlabeledImagenetLMDB(
            self.data_dir,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            split="val",
            transform=transforms,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """Uses the validation split of imagenet2012 for testing."""
        transforms = self.val_transform() if self.test_transforms is None else self.test_transforms

        dataset = UnlabeledImagenetLMDB(
            self.data_dir,
            num_imgs_per_class=-1,
            split="test",
            transform=transforms,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader


class FewShotImagenetDataModule(ImagenetDataModule):
    name = "few-shot-imagenet"

    def __init__(self, *args, label_pct: int = 1, **kwargs):
        """
        Args:
            label_pct: % of labels for training
        """
        super().__init__(*args, **kwargs)
        self.label_pct = label_pct
        self.num_samples = int(label_pct / 100 * self.num_samples)

        # just to silence the error from the linter
        self.train_transforms = None

    def train_dataloader(self) -> DataLoader:
        """
        Uses the train split of imagenet2012, puts away a portion of it
        for the validation split, and samples `top-n`% of labeled
        training set in class-balanced way."""
        if self.train_transforms is None:
            transforms = self.train_transform()
        else:
            transforms = self.train_transforms

        dataset = UnlabeledImagenet(
            self.data_dir,
            num_imgs_per_class=self.num_samples // self.num_classes,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            meta_dir=self.meta_dir,
            split="train",
            transform=transforms,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader


class FewShotImagenetLMDBDataModule(ImagenetLMDBDataModule, FewShotImagenetDataModule):
    def train_dataloader(self) -> DataLoader:
        """
        Uses the train split of imagenet2012, puts away a portion of it
        for the validation split, and samples `top-n`% of labeled
        training set in class-balanced way."""
        if self.train_transforms is None:
            transforms = self.train_transform()
        else:
            transforms = self.train_transforms

        dataset = UnlabeledImagenetLMDB(
            self.data_dir,
            num_imgs_per_class=self.num_samples // self.num_classes,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            split="train",
            transform=transforms,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader


class CIFAR100DataModule(CIFAR10DataModule):
    name = "cifar100"
    dataset_cls = CIFAR100

    @property
    def num_classes(self) -> int:
        return 100


class Flowers102DataModule(LightningDataModule):
    """
    Oxford 102 Flowers train, val and test dataloaders.
    """

    name = "flowers102"

    def __init__(
            self,
            data_dir: str,
            image_size: int = 224,
            num_workers: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to save/load the data
            image_size: final image size
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        return 102

    def prepare_data(self) -> None:
        Flowers102(self.data_dir, download=True)

    @staticmethod
    def normalization() -> Callable:
        return transform_lib.Normalize(
            mean=[x / 255.0 for x in [110.4, 97.4, 75.6]],
            std=[x / 255.0 for x in [75.1, 62.9, 69.7]],
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_transforms is None:
            transforms = default_train_transform(self.image_size, self.normalization)
        else:
            transforms = self.train_transforms

        dataset = Flowers102(self.data_dir, split='train', transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

    def val_dataloader(self) -> DataLoader:
        if self.val_transforms is None:
            transforms = default_val_transform(self.image_size, self.normalization)
        else:
            transforms = self.val_transforms

        dataset = Flowers102(self.data_dir, split='val', transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

    def test_dataloader(self) -> DataLoader:
        if self.val_transforms is None:
            transforms = default_val_transform(self.image_size, self.normalization)
        else:
            transforms = self.test_transforms

        dataset = Flowers102(self.data_dir, split='test', transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader


class OxfordIIITPetDataModule(VisionDataModule):
    name = "oxford-iit-pet"
    dataset_cls = OxfordIIITPet

    def __init__(self, *args, image_size: int = 224, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = image_size

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves files to data_dir.
        Splits in Oxford IIIT Pet are specified using `split` argument rather
        than `train` in torchvision, we need to override the default method.
        """
        self.dataset_cls(self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset.
        Splits in Oxford IIIT Pet are specified using `split` argument rather
        than `train` in torchvision, we need to override the default method.
        """
        if stage == "fit" or stage is None:
            if self.train_transforms is None:
                train_transforms = default_train_transform(self.image_size, self.normalization)
            else:
                train_transforms = self.train_transforms
            if self.val_transforms is None:
                val_transforms = default_val_transform(self.image_size, self.normalization)
            else:
                val_transforms = self.val_transforms

            dataset_train = self.dataset_cls(
                self.data_dir,
                split='trainval',
                transform=train_transforms,
                **self.EXTRA_ARGS,
            )
            dataset_val = self.dataset_cls(
                self.data_dir,
                split='trainval',
                transform=val_transforms,
                **self.EXTRA_ARGS,
            )

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            if self.test_transforms is None:
                test_transforms = default_val_transform(self.image_size, self.normalization)
            else:
                test_transforms = self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir,
                split='test',
                transform=test_transforms,
                **self.EXTRA_ARGS,
            )

    @staticmethod
    def normalization() -> Callable:
        return transform_lib.Normalize(
            mean=[x / 255.0 for x in [122., 113.7, 100.9]],
            std=[x / 255.0 for x in [68.3, 67.1, 68.8]],
        )

    @property
    def num_classes(self) -> int:
        return 37

    def default_transforms(self) -> Callable:
        return default_val_transform(self.image_size, self.normalization)
