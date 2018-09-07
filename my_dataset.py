import os
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as tv_transforms
from torch.utils import data as D


class AugDataset(D.Dataset):
    """
    A customized dataset to add augmentation transforms.

    Augmentations are assumed to be instances of albumentations transforms
    """
    def __init__(self, root='', aug=None, train=True, debug=False):
        """ Intialize the dataset
        """
        self.root = root
        self.train = train
        self.aug = aug
        self.debug = debug
        self._make_tranforms_fn()
        self._load_data()

        self.len = self.img_ids.shape[0]

    def _load_data(self):
        self.depths = pd.read_csv(os.path.join(self.root, 'depths.csv'), dtype=str).fillna('')

        if self.train:
            self.img_ids = pd.read_csv(os.path.join(self.root, 'train.csv'), dtype=str).fillna('')
            self.img_dir = 'train_images'
        else:
            self.img_ids = pd.read_csv(os.path.join(self.root, 'sample_submission.csv'), dtype=str).fillna('')
            self.img_dir = 'test_images'

        self.img_ids = pd.merge(self.img_ids, self.depths, how='left')

        self.images = self.img_ids['id'].apply(
            lambda img_id: cv2.imread(
                os.path.join(self.root, self.img_dir, '{}.png'.format(img_id))).max(axis=-1, keepdims=1)[None, :]
        )
        self.images = np.concatenate(self.images.values, axis=0)

        if self.train:
            self.masks = self.img_ids['id'].apply(
                lambda img_id: cv2.imread(
                    os.path.join(
                        self.root, 'train_masks', '{}.png'.format(img_id))).max(axis=-1, keepdims=1)[None, :]
            )
            self.masks = np.concatenate(self.masks.values, axis=0)

    def _make_tranforms_fn(self):
        if self.aug is not None:
            self.transform = self.aug
        else:
            self.transform = tv_transforms.ToTensor()

    def __getitem__(self, index):
        """
        Get a sample from the dataset, apply transformations, convert to a tensor
        """
        image = self.images[index]
        if self.train:
            mask = self.masks[index]

        if self.aug is not None:
            # make a dict to pass to the augmenter
            if self.train:
                data = {"image": np.array(image), "mask": mask}
            else:
                data = {"image": np.array(image)}

            transformed = self.transform(**data)

            if self.debug:
                print(transformed['image'].shape)
            image = (transformed['image'] / 255).transpose(2, 0, 1).astype(np.float32)

            if self.train:
                # add mask
                if self.debug:
                    print(transformed['mask'].shape)
                mask = (transformed['mask'] / 255).transpose(2, 0, 1).astype(np.float32)

                return image, mask
            else:
                return image

        else:
            # easy pass
            if self.train:
                return self.transform(image), self.transform(mask)
            return self.transform(image)

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


class AugDatasetDual(D.Dataset):
    """
    A customized data loader to use with dual target, one is the actual mask and another
    is the original image (restoration by some sort of autoencoder).

    Augmentations are assumed to be instances of albumentations transforms

    If input image is transformed by adding noise, occlusions, etc.. the target image
    should not have those transformations. At the same time if it is a geometrical
    transformation, the target image should be transformed as well. Because of this target
    image should be added separately as an additional field in the data dict and "image_only"
    transformation should be applied to it.
    """
    def __init__(self, root='', aug=None, add_img_to_target=False, debug=False):
        """ Intialize the dataset
        """
        self.root = root
        self.aug = aug
        self.add_img_to_target = add_img_to_target
        self.debug = debug
        self._make_tranforms_fn()
        self._load_data()

        self.len = self.img_ids.shape[0]

    def _load_data(self):
        self.depths = pd.read_csv(os.path.join(self.root, 'depths.csv'), dtype=str).fillna('')

        self.img_ids_tr = pd.read_csv(os.path.join(self.root, 'train.csv'), dtype=str).fillna('')
        self.img_dir_tr = 'train_images'
        self.img_ids_te = pd.read_csv(os.path.join(self.root, 'sample_submission.csv'), dtype=str).fillna('')
        self.img_dir_te = 'test_images'

        self.img_ids = pd.merge(pd.concat([self.img_ids_tr, self.img_ids_te], axis=0), self.depths, how='left')

        self.images_tr = self.img_ids_tr['id'].apply(
            lambda img_id: cv2.imread(
                os.path.join(self.root, self.img_dir_tr, '{}.png'.format(img_id))).max(axis=-1, keepdims=1)[None, :]
        )
        self.images_te = self.img_ids_te['id'].apply(
            lambda img_id: cv2.imread(
                os.path.join(self.root, self.img_dir_te, '{}.png'.format(img_id))).max(axis=-1, keepdims=1)[None, :]
        )
        self.images_tr = np.concatenate(self.images_tr.values, axis=0)
        self.images_te = np.concatenate(self.images_te.values, axis=0)
        self.images = np.concatenate([self.images_tr, self.images_te], axis=0)

        self.masks = self.img_ids_tr['id'].apply(
            lambda img_id: cv2.imread(
                os.path.join(
                    self.root, 'train_masks', '{}.png'.format(img_id))).max(axis=-1, keepdims=1)
        )
        self.masks = self.masks.tolist()
        self.max_mask_ix = len(self.masks)
        self.default_mask = np.zeros_like(self.masks[0])

    def _make_tranforms_fn(self):
        if self.aug is not None:
            self.transform = self.aug
        else:
            self.transform = tv_transforms.ToTensor()

    def __getitem__(self, index):

        """ Get a sample from the dataset
        """
        image = self.images[index]
        has_mask = index < self.max_mask_ix
        mask = self.masks[index] if has_mask else self.default_mask

        if self.aug is not None:
            data = {"image": np.array(image), "mask": mask}
            if self.add_img_to_target:
                data['target_image'] = image

            transformed = self.transform(**data)

            if self.debug:
                print("image shape", transformed['image'].shape)
            image = (transformed['image'] / 255).transpose(2, 0, 1).astype(np.float32)

            if self.debug:
                print("mask shape", transformed['mask'].shape)
            mask = (transformed['mask'] / 255).transpose(2, 0, 1).astype(np.float32)

            if self.add_img_to_target:
                if self.debug:
                    print("target image shape", transformed['target_image'].shape)
                target_image = (transformed['target_image'] / 255).transpose(2, 0, 1).astype(np.float32)
                return image, (mask, has_mask, target_image)
            else:
                return image, (mask, has_mask)
        elif self.add_img_to_target:
            return self.transform(image), (self.transform(mask), has_mask, self.transform(image))
        else:
            return self.transform(image), (self.transform(mask), has_mask)

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len