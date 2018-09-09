import albumentations as aau
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


# utils and fixes

def get_augs_meta(obj):
    """Return nested list with some nice looking names for a composition of transformations"""
    if hasattr(obj, 'transforms'):
        return [get_augs_meta(t) for t in obj.transforms]
    else:
        return obj.__module__ + '.' + obj.__class__.__name__


class CorrectDims(aau.DualTransform):
    def __init__(self):
        super().__init__(1)

    def apply(self, image, **params):
        if image.ndim == 2:
            image = image[..., None]
        return image


def add_target_field(cls, new_img_only_field=None):
    """
    Given a class, add/modify property `targets`
    This is needed to add additional fields to be transformed

    new_img_only_field can be a string or a list of strings
    """
    if isinstance(new_img_only_field, str):
        new_img_only_field = [new_img_only_field]
    elif isinstance(new_img_only_field, (list, tuple)):
        pass
    else:
        return cls

    class NewClass(cls):
        __module__ = cls.__module__
        __name__ = getattr(cls, '__name__', cls.__repr__)
        @property
        def targets(self):
            res = {'image': self.apply,
                   'mask': self.apply_to_mask,
                   'bboxes': self.apply_to_bboxes}
            for f in new_img_only_field:
                res[f] = self.apply
            return res

    return NewClass


# additional augmentations

class IAACrop(aau.DualIAATransform):
    def __init__(self, px, p=0.5):
        super().__init__(p)
        self.processor = iaa.Crop(px=px)


class IAAAffine(aau.DualIAATransform):
    def __init__(self, scale=1.0, translate_percent=None, translate_px=None,
                 rotate=0.0, shear=0.0, order=3, cval=0, mode='reflect', p=0.5):
        super().__init__(p)
        self.processor = iaa.Affine(scale=scale, translate_percent=translate_percent, translate_px=translate_px,
                                    rotate=rotate, shear=shear, order=order, cval=cval, mode=mode)


class IAAContrastNormalization(aau.ImageOnlyIAATransform):
    def __init__(self, alpha=1.0, p=0.5):
        super().__init__(p)
        self.processor = iaa.ContrastNormalization(alpha=alpha)


class CShift(iaa.Augmenter):
    def __init__(self, val=None, norm=True, name=None, deterministic=False, random_state=None):
        super().__init__(name, deterministic, random_state)
        if ia.is_single_number(val):
            self.val = ia.parameters.Binomial(val)
        elif ia.is_iterable(val) and len(val) == 2:
            self.val = ia.parameters.Uniform(*val)
        elif isinstance(val, ia.parameters.StochasticParameter):
            self.val = val
        self.norm = norm

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.val.draw_samples((nb_images,), random_state=random_state)
        images = np.array(images)/255
        images = np.concatenate([images[i][None, :,:,:] + samples[i] for i in range(nb_images)], axis=0)
        images[images>1] = 2 - images[images>1]
        images[images<0] = -images[images<0]

        if self.norm:
            images = images + 0.5 - images.mean(axis=-1, keepdims=1).mean(axis=-2, keepdims=1)
            images[images > 1] = 1.
            images[images < 0] = 0.

        images = (images * 255).astype(np.uint8)
        return images
    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise NotImplementedError()
    def _augment_heatmaps(self, keypoints_on_images, random_state, parents, hooks):
        raise NotImplementedError()
    def get_parameters(self):
        return [self.val]


class IAACShift(aau.ImageOnlyIAATransform):
    def __init__(self, val=None, p=0.5):
        super().__init__(p)
        self.processor = CShift(val=val)


class FFTMask(iaa.Augmenter):
    def __init__(self, mask_type='square', abover_th=0, below_th=1, invert=False,
                 deterministic=False, random_state=None, name='FFTMask'):
        """
        Apply round or square (diamond), sharp or gaussian, mask in frequency domain

        Can be a ring masked or ring left

        Parameters
        ----------
        mask_type
            possible values
                - 'square' will create a diamond shaped mask with sharp edges,
                - 'square_gauss' - will create diamond shaped mask as gaussian form the center
                - '' - round mask with sharp edges
                - 'gauss' - round mask as gaussian form the center

        abover_th
            leaves values above this distance from center (0,0) freq, if gaussian mask, that would be sigma

        below_th
            leaves values below this distance from center (0,0) freq, if gaussian mask, that would be sigma

        invert
            after making mask, invert values: mask = 1 - mask

        deterministic
            passed to parent

        random_state
            passed to parent

        name
            passed to parent
        """
        super().__init__(name, deterministic, random_state)

        if ia.is_single_number(abover_th):
            self.abover_th = ia.parameters.Binomial(abover_th)
        elif ia.is_iterable(abover_th) and len(abover_th) == 2:
            self.abover_th = ia.parameters.Uniform(*abover_th)
        elif isinstance(abover_th, ia.parameters.StochasticParameter):
            self.abover_th = abover_th

        if ia.is_single_number(below_th):
            self.below_th = ia.parameters.Binomial(below_th)
        elif ia.is_iterable(below_th) and len(below_th) == 2:
            self.below_th = ia.parameters.Uniform(*below_th)
        elif isinstance(below_th, ia.parameters.StochasticParameter):
            self.below_th = below_th

        self.mask_type = mask_type
        self.invert = invert

    def _augment_images(self, images, random_state, parents, hooks):
        # get number of images, their size
        # and number of dimentions (are they with channels or grayscale)
        nb_images = len(images)
        img_size = images[0].shape[:2]
        img_ndim = images[0].ndim

        # get params samples per image
        below_ths = self.below_th.draw_samples((nb_images,), random_state=random_state)
        above_ths = self.abover_th.draw_samples((nb_images,), random_state=random_state)

        # make distance-from-center matrix
        mg = np.meshgrid(np.linspace(-1, 1, img_size[0]), np.linspace(-1, 1, img_size[1]))
        if self.mask_type.startswith('square'):
            d = (np.abs(mg[0]) ** 1 + np.abs(mg[1]) ** 1)
        else:
            d = np.sqrt(np.abs(mg[0]) ** 2 + np.abs(mg[1]) ** 2)

        # make masks for each image
        masks = []
        for img, b_th, a_th in zip(images, below_ths, above_ths):
            if self.mask_type.endswith('gauss'):
                fft_mask = np.exp(-(d ** 2 / 2 / (b_th + 1e-3) ** 2))
                if a_th > 0:
                    fft_mask *= (1 - np.exp(-(d ** 2 / 2 / a_th ** 2)))
                if self.invert:
                    fft_mask = 1 - fft_mask
            else:
                fft_mask = ((d <= b_th) & (d >= a_th))
                if self.invert:
                    fft_mask = ~fft_mask
                fft_mask = fft_mask * 1

            if (fft_mask ** 2).sum() == 0:
                fft_mask += 1
            # else:
            #     fft_mask /= fft_mask.mean()

            masks.append(fft_mask)

        # concat everything for performance
        images_arr = np.concatenate([img[None, ...] for img in images], axis=0) / 255
        masks = np.concatenate([mask[None, ...] for mask in masks], axis=0)
        if img_ndim == 3:
            masks = masks[..., None]

        # do fft transform, apply mask, transform back
        X_fft = np.fft.fft2(images_arr, axes=(1, 2))
        X_fft = np.fft.fftshift(X_fft, axes=(1, 2))
        X_fft = X_fft * masks
        X_fft = np.fft.ifftshift(X_fft, axes=(1, 2))
        images_arr = np.fft.ifft2(X_fft, axes=(1, 2))

        # to real space and rescale back to uint8
        images = np.real(images_arr)
        images_min = images.min(axis=1, keepdims=1).min(axis=2, keepdims=1)
        images_max = images.max(axis=1, keepdims=1).max(axis=2, keepdims=1)
        images = (images - images_min) / (images_max - images_min)
        images = (images * 255).astype(np.uint8)
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise NotImplementedError()

    def _augment_heatmaps(self, keypoints_on_images, random_state, parents, hooks):
        raise NotImplementedError()

    def get_parameters(self):
        return [self.abover_th, self.below_th]


class IAAFFTMask(aau.ImageOnlyIAATransform):
    def __init__(self, mask_type='square', abover_th=0, below_th=1, invert=False, p=0.5):
        super().__init__(p)
        self.processor = FFTMask(mask_type=mask_type, abover_th=abover_th, below_th=below_th, invert=invert)
