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
    def __init__(self, val=None, name=None, deterministic=False, random_state=None):
        super().__init__(name, deterministic, random_state)
        if ia.is_single_number(val):
            self.val = ia.parameters.Binomial(val)
        elif ia.is_iterable(val) and len(val) == 2:
            self.val = ia.parameters.Uniform(*val)
        elif isinstance(val, ia.parameters.StochasticParameter):
            self.val = val

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.val.draw_samples((nb_images,), random_state=random_state)
        images = images/255
        images = np.concatenate([images[i][None, :,:,:] + samples[i] for i in range(nb_images)], axis=0)
        images[images>1] = 2 - images[images>1]
        images[images<0] = -images[images<0]
        images = (images * 255).astype(np.uint8)
        return images
    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise NotImplementedError()
    def _augment_heatmaps(self, keypoints_on_images, random_state, parents, hooks):
        raise NotImplementedError()
    def get_parameters(self):
        return [self.val]
