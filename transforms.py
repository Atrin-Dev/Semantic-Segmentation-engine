import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from config import ENCODER, ENCODER_WEIGHTS, CROP_SIZE

preprocess = get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def get_train_transforms():
    return A.Compose([
        A.RandomCrop(CROP_SIZE, CROP_SIZE),
        A.Lambda(name="preprocessing", image=preprocess),
        A.Lambda(name="to_tensor", image=lambda x, **kwargs: x.transpose(2, 0, 1).astype("float32")),
    ])

def get_eval_transforms():
    return A.Compose([
        A.Lambda(name="preprocessing", image=preprocess),
        A.Lambda(name="to_tensor", image=lambda x, **kwargs: x.transpose(2, 0, 1).astype("float32")),
    ])
