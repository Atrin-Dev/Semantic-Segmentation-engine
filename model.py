import segmentation_models_pytorch as smp
from config import NUM_CLASSES


def build_model(model_name="deeplabv3plus", encoder_name="resnet34", encoder_weights="imagenet"):
    if model_name.lower() == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    elif model_name.lower() == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from ['deeplabv3plus', 'unet'].")

    return model
