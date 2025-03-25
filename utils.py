###########
# Imports #
###########

import os
import gc
import torch
import segmentation_models_pytorch as smp
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# fix possible truncation during training
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True




################################
# Utility and set up functions #
################################
def clear_model_and_cache():
    """
    Utility function to delete existing model and optimizer objects
    and clear GPU memory to avoid memory leaks.
    """
    if 'model' in globals():
        print("Deleting existing model...")
        del globals()['model']
        del globals()['optimizer']
    gc.collect()
    torch.cuda.empty_cache()

def load_image_processor_mask2former(pretrained_model_name, do_rescale=True, do_normalize=False, do_resize=True):
    """
    Load the Mask2Former image processor with specified settings.

    Parameters:
    - pretrained_model_name: Hugging Face model name.
    - do_rescale: Whether to rescale image values.
    - do_normalize: Whether to normalize image values.
    - do_resize: Whether to resize the images.

    Returns:
    - image_processor: The configured image processor object.
    """
    image_processor = Mask2FormerImageProcessor.from_pretrained(
        pretrained_model_name,
        do_rescale=do_rescale,
        do_normalize=do_normalize,
        do_resize=do_resize
    )
    return image_processor

def load_mask2former_model(pretrained_model_name, num_labels, ignore_mismatched_sizes=True, freeze_encoder=True):
    """
    Load the Mask2Former model for universal segmentation.

    Parameters:
    - pretrained_model_name: Hugging Face model name.
    - num_labels: Number of segmentation labels (e.g., 2 for binary segmentation).
    - ignore_mismatched_sizes: Whether to allow resizing of model parameters.
    - freeze_encoder: Whether to freeze the encoder backbone.

    Returns:
    - model: The Mask2Former model object.
    """
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=ignore_mismatched_sizes
    )

    if freeze_encoder:
        # freeze the encoder backbone
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False

    return model

def print_trainable_layers_mask2former(model):
    """
    Display which layers of the model are trainable or frozen.

    Parameters:
    - model: The Mask2Former model.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is frozen")

def save_model_and_processor_mask2former(model, image_processor, save_dir):
    """
    Save the trained model and image processor to a specified directory.

    Args:
        model: The trained Mask2Former model.
        image_processor: The corresponding image processor used for preprocessing.
        save_dir: Path to the directory where the model and processor will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save_pretrained(save_dir)
    image_processor.save_pretrained(save_dir)
    print(f"Model and image processor saved to {save_dir}")


def load_model_and_processor_mask2former(save_dir):
    """
    Load a trained Mask2Former model and its corresponding image processor from a directory.

    Args:
        save_dir: Path to the directory where the model and processor are stored.

    Returns:
        model: The loaded Mask2Former model.
        image_processor: The loaded image processor.
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Directory {save_dir} does not exist.")

    model = Mask2FormerForUniversalSegmentation.from_pretrained(save_dir)
    image_processor = Mask2FormerImageProcessor.from_pretrained(save_dir)
    print(f"Model and image processor loaded from {save_dir}")
    return model, image_processor

################################
# Utility Functions for Unet++ #
################################


def clear_model_and_cache():
    """
    utility function to delete existing model and optimizer objects
    and clear gpu memory to avoid memory leaks.
    """
    if 'model' in globals():
        print("deleting existing model...")
        del globals()['model']
        del globals()['optimizer']
    gc.collect()
    torch.cuda.empty_cache()


def load_image_processor_unet(encoder_name, do_rescale=True, do_normalize=False, do_resize=True):
    """
    dummy image processor for unet++.
    not used in unet++, returns none.
    """
    return None


def load_unetpp_model_unet(encoder_name, num_labels, freeze_encoder=True):
    """
    load the unet++ model for segmentation.

    parameters:
    - encoder_name: name of the encoder to use (e.g., 'resnet34').
    - num_labels: number of segmentation labels (e.g., 1 for binary segmentation).
    - freeze_encoder: whether to freeze the encoder backbone.

    returns:
    - model: the unet++ model object.
    """

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_labels,
        activation=None
    )
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    return model


def print_trainable_layers_unet(model):
    """
    display which layers of the model are trainable or frozen.

    parameters:
    - model: the unet++ model.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is frozen")


def save_model_and_processor_unet(model, image_processor, save_dir):
    """
    save the trained model to a specified directory.

    args:
    - model: the trained unet++ model.
    - image_processor: not used in unet++ (pass none).
    - save_dir: path to the directory where the model will be saved.
    """
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    print(f"model saved to {save_dir}")


def load_model_and_processor_unet(save_dir, encoder_name, num_labels):
    """
    load a trained unet++ model from a directory.

    args:
    - save_dir: path to the directory where the model is stored.
    - encoder_name: name of the encoder used.
    - num_labels: number of segmentation labels.

    returns:
    - model: the loaded unet++ model.
    - image_processor: none.
    """
    import os
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"directory {save_dir} does not exist.")

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=num_labels,
        activation=None
    )
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pth"), map_location="cpu"))
    print(f"model loaded from {save_dir}")
    return model, None