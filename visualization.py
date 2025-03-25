
###########
# Imports #
###########

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms

# fix possible truncation during training
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

###############################
# Data Alignment Verification #
###############################

def verify_data_alignment(dataset, image_files, mask_files, num_samples=5):
    """
    verify data processing by displaying original images, original gt masks,
    preprocessed images, and processed masks side by side.

    args:
        dataset: dataset object that provides preprocessed images and masks.
        image_files: list of full paths to the original image files.
        mask_files: list of full paths to the original gt mask files.
        num_samples: number of samples to visualize.
    """
    for i in range(num_samples):
        # retrieve the preprocessed image and mask from the dataset
        preprocessed_image, processed_mask = dataset[i]

        # if the preprocessed image is a tensor, convert it to a pil image otherwise, keep as is
        if not isinstance(preprocessed_image, Image.Image):
            preprocessed_image_pil = transforms.ToPILImage()(preprocessed_image)
        else:
            preprocessed_image_pil = preprocessed_image

        # do the same for the processed mask
        if not isinstance(processed_mask, Image.Image):
            processed_mask_pil = transforms.ToPILImage()(processed_mask)
        else:
            processed_mask_pil = processed_mask

        # load the original image and original gt mask from disk
        original_image = Image.open(image_files[i]).convert("RGB")
        original_mask = Image.open(mask_files[i]).convert("L")

        # display the images in a 2x2 grid:
        # top row: original image and original gt mask
        # bottom row: preprocessed image and processed mask
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].imshow(original_image)
        axs[0, 0].set_title(f"original image {i + 1}")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(original_mask, cmap="gray")
        axs[0, 1].set_title(f"original gt mask {i + 1}")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(preprocessed_image_pil)
        axs[1, 0].set_title(f"preprocessed image {i + 1}")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(processed_mask_pil, cmap="gray")
        axs[1, 1].set_title(f"processed mask {i + 1}")
        axs[1, 1].axis("off")

        plt.tight_layout()
        plt.show()

def visualize_batch_from_loader(loader, num_batches=1):
    """
    visualize a few batches of images and masks from the dataloader.

    args:
        loader: pytorch dataloader providing images and masks.
        num_batches: number of batches to visualize.
    """
    loader_iter = iter(loader)
    for batch_idx in range(num_batches):

        # get the next batch of images and masks
        images, masks = next(loader_iter)

        # move tensors to cpu if necessary and convert to numpy arrays
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()
        masks_np = masks.squeeze(1).cpu().numpy()  # remove channel dimension for masks

        # display images and masks side by side
        batch_size = images_np.shape[0]
        fig, axes = plt.subplots(
            batch_size, 2, figsize=(10, 5 * batch_size)
        ) if batch_size > 1 else plt.subplots(1, 2, figsize=(10, 5))

        # ensure axes is always a 2d array for consistency
        if batch_size == 1:
            axes = np.expand_dims(axes, axis=0)
        for i in range(batch_size):
            axes[i, 0].imshow(images_np[i])
            axes[i, 0].set_title(f"image {batch_idx * batch_size + i + 1}")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(masks_np[i], cmap="gray")
            axes[i, 1].set_title(f"gt mask {batch_idx * batch_size + i + 1}")
            axes[i, 1].axis("off")
        plt.tight_layout()
        plt.show()

def visualize_validation_metrics(val_metrics):
    """
    Visualize validation metrics (loss, IoU, Dice, pixel accuracy) over epochs.

    Args:
        val_metrics (dict): Dictionary of validation metrics with keys 'loss', 'iou', 'dice', 'pixel_acc'.
    """
    epochs = range(1, len(val_metrics['loss']) + 1)

    # create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Validation Metrics per Epoch', fontsize=16)

    # loss

    # plot the validation loss
    axs[0, 0].plot(epochs, val_metrics['loss'],label='Validation Loss',marker='o',color='orange')

    # set the title, labels, and legend
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # iou

    # plot the validation IoU
    axs[0, 1].plot(epochs, val_metrics['iou'], label='Validation IoU',marker='o', color='orange')

    # set the title, labels, and legend
    axs[0, 1].set_title('IoU')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('IoU')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # dice

    # plot the validation Dice score
    axs[1, 0].plot(epochs, val_metrics['dice'], label='Validation Dice', marker='o', color='orange')

    # set the title, labels, and legend
    axs[1, 0].set_title('Dice Score')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Dice Score')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # pixel Accuracy

    # plot the validation pixel accuracy
    axs[1, 1].plot(epochs, val_metrics['pixel_acc'], label='Validation Pixel Accuracy', marker='o', color='orange')

    # set the title, labels, and legend
    axs[1, 1].set_title('Pixel Accuracy')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Pixel Accuracy')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # clean up the layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def visualize_training_metrics(train_metrics):
    """
    Visualize training metrics (loss, IoU, Dice,pixel accuracy) over epochs.

    Args:
        train_metrics(dict): Dictionary of training metrics with keys'loss', 'iou', 'dice', 'pixel_acc'.
    """
    epochs = range(1, len(train_metrics['loss']) + 1)

    # create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Metrics per Epoch', fontsize=16)

    # loss

    # plot the training loss
    axs[0, 0].plot(epochs, train_metrics['loss'], label='Train Loss', marker='o')

    # set the title, labels, and legend
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # ioU

    # plot the training IoU
    axs[0, 1].plot(epochs, train_metrics['iou'],label='Train IoU',marker='o')

    # set the title, labels, and legend
    axs[0, 1].set_title('IoU')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('IoU')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # dice

    # plot the training Dice score
    axs[1, 0].plot(epochs, train_metrics['dice'],label='Train Dice',marker='o')

    # set the title, labels, and legend
    axs[1, 0].set_title('Dice Score')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Dice Score')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # pixel Accuracy

    # plot the training pixel accuracy
    axs[1, 1].plot(epochs, train_metrics['pixel_acc'],label='Train Pixel Accuracy', marker='o')

    # set the title, labels, and legend
    axs[1, 1].set_title('Pixel Accuracy')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Pixel Accuracy')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # clean up the layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

