###########
# Imports #
###########

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

# fix possible truncation during training
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True





############################
# Inference and Evaluation #
############################

def calculate_iou_infer(pred, target):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    Treat cases with zero union as perfect matches.

    Args:
        pred (numpy.ndarray): Predicted binary mask.
        target (numpy.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score.
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:  # if union is zero, treat as a perfect match
        return 1.0

    return intersection / (union + 1e-6)  # calculate IoU

def calculate_dice_infer(pred, target):
    """
    Calculate Dice coefficient between two binary masks.
    Treat cases with zero denominator as perfect matches.

    Args:
        pred (numpy.ndarray): Predicted binary mask.
        target (numpy.ndarray): Ground truth binary mask.

    Returns:
        float: Dice coefficient score.
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()

    if denominator == 0:  # ff denominator is zero, treat as a perfect match
        return 1.0

    return (2 * intersection) / (denominator + 1e-6)  # calculate Dice

def infer_and_display_mask2former(model, image_processor, dataloader, device, num_samples=5, target_size=(128, 128)):
    """
    Perform inference and display the original image, ground truth mask, and predicted mask side by side.
    Additionally, calculate IoU, Dice, and Pixel Accuracy.

    Args:
        model: The trained segmentation model.
        image_processor: Preprocessing module for the input images.
        dataloader: DataLoader providing images and ground truth masks.
        device: Computation device (CPU or CUDA).
        num_samples: Number of samples to visualize.
        target_size: Target size for resizing masks during visualization.
    """
    model.eval()
    samples_displayed = 0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    num_evaluated = 0

    with torch.no_grad():
        for images, ground_truth_masks in tqdm(dataloader, desc="Inferencing"):
            # move ground truth masks to device and convert to float
            ground_truth_masks = ground_truth_masks.to(device, dtype=torch.float32)

            # convert tensors to PIL images for the image processor
            pil_images = [transforms.ToPILImage()(img) for img in images]
            inputs = image_processor(images=pil_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            # forward pass
            outputs = model(pixel_values=pixel_values)
            tissue_logits = outputs.masks_queries_logits[:, 1]  # binary segmentation logits

            # resize logits to match mask size
            tissue_logits_resized = torch.sigmoid(F.interpolate(
                tissue_logits.unsqueeze(1),
                size=ground_truth_masks.shape[-2:],  # match ground truth mask size
                mode="bilinear",
                align_corners=False
            ).squeeze(1))  # remove channel dimension

            # convert predicted logits to binary masks
            predicted_masks = (tissue_logits_resized > 0.5).cpu().numpy().astype(np.uint8)

            # display the first few samples
            for i in range(len(images)):
                if samples_displayed >= num_samples:
                    # print the average metrics
                    avg_iou = total_iou / num_evaluated
                    avg_dice = total_dice / num_evaluated
                    avg_pixel_acc = total_pixel_acc / num_evaluated
                    print(f"\nMean IoU: {avg_iou:.4f}")
                    print(f"Mean Dice: {avg_dice:.4f}")
                    print(f"Mean Pixel Accuracy: {avg_pixel_acc:.4f}")
                    return

                # convert images, ground truths, and predictions to displayable formats
                original_image = pil_images[i]
                ground_truth_mask = ground_truth_masks[i].cpu().numpy().squeeze()  # remove extra dimensions
                predicted_mask = predicted_masks[i]

                # ensure ground truth mask is binary (0 or 1) and scaled to 255
                ground_truth_mask = (ground_truth_mask > 0.5).astype(np.uint8) * 255

                # calculate IoU, Dice, and Pixel Accuracy for the current sample
                iou = calculate_iou_infer(predicted_mask, ground_truth_mask // 255)  # Divide by 255 for binary comparison
                dice = calculate_dice_infer(predicted_mask, ground_truth_mask // 255)  # Divide by 255 for binary comparison
                pixel_acc = np.mean(predicted_mask == (ground_truth_mask // 255))
                total_iou += iou
                total_dice += dice
                total_pixel_acc += pixel_acc
                num_evaluated += 1

                # resize ground truth and predicted masks for visualization
                ground_truth_mask_resized = np.array(Image.fromarray(ground_truth_mask).resize(target_size, Image.NEAREST))
                predicted_mask_resized = np.array(Image.fromarray((predicted_mask * 255).astype(np.uint8)).resize(target_size, Image.NEAREST))

                # display side by side
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(original_image)
                axs[0].set_title(f"Original Image {samples_displayed + 1}")
                axs[0].axis("off")

                axs[1].imshow(ground_truth_mask_resized, cmap="gray", vmin=0, vmax=255)
                axs[1].set_title(f"Ground Truth Mask {samples_displayed + 1}")
                axs[1].axis("off")

                axs[2].imshow(predicted_mask_resized, cmap="gray", vmin=0, vmax=255)
                axs[2].set_title(f"Predicted Mask {samples_displayed + 1}")
                axs[2].axis("off")

                plt.show()
                samples_displayed += 1

    # print the final average metrics
    avg_iou = total_iou / num_evaluated if num_evaluated > 0 else 0
    avg_dice = total_dice / num_evaluated if num_evaluated > 0 else 0
    avg_pixel_acc = total_pixel_acc / num_evaluated if num_evaluated > 0 else 0
    print(f"\nMean IoU: {avg_iou:.4f}")
    print(f"Mean Dice: {avg_dice:.4f}")
    print(f"Mean Pixel Accuracy: {avg_pixel_acc:.4f}")





###################################
# Inferencing Adapated for Unet++ #
###################################


def infer_and_display_unet(model, image_processor, dataloader, device, num_samples=5, target_size=(128, 128)):
    """
    perform inference and display the original image, ground truth mask, and predicted mask side by side.
    additionally, calculate iou, dice, and pixel accuracy.

    args:
      model: the trained segmentation model.
      image_processor: not used in unet++ (pass none).
      dataloader: dataloader providing images and ground truth masks.
      device: computation device (cpu or cuda).
      num_samples: number of samples to visualize.
      target_size: target size for resizing masks during visualization.
    """
    model.eval()
    samples_displayed = 0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    num_evaluated = 0

    with torch.no_grad():
        for images, ground_truth_masks in tqdm(dataloader, desc="inferencing"):
            ground_truth_masks = ground_truth_masks.to(device, dtype=torch.float32)
            # unet++ does not require an image processor; use images directly
            inputs = images.to(device, dtype=torch.float32)
            outputs = model(inputs)
            logits_resized = torch.sigmoid(F.interpolate(outputs, size=ground_truth_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1))
            predicted_masks = (logits_resized > 0.5).cpu().numpy().astype(np.uint8)
            pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]

            for i in range(len(images)):
                if samples_displayed >= num_samples:
                    avg_iou = total_iou / num_evaluated
                    avg_dice = total_dice / num_evaluated
                    avg_pixel_acc = total_pixel_acc / num_evaluated
                    print(f"\nmean iou: {avg_iou:.4f}")
                    print(f"mean dice: {avg_dice:.4f}")
                    print(f"mean pixel accuracy: {avg_pixel_acc:.4f}")
                    return

                original_image = pil_images[i]
                ground_truth_mask = ground_truth_masks[i].cpu().numpy().squeeze()
                predicted_mask = predicted_masks[i]
                ground_truth_mask = (ground_truth_mask > 0.5).astype(np.uint8) * 255

                iou = calculate_iou_infer(predicted_mask, ground_truth_mask // 255)
                dice = calculate_dice_infer(predicted_mask, ground_truth_mask // 255)
                pixel_acc = np.mean(predicted_mask == (ground_truth_mask // 255))
                total_iou += iou
                total_dice += dice
                total_pixel_acc += pixel_acc
                num_evaluated += 1

                ground_truth_mask_resized = np.array(Image.fromarray(ground_truth_mask).resize(target_size, Image.NEAREST))
                predicted_mask_resized = np.array(Image.fromarray((predicted_mask * 255).astype(np.uint8)).resize(target_size, Image.NEAREST))

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(original_image)
                axs[0].set_title(f"original image {samples_displayed + 1}")
                axs[0].axis("off")

                axs[1].imshow(ground_truth_mask_resized, cmap="gray", vmin=0, vmax=255)
                axs[1].set_title(f"ground truth mask {samples_displayed + 1}")
                axs[1].axis("off")

                axs[2].imshow(predicted_mask_resized, cmap="gray", vmin=0, vmax=255)
                axs[2].set_title(f"predicted mask {samples_displayed + 1}")
                axs[2].axis("off")

                plt.show()
                samples_displayed += 1

    avg_iou = total_iou / num_evaluated if num_evaluated > 0 else 0
    avg_dice = total_dice / num_evaluated if num_evaluated > 0 else 0
    avg_pixel_acc = total_pixel_acc / num_evaluated if num_evaluated > 0 else 0
    print(f"\nmean iou: {avg_iou:.4f}")
    print(f"mean dice: {avg_dice:.4f}")
    print(f"mean pixel accuracy: {avg_pixel_acc:.4f}")