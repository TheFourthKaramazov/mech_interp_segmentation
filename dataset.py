###########
# Imports #
###########


import os
import numpy as np
from PIL import Image, ImageFilter, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# fix possible truncation during training
ImageFile.LOAD_TRUNCATED_IMAGES = True


###################
# Data Processing #
###################


def gaussian_blur(image, radius=2):
    """Apply Gaussian blur to an image to avoid capturing noise as tissue."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess an image: apply Gaussian blur and resize."""
    # apply blur to reduce noise
    image = Image.open(image_path).convert("RGB")
    blurred_image = gaussian_blur(image, 2)

    # resize to target size
    resized_image = blurred_image.resize(target_size, Image.BICUBIC)
    return resized_image

def preprocess_mask(mask_path, target_size=(128, 128)):
    """Convert mask to binary format and resize."""
    # convert mask to grayscale
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)

    # normalize binary mask (0 or 1)
    binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)
    binary_mask = Image.fromarray(binary_mask * 255)

    # resize using nearest-neighbor
    resized_mask = binary_mask.resize(target_size, Image.NEAREST)
    return resized_mask

class ISICDataset(Dataset):
    """
    Dataset class for loading ISIC 2018 images and masks.
    """
    def __init__(self, image_files, mask_files, image_processor=None, mask_transform=None):
        """
        Initialize the dataset.

        Parameters:
        - image_files: List of image file paths.
        - mask_files: List of mask file paths.
        - image_processor: Optional preprocessing function for images.
        - mask_transform: Optional preprocessing function for masks.
        """
        self.image_files = image_files
        self.mask_files = mask_files
        self.image_processor = image_processor
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # load image and mask
        image = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx]).convert("L")

        # preprocess image and mask
        image = preprocess_image(self.image_files[idx])
        mask = preprocess_mask(self.mask_files[idx])

        # apply image processor if provided
        if self.image_processor:
            image = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            image = image_transform(image)

        # apply mask transform if provided
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

def get_file_paths(image_folder, mask_folder, valid_extensions=(".jpg", ".png", ".jpeg")):
    """Retrieve valid image and mask file paths while excluding non-image files."""
    # collect valid image files
    image_files = sorted([
        os.path.join(image_folder, f) for f in os.listdir(image_folder)
        if f.lower().endswith(valid_extensions)
    ])

    # collect valid mask files
    mask_files = sorted([
        os.path.join(mask_folder, f) for f in os.listdir(mask_folder)
        if f.lower().endswith(valid_extensions)
    ])

    assert len(image_files) == len(mask_files), "Mismatch between image and mask files."
    return image_files, mask_files

image_transform = transforms.Compose([
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
])

def create_dataloaders(data_root, batch_size=16, image_processor=None, mask_transform=mask_transform):
    """
    Create train, validation, and test dataloaders using the provided dataset structure.
    """
    # define folder paths
    train_img_folder = os.path.join(data_root, "ISIC2018_Training_Input")
    train_mask_folder = os.path.join(data_root, "ISIC2018_Training_GroundTruth")
    val_img_folder = os.path.join(data_root, "ISIC2018_Validation_Input")
    val_mask_folder = os.path.join(data_root, "ISIC2018_Validation_GroundTruth")
    test_img_folder = os.path.join(data_root, "ISIC2018_Test_Input")
    test_mask_folder = os.path.join(data_root, "ISIC2018_Test_GroundTruth")

    # retrieve file paths
    train_images, train_masks = get_file_paths(train_img_folder, train_mask_folder)
    val_images, val_masks = get_file_paths(val_img_folder, val_mask_folder)
    test_images, test_masks = get_file_paths(test_img_folder, test_mask_folder)

    # create datasets
    train_dataset = ISICDataset(train_images, train_masks, image_processor, mask_transform)
    val_dataset = ISICDataset(val_images, val_masks, image_processor, mask_transform)
    test_dataset = ISICDataset(test_images, test_masks, image_processor, mask_transform)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, train_images, val_images, test_images, train_masks, val_masks, test_masks


