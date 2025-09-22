#!/usr/bin/env python3
"""
Data augmentation script for CropXR project.
Applies various augmentation techniques to increase dataset diversity.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import random
import os
from pathlib import Path
from tqdm import tqdm
import argparse


class ImageAugmenter:
    """Class for image augmentation operations."""
    
    def __init__(self, rotation_range=15, brightness_range=(0.8, 1.2), 
                 contrast_range=(0.8, 1.2), zoom_range=(0.9, 1.1)):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.zoom_range = zoom_range
    
    def rotate(self, image, angle=None):
        """Rotate image by random angle within range."""
        if angle is None:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, M, (w, h))
        else:
            return image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    
    def adjust_brightness(self, image, factor=None):
        """Adjust image brightness."""
        if factor is None:
            factor = random.uniform(*self.brightness_range)
        
        if isinstance(image, np.ndarray):
            return np.clip(image * factor, 0, 255).astype(np.uint8)
        else:
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
    
    def adjust_contrast(self, image, factor=None):
        """Adjust image contrast."""
        if factor is None:
            factor = random.uniform(*self.contrast_range)
        
        if isinstance(image, np.ndarray):
            # Convert to float for computation
            img_float = image.astype(np.float32)
            mean = np.mean(img_float)
            return np.clip(mean + factor * (img_float - mean), 0, 255).astype(np.uint8)
        else:
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
    
    def horizontal_flip(self, image):
        """Flip image horizontally."""
        if isinstance(image, np.ndarray):
            return cv2.flip(image, 1)
        else:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    def add_noise(self, image, noise_factor=0.1):
        """Add random noise to image."""
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        noise = np.random.normal(0, noise_factor * 255, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def blur(self, image, kernel_size=None):
        """Apply blur to image."""
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7])
        
        if isinstance(image, np.ndarray):
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            return image.filter(ImageFilter.GaussianBlur(radius=kernel_size//2))
    
    def augment_image(self, image, num_augmentations=1):
        """Apply random augmentations to image."""
        augmentations = [
            self.rotate,
            self.adjust_brightness,
            self.adjust_contrast,
            self.horizontal_flip,
            lambda img: self.add_noise(img, 0.05),
            lambda img: self.blur(img, 3)
        ]
        
        augmented_images = []
        for _ in range(num_augmentations):
            aug_img = image.copy()
            # Apply 1-3 random augmentations
            selected_augs = random.sample(augmentations, random.randint(1, 3))
            for aug_func in selected_augs:
                try:
                    aug_img = aug_func(aug_img)
                except Exception as e:
                    print(f"Augmentation failed: {e}")
                    continue
            augmented_images.append(aug_img)
        
        return augmented_images


def augment_dataset(input_dir, output_dir, augmentations_per_image=3):
    """
    Augment all images in input directory and save to output directory.
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save augmented images
        augmentations_per_image: Number of augmented versions per original image
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    augmenter = ImageAugmenter()
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to augment")
    
    for img_path in tqdm(image_files, desc="Augmenting images"):
        try:
            # Load original image
            image = Image.open(img_path).convert('RGB')
            
            # Save original image to output directory
            orig_name = img_path.stem
            orig_ext = img_path.suffix
            image.save(output_path / f"{orig_name}_orig{orig_ext}")
            
            # Generate augmented versions
            augmented = augmenter.augment_image(image, augmentations_per_image)
            
            for i, aug_img in enumerate(augmented):
                aug_name = f"{orig_name}_aug_{i+1}{orig_ext}"
                if isinstance(aug_img, np.ndarray):
                    aug_img = Image.fromarray(aug_img)
                aug_img.save(output_path / aug_name)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Augmentation complete! Augmented images saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment image dataset")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing original images")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save augmented images")
    parser.add_argument("--augmentations", type=int, default=3,
                       help="Number of augmented versions per image")
    
    args = parser.parse_args()
    
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    augment_dataset(args.input_dir, args.output_dir, args.augmentations)


if __name__ == "__main__":
    main()