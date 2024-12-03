"""
Parses the 'dataset-processed' directory and generates enough augmented images
of the scoliosis and normal classes to reach a specified quantity.

RUN THIS BEFORE TRAINING ANY MODELS
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
from PIL import Image
from scipy.ndimage import rotate
from tqdm import tqdm

def count_files(dir):
    return len([f for f in Path(dir).rglob('*') if f.is_file()])

def apply_rotation(img_array, max_rotation=10):
    angle = random.uniform(-max_rotation, max_rotation)
    return rotate(img_array, angle, reshape=False, mode='constant', cval=0)

def apply_shift(img_array, max_shift=20):
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    shifted = np.zeros_like(img_array)

    if shift_x >= 0:
        shifted[shift_x:, :] = img_array[:-shift_x, :] if shift_x > 0 else img_array
    else:
        shifted[:shift_x, :] = img_array[-shift_x:, :]
    
    if shift_y >=0:
        shifted[:, shift_y:] = img_array[:, :-shift_y] if shift_y > 0 else img_array
    else:
        shifted[:, :shift_y] = img_array[:, -shift_y:]

    return shifted

def apply_noise(img_array, noise_factor=0.05):
    noise = np.random.normal(0, noise_factor * np.mean(img_array), img_array.shape)
    return np.clip(img_array + noise, 0, 255).astype(np.uint8)

def apply_flip(img_array):
    return np.fliplr(img_array)

def apply_augmentation(image, max_rotation=10, max_shift=20, noise_factor=0.05):
    img_array = np.array(image)

    movement_augmentations = {
        'rotation': lambda: apply_rotation(img_array, max_rotation),
        'shift': lambda: apply_shift(img_array, max_shift),
    }
    other_augmentations = {
        'noise': lambda: apply_noise(img_array, noise_factor),
        'flip': lambda: apply_flip(img_array)
    }

    movement_aug = []
    if random.random() < 0.7:
        movement_aug = [random.choice(list(movement_augmentations.keys()))]
    num_other_augmentations = random.randint(0,len(other_augmentations))
    other_aug = random.sample(list(other_augmentations.keys()), num_other_augmentations)

    selected_augmentations = movement_aug + other_aug

    if not selected_augmentations:
        all_augmentations = {**movement_augmentations, **other_augmentations}
        selected_augmentations = [random.choice(list(all_augmentations.keys()))]

    current_img = img_array

    all_augmentations = {**movement_augmentations, **other_augmentations}
    for aug_name in selected_augmentations:
        current_img = all_augmentations[aug_name]()
    
    return current_img

def augment_images(input_dir, output_dir, target_count, max_rotation=10, max_shift=20, noise_factor=0.05):
    os.makedirs(output_dir, exist_ok=True)

    image_files = list(Path(input_dir).rglob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg','.jpeg', '.png']]

    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        output_path = os.path.join(output_dir, f'original_{i}{img_path.suffix}')
        cv2.imwrite(output_path, img)

    current_count = len(image_files)
    augs_needed = max(0, target_count - current_count)

    with tqdm(total=augs_needed) as pbar:
        aug_i = 0
        while aug_i < augs_needed:
            img_path = np.random.choice(image_files)
            img = cv2.imread(str(img_path))

            if img is None:
                continue

            augmented = apply_augmentation(
                img,
                max_rotation=max_rotation,
                max_shift=max_shift,
                noise_factor=noise_factor
            )

            output_path = os.path.join(output_dir, f'aug_{aug_i}{img_path.suffix}')
            cv2.imwrite(output_path, augmented)

            aug_i += 1
            pbar.update(1)

def main():
    base_dir = 'dataset-processed'
    output_base_dir = 'og-augmented'

    os.makedirs(output_base_dir, exist_ok=True)

    categories = ['normal', 'scoliosis']
    for c in categories:
        input_dir = os.path.join(base_dir, c)
        file_count = count_files(input_dir)
        print(f"Number of files in {c}: {file_count}")

        output_dir = os.path.join(output_base_dir, c)

        target_count = 1200
        print(f"\nAugmenting {c} images to reach {target_count} images...")
        augment_images(
            input_dir,
            output_dir,
            target_count,
            max_rotation=10,
            max_shift=20,
            noise_factor=0.05
        )

        final_count = count_files(output_dir)
        print(f"final number of images in {c}: {final_count}\n")

if __name__ == "__main__":
    main()