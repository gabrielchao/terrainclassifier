# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:32:25 2018

@author: User

Utilities for encoding and decoding between RGB and one-hot images.

IMPORTANT: NumPy uses height x width convention!
"""

import time
import numpy as np
import PIL

import preprocess

CLASSES = [
        (255, 255, 255), # Impervious surfaces
        (255, 0, 0), # Clutter/background
        (255, 255, 0), # Car 
        (0, 255, 0), # Tree
        (0, 255, 255), # Low vegetation
        (0, 0, 255)]  # Building
NB_LABELS = len(CLASSES)

# This is an example of a naive way of using NumPy
def old_one_hot_encode(label_image): 
    """(Deprecated) One-hot encode an RGB label image."""
    mask = np.zeros((label_image.shape[0], label_image.shape[1], NB_LABELS))
    for y in range(label_image.shape[0]):
        for x in range(label_image.shape[1]):
            pixel = tuple(label_image[y][x])
            try:
                mask.itemset(y, x, CLASSES.index(pixel), 1)
            except:
                print("Error: undefined label image colour at " + str(x) + ", " + str(y))
                print(label_image[y][x])
                print(pixel)
                input("continue...")
    return mask


# This function is more optimised to take advantage of NumPy
def one_hot_encode(label_image):
    """One-hot encode an RGB label image."""
    mask = np.zeros((label_image.shape[0], label_image.shape[1], NB_LABELS))
    for index, tclass in enumerate(CLASSES):
        # Generate boolean mask for this class
        tmask = np.equal(label_image, np.array(tclass)).all(axis=2) 
        mask[tmask, index] = 1 # Set one-hot using class mask
    return mask


# TODO: optimise this function
def one_hot_decode(mask_image):
    """Decode a one-hot mask into an RGB label image."""
    image = np.empty((mask_image.shape[0], mask_image.shape[1], 3), dtype = np.uint8)
    for y in range(mask_image.shape[0]):
        for x in range(mask_image.shape[1]):
            index = -1
            for i in range(NB_LABELS):
                if(mask_image.item(y, x, i) == 1):
                    index = i
                    break;
            if index != -1:
                image[y][x] = np.array(CLASSES[index])
            else:
                print("Error: undefined one-hot encoding.")
    return image


# TODO: optimise this function
def probability_decode(prob_image):
    """Convert a softmax probability mask to an RGB label image."""
    image = np.zeros((prob_image.shape[0], prob_image.shape[1], 3), dtype = np.uint8)
    for y in range(prob_image.shape[0]):
        for x in range(prob_image.shape[1]):
            index = np.argmax(prob_image[y][x]) # Label with highest probability
            if(0 <= index < NB_LABELS):
                image[y][x] = np.array(CLASSES[index])
            else:
                print("Error: undefined one-hot encoding.")
    return image


# Debug
if __name__ == "__main__":
    im_path = "C:/Users/User/Documents/terrainclassifier/datasets/sample_patch_label.png"
    label = preprocess.read_image(im_path)
    start = time.time()
    recoded_label = one_hot_decode(one_hot_encode(label))
    PIL.Image.fromarray(recoded_label.astype(np.uint8)).save("recoded_label.png")
    duration = time.time() - start
    print(duration)