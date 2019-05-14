"""
Image preprocessing utilities for the Potsdam dataset. Images are read using
OpenCV and handled as NumPy arrays with channels last.
"""

import math
import random
import os
import time

import numpy as np
import cv2 # Using cv2 for manipulating large images
from keras.preprocessing.image import ImageDataGenerator

import one_hot_labels

TRAIN_X_PATH = "datasets/Potsdam/Training/RGB/"
TRAIN_Y_PATH = "datasets/Potsdam/Training/Labels/"
VAL_X_PATH = "datasets/Potsdam/Validation/RGB/"
VAL_Y_PATH = "datasets/Potsdam/Validation/Labels/"
TEST_X_PATH = "datasets/Potsdam/Testing/RGB/"
TEST_Y_PATH = "datasets/Potsdam/Testing/Labels/"
IMAGE_TYPE = "tif"
PATCH_SIZE = 224 # Default patch size
SEED_RANGE = 2**32-1 # Maximum value for ImageDataGenerator seed

def get_image_size(im_path):
    """Get the height and length of the image."""
    if not os.path.isfile(im_path):
        print("Error: {} is not a valid file path.".format(im_path))
        return None
    height, length, _ = cv2.imread(im_path).shape
    return height, length


def read_image(im_path):
    """Read the image and return as a NumPy array."""
    if not os.path.isfile(im_path):
        print("Error: {} is not a valid file path.".format(im_path))
        return None
    bgr_im = cv2.imread(im_path) # Numpy array is (Height x Width)
    # Convert from default CV2 BGR to RGB
    rgb_im = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB) 
    return rgb_im


def save_image(image, im_name):
    """Save RGB format image."""
    bgr_im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(im_name, bgr_im)


def _get_pathset(x_path=TRAIN_X_PATH, y_path=TRAIN_Y_PATH):
    """Get paths of all label and RGB images in the given directories."""
    y_filenames = [y for y in os.listdir(y_path) if y.endswith("." + IMAGE_TYPE)]
    x_filenames = [x.replace('label', 'RGB') for x in y_filenames]
    # Assumes that each labelled image has an RGB equivalent
    x_filepaths = [x_path + x for x in x_filenames]
    y_filepaths = [y_path + y for y in y_filenames]
    return x_filepaths, y_filepaths


def check_corruption(image):
    """Check label image for corruption (values not 0 or 255)."""
    for row in image:
        for pixel in row:
            for channel in pixel:
                if(not (int(channel) == 0 or int(channel) == 255)):
                    print(pixel)
                    
    
def crop_image(image, x_start, x_end, y_start, y_end): 
    """Crop the image to the given coordinates."""
    # Does not include the end pixels!
    if x_end > image.shape[1] or y_end > image.shape[0]:
        print("Error: ", x_start, x_end, y_start, y_end)
    return image[y_start:y_end, x_start:x_end] # Height x Width


def get_random_patchpair(image1, image2, patch_size):
    """Get a corresponding pair of randomly cropped patches from the given image pair."""   
    im_length = image1.shape[1]
    im_height = image1.shape[0]
    if(patch_size > im_height or patch_size > im_length):
        print("error: patch_size is larger than image size")
        return
    elif(image1.shape != image2.shape):
        print("error: image1 and image2 are of different shape")
        return
    
    x_patch = random.randint(0, im_length - patch_size)
    y_patch = random.randint(0, im_height - patch_size)
    
    patch1 = crop_image(image1, x_patch, x_patch+patch_size, y_patch, y_patch+patch_size)
    patch2 = crop_image(image2, x_patch, x_patch+patch_size, y_patch, y_patch+patch_size)
    return patch1, patch2


def augment_brightness(image, brightness_range=0.5):
    """Augment Brightness
    Randomly upscale or downscale the brightness of the image.
    
    # Arguments
        image: 3D NumPy array in RGB.
        brightness_range: Float. Range for random brightness adjustment.
    
    # Returns
        3D NumPy array in RGB.
    """
    if not 0.0 < brightness_range < 1.0:
        print("Invalid brightness range. Defaulting to 0.5")
        brightness_range = 0.5
        
    # Convert to HSV colour space
    im = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    im = np.array(im, dtype=np.float64)
    
    # Get random brightness scale factor
    random_bright = random.uniform(1-brightness_range, 1+brightness_range)
    # Scale the V channel to change brightness
    im[:, :, 2] = im[:, :, 2] * random_bright
    # Clip excessive values to 255
    im[:, :, 2][im[:, :, 2] > 255] = 255
    
    # Convert back to RGB
    im = np.array(im, dtype=np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
    return im


def augment_batches(rgb_batch, label_batch):
    """Augment Batches
    Randomly augment the batch pair with matching transformations:
        - Brightness scaling
        - 90 degree rotations
        - Horizontal flips
        - Vertical flips
    """
    if len(rgb_batch) != len(label_batch):
        print("Error: cannot augment; batch sizes different.")
        return rgb_batch, label_batch
      
    for i in range(len(rgb_batch)):   
        # Randomly scale brightness of RGB image
        if random.random() < 0.8: # 80% chance of brightness scaling
            rgb_batch[i] = augment_brightness(rgb_batch[i])
        # Randomly rotate counterclockwise by 90 degrees
        if random.random() < 0.5: # 50% chance of rotation
            rgb_batch[i] = np.rot90(rgb_batch[i])
            label_batch[i] = np.rot90(label_batch[i])
            
    # Next, randomly flip horizontally and/or vertically
    # Must ensure both batches receive same transformations
    # Create two instances with the same arguments
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,)
    datagen1 = ImageDataGenerator(**data_gen_args)
    datagen2 = ImageDataGenerator(**data_gen_args)   
    # Provide the same seed and keyword arguments.
    seed = random.randint(0, SEED_RANGE)
    gen1 = datagen1.flow(rgb_batch, seed=seed)
    gen2 = datagen2.flow(label_batch, seed=seed)
   
    return gen1.next(), gen2.next()

    
def load_dataset(batch_size, x_directory=TRAIN_X_PATH, 
                 y_directory=TRAIN_Y_PATH, patch_size=PATCH_SIZE, save=False):
    """Load dataset
    Load and augment a batch of RGB and label image patch pairs randomly 
    selected from the dataset. The patches are randomly cropped from the images
    selected.
    
    Each RGB image file name must end with the suffix 'RGB' before the file 
    type and have a corresponding label image ending with 'label'. The file 
    name prefixes must be identical.
    
    Example: 'top_potsdam_2_10_RGB.tif' and 'top_potsdam_2_10_label.tif' is a
    valid pair.
    
    # Arguments
        batch_size: Integer, number of image pairs to include in the batch.
        x_directory: String, the path of the directory containing RGB images.
        y_directory: String, the path of the directory containing label images.
        patch_size: Integer, the height and width of each cropped square patch. 
        save: Boolean. If true, patches will be saved as png images in the
            parent folder.
    
    # Returns
        Tuple of two 4D NumPy arrays with dimensions (batch_size, patch_size,
        patch_size, 3).
    """
    x_filepaths, y_filepaths = _get_pathset(x_directory, y_directory)
    while True:
        print("Preprocessing batch...")
        start = time.time()
        rgbs = []
        labels = []
        remainingIm = batch_size
        i = 0
        while remainingIm > 0:
            image_index = random.randint(0, len(x_filepaths)-1) # Choose a tile
            rgb_im = read_image(x_filepaths[image_index])
            label_im = read_image(y_filepaths[image_index])
            for j in range(min(10, remainingIm)): # Get max 10 patches from tile
                rgb_patch, label_patch = get_random_patchpair(rgb_im, label_im, patch_size)
                if save:
                    save_image(rgb_patch, "{}{}_rgb.png".format(i, j))
                    save_image(label_patch, "{}{}_label.png".format(i, j))
                rgbs.append(rgb_patch)
                labels.append(label_patch)
                remainingIm -= 1
            i += 1
        # convert list of 3D arrays into 4D array and augment
        aug_rgbs, aug_labels = augment_batches(np.array(rgbs, dtype=np.uint8), 
                                               np.array(labels, dtype=np.uint8))
        
        aug_masks = np.empty((batch_size, patch_size, patch_size, 
                              one_hot_labels.NB_LABELS))
        for i, image in enumerate(aug_labels):    
            aug_masks[i] = one_hot_labels.one_hot_encode(image)
            
        duration = time.time() - start
        print("Preprocessed in {:.3f}s. ({:.3f}s/image)".format(duration, duration/batch_size))
        yield aug_rgbs, aug_masks
        
        
def load_tile_patches(tile_path, patch_size=PATCH_SIZE, stride=None):
    """ Load Tile Patches
    Generator function that loads an entire large image tile as patches. Scans 
    left to right columnwise, up to down. Depending on tile and patch 
    dimensions, there may be overlap between patches of the bottom and right 
    sides of the tile.
    
    # Arguments
        tile_path: String, the path of the image tile.
        patch_size: Integer, the height and width of each cropped square patch.
        stride: Integer, the stride between each patch. If not specified, is
            set as half the patch size.
    
    # Yields
        X coordinate, Y coordinate, and patch image as NumPy array.
    """
    if stride == None:
        stride = patch_size//2 # Assume this default value
    tile_im = read_image(tile_path)
    height = tile_im.shape[0]
    length = tile_im.shape[1]
    x = 0
    y = 0
    while x+patch_size <= length: # Generate left majority of tile
        y = 0
        while y+patch_size <= height:
            x_co, y_co = x, y
            yield x_co, y_co, crop_image(tile_im, x_co, x+patch_size, y_co, y+patch_size)
            y += stride
        if(y < height): # Generate remaining bottom patch of column
            x_co, y_co = x, height-patch_size
            yield x_co, y_co, crop_image(tile_im, x_co, x+patch_size, y_co, height)
        x += stride
    if(x < length): # Generate remaining right edge
        y = 0
        while y+patch_size <= height:
            x_co, y_co = length-patch_size, y
            yield x_co, y_co, crop_image(tile_im, x_co, length, y_co, y+patch_size)
            y += stride
        if(y < height): # Generate remaining bottom patch of column
            x_co, y_co = length-patch_size, height-patch_size
            yield x_co, y_co, crop_image(tile_im, x_co, length, y_co, height)
            
            
def load_patch_batch(tile_path, batch_size, patch_size=PATCH_SIZE, stride=None):
    """ Load Patch Batch
    Generator function that loads an entire large image tile as batches of 
    patches. 
    
    # Arguments
        tile_path: String, the path of the image tile.
        batch_size: Integer, the number of patches to include in each batch.
        patch_size: Integer, the height and width of each cropped square patch.
        stride: Integer, the stride between each patch. If not specified, is
            set as half the patch size.
    
    # Yields
        A 2D Numpy array of x, y coordinates of shape [batch_size, 2],
        and 4D Numpy array of corresponding patch images of shape [batch_size,
        patch_size, patch_size, 3]
    """
    generator = load_tile_patches(tile_path, patch_size, stride)
    nb_patches_remaining = calc_nb_tile_patches(tile_path, patch_size, stride)
    while nb_patches_remaining > 0:
        nb_this_batch = min(nb_patches_remaining, batch_size)
        x, y, patch = next(generator)
        coordinates = np.empty((nb_this_batch, 2), dtype = np.uint32)
        patches = np.empty((nb_this_batch, patch.shape[0], patch.shape[1], patch.shape[2]))
        coordinates.itemset(0, 0, x)
        coordinates.itemset(0, 1, y)
        patches[0] = patch
        for i in range(1, nb_this_batch):
            x, y, patch = next(generator)
            coordinates.itemset(i, 0, x)
            coordinates.itemset(i, 1, y)
            patches[i] = patch
        yield coordinates, patches
        nb_patches_remaining -= nb_this_batch
        
            
def calc_nb_tile_patches(tile_path, patch_size=PATCH_SIZE, stride=None):
    """Calculate number of patches from the tile with given patch size and stride."""
    if stride == None:
        stride = patch_size//2 # Assume this default value
    height, length = get_image_size(tile_path)
    if patch_size > height or patch_size > length or stride > patch_size:
        return 0
    ny = 1 + math.ceil((float(height)-patch_size)/stride)
    nx = 1 + math.ceil((float(length)-patch_size)/stride)
    return ny * nx


def calc_nb_batches(tile_path, batch_size, patch_size=PATCH_SIZE, stride=None):
    """Calculate number of patch batches from tile with given patch size and stride."""
    if stride == None:
        stride= patch_size//2 # Assume this default value
    return math.ceil(float(calc_nb_tile_patches(
            tile_path, patch_size=patch_size, stride=stride))/batch_size)
    

# Debug
if __name__ == "__main__":
    tile_path = "datasets/sample_minitile_rgb.tif"
    gen = load_dataset(16, save=True)
    b1, b2 = next(gen)
        