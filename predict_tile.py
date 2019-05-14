# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:10:39 2018

@author: User

For predicting images larger than the input patch size of the model. Contains
code for breaking the large image tile into patches, predicting those patches
as batches, and aggregating the predictions into an output tile.
"""

import os
import time
import itertools
import argparse

import keras

import preprocess
from ProbabilityTile import ProbabilityTile
from models_config import MODELS_DEFAULT_CONFIG

models_default_params = MODELS_DEFAULT_CONFIG # Fetch defaults from models_config.py

def predict_tile(model_name, tile_file, save_name, model_file=None, batch_size=8):
    """Predict Tile
    Predict and save a terrain classification label for an input image tile file.
    
    # Arguments
        model: String. The name of an implemented model.
        tile_file: String. The filename of the input image tile file.
        save_name: String. Filename to save predicted image.
        model_file: String. Filename of Model file as h5 format. If not 
            specified, uses default filename.
        batch_size: Integer. Number of patches in each batch.
    """
    print("- Loading configuration...")
    if model_name in models_default_params:
        default_params = models_default_params[model_name]
    else:
        print("model_name not specified. Loading generic configuration.")
        default_params = models_default_params['generic']
    custom_objects = default_params['custom_objects']
    stride = default_params['stride']
    patch_size = default_params['patch_size']
    if model_file is None:
        model_file = default_params['default_path']
    if stride > patch_size:
        print("Warning: stride of {} is greater than patch size of {}."
              .format(stride, patch_size))
    print("- Configuration loaded.")
    
    print("- Loading tile...")
    if not os.path.isfile(tile_file):
        print("Error: {} is not a valid file.".format(tile_file))
        return
    patch_gen = preprocess.load_patch_batch(tile_file, batch_size, 
                                            patch_size = patch_size, stride = stride)
    steps = preprocess.calc_nb_batches(tile_file, batch_size, 
                                       patch_size = patch_size, stride = stride)
    height, length = preprocess.get_image_size(tile_file)
    print("- Tile loaded.")
    
    print("- Initialising model...")
    model = keras.models.load_model(model_file, custom_objects = custom_objects)
    model.summary()
    print("- Model initialised.")
    
    print("- Predicting tile...")
    start = time.time()
    pred_b_coordinates = []
    pred_b_probs = []
    for i in range(steps):
        print("Batch {}/{}".format(i+1, steps))
        b_start = time.time()
        coordinates, patches = next(patch_gen)
        pred_b_coordinates.append(coordinates)
        pred_b_probs.append(model.predict_on_batch(patches))
        b_duration = time.time() - b_start
        print("Batch predicted in {:.3f}s".format(b_duration))
    duration = time.time() - start
    print("- Prediction complete in {:.3f}s.".format(duration))
    
    print("- Averaging predictions...")
    start = time.time()
    tile = ProbabilityTile(height, length)
    predicted_coordinates = list(itertools.chain.from_iterable(pred_b_coordinates))
    predicted_probs = list(itertools.chain.from_iterable(pred_b_probs))
    for i in range(len(predicted_probs)):
        tile.merge_prob_patch(predicted_coordinates[i][0], predicted_coordinates[i][1],
                              predicted_probs[i])
    tile_label = tile.get_tile_label()
    duration = time.time() - start
    print("- Averaging complete in {:.3f}s".format(duration))
    
    print("- Saving predicted tile...")
    preprocess.save_image(tile_label, save_name)
    print("- Saved.")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='mobilenetv2',
        help='Model to use',
        choices=['fcn',
                 'pspnet',
                 'mobilenetv2']
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default=None,
        help='Model file as h5 format. If not specified, uses default file for given model name',
        choices=['artifacts/model_FCN.h5',
                 'artifacts/model_PSPNet.h5',
                 'artifacts/model_MobileNetv2.h5']
    )
    parser.add_argument(
        '--tile_file',
        type=str,
        default='datasets/sample_minitile_rgb.tif',
        #default='datasets/Potsdam/Validation/RGB/top_potsdam_3_12_RGB.tif',
        help='Tile file as tif format'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Number of patches in each batch',
    )
    parser.add_argument(
        '--save_name',
        type=str,
        default='tilepred.png',
        help='Filename to save predicted image',
    )
    args, _ = parser.parse_known_args()
    predict_tile(**vars(args))