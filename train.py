"""
Training function for implemented models. Currently the following models have
been implemented:
    - FCN 'fcn'
    - PSPNet 'pspnet'
    - MobileNetV2 'mobilenetv2'
Convenience training functions for each model have been defined at the bottom
of the file.
To add new models, complete the following:
    1. Add the model's default configuration settings to MODELS_DEFAULT_CONFIG 
        in models_config.py
    2. Add code for creating a new model in the train function
To enable running from the command line, uncomment the line # train(**vars(args))
at the bottom of the file and remove all other calls to train()
"""

import argparse
import os

from keras import callbacks, optimizers, losses, metrics
from keras import backend as K
import keras

import preprocess
import one_hot_labels
from models_config import MODELS_DEFAULT_CONFIG
import nets.fcn.fcn as fcn
import nets.pspnet.pspnet as pspnet
import nets.mobilenetv2.mobilenetv2 as mobilenetv2

models_default_params = MODELS_DEFAULT_CONFIG # Retrieve configuration from models_config.py
checkpoint_path = 'artifacts/checkpoint_weights.{epoch:02d}-{val_loss:.2f}.h5'
channels = 3 # Number of channels in input images
nb_labels = one_hot_labels.NB_LABELS # Number of terrain types

K.set_image_data_format("channels_last")

def train(model_name, batch_size, steps_per_epoch, epochs, validation_steps, 
          model_file=None, save_path=None):
    """Train
    Train an implemented model.
    
    # Arguments
        model_name: String. The name of an implemented model.
        batch_size: Integer. Number of patches in each training and validation batch.
        steps_per_epoch: Integer. Number of training batches in each epoch.
        epochs: Integer. Number of epochs.
        validation_steps: Integer. Number of batches for validation after 
            each epoch.
        model_file: String. Model file as h5 format. If not specified, a new 
            model will be created.
        save_path: String. Path to save trained model file. If not specified, 
            will save in default path. 
    """
    
    print("- Loading configuration...")
    if model_name in models_default_params:
        default_params = models_default_params[model_name]
    else:
        print("Error: the model '{}' has not been implemented".format(model_name))
        return
    custom_objects = default_params['custom_objects']
    patch_size = default_params['patch_size']
    if save_path is None:
        save_path = default_params['default_path']
    if os.path.isfile(save_path):
        print("Warning: {} is an existing file and will be overwritten.".format(save_path))
    print("- Configuration loaded.")
        
    print("- Loading datasets...")
    train_gen = preprocess.load_dataset(batch_size, x_directory = "datasets/Potsdam/Training/RGB/",
                                        y_directory = "datasets/Potsdam/Training/Labels/",
                                        patch_size = patch_size)
    val_gen = preprocess.load_dataset(batch_size, x_directory = "datasets/Potsdam/Validation/RGB/",
                                      y_directory = "datasets/Potsdam/Validation/Labels/",
                                      patch_size = patch_size)
    print("- Data loaded.")
    
    print("- Initialising model...")
    if(model_file is not None): # Further train existing model
        model = keras.models.load_model(model_file, custom_objects=custom_objects)
    else: # Create new model
        if model_name == 'fcn':
            model = fcn.make_fcn_resnet((patch_size, patch_size, channels), nb_labels, 
                                        use_pretraining=False, freeze_base=False)
        elif model_name == 'pspnet':
            model = pspnet.build_pspnet(nb_classes=nb_labels, resnet_layers=50,
                                        input_shape=patch_size)
        elif model_name == 'mobilenetv2':
            model = mobilenetv2.MobileNetv2((patch_size, patch_size, channels), nb_labels) 

        model.compile(
            optimizer = optimizers.Adam(lr = 0.00001),
            loss = losses.categorical_crossentropy,
            metrics = [metrics.categorical_accuracy])          
    model.summary()      
    print("- Model initialised.")
    
    tensorboard = callbacks.TensorBoard(log_dir='./logs')
    csv_logger = callbacks.CSVLogger('logs/training.csv')
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           save_best_only=True)
    
    print("- Starting training.")
    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        # callbacks=[checkpoint, csv_logger]
    )
    print("- Training complete.")
    
    model.save(save_path)
    print("- Model saved to {}".format(save_path))
  
    
if __name__ == "__main__":
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='mobilenetv2',
        help='Model to train',
        choices=['fcn',
                 'pspnet',
                 'mobilenetv2']
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default=None,
        help='Model file as h5 format. If not specified, new model will be created',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Path to save trained model file. If not specified, will save in default path',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Number of patches in each training and validation batch'
    )
    parser.add_argument(
        '--steps_per_epoch',
        type=int,
        default=30,
        help='Number of training batches in each epoch'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs'
    )
    parser.add_argument(
        '--validation_steps',
        type=int,
        default=5,
        help='Number of batches for validation after each epoch'
    )
    args, _ = parser.parse_known_args()
    # Uncomment the following line to enable running from command line:
    # train(**vars(args))
    
    # Convenience functions for training. Edit as desired
    def train_fcn():
        train(model_name='fcn', 
              batch_size=8, 
              steps_per_epoch=30, 
              epochs=5,
              validation_steps=5, 
              model_file=models_default_params['fcn']['default_path'],
              save_path=models_default_params['fcn']['default_path'])
    def train_pspnet():
        train(model_name='pspnet', 
              batch_size=2, 
              steps_per_epoch=120, 
              epochs=5,
              validation_steps=20, 
              model_file=models_default_params['pspnet']['default_path'],
              save_path=models_default_params['pspnet']['default_path'])
    def train_mobilenetv2():
        train(model_name='mobilenetv2', 
              batch_size=8, 
              steps_per_epoch=30, 
              epochs=5,
              validation_steps=5, 
              model_file=models_default_params['mobilenetv2']['default_path'],
              save_path=models_default_params['mobilenetv2']['default_path'])
    # Finally, train the model:
    # Remove this if enabling command line
    train_mobilenetv2()