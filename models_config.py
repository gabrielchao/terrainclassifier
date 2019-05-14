"""
Default configuration parameters for the implemented models.
TODO: Maybe put in JSON file?
"""

import tensorflow as tf

import nets.pspnet.pspnet as pspnet
import nets.mobilenetv2.mobilenetv2 as mobilenetv2

MODELS_DEFAULT_CONFIG = {
    'fcn': {'default_path' : 'artifacts/model_FCN.h5',
            'custom_objects' : {'tf':tf}, # Required for lambda layer
            'stride' : 112,
            'patch_size' : 224},
    'pspnet': {'default_path' : 'artifacts/model_PSPNet.h5',
            'custom_objects' : {'Interp':pspnet.Interp},
            'stride' : 240,
            'patch_size' : 473},
    'mobilenetv2': {'default_path' : 'artifacts/model_MobileNetv2.h5',
            'custom_objects' : {'tf':tf,
                                'relu6':mobilenetv2.relu6,
                                'DepthwiseConv2D':mobilenetv2.DepthwiseConv2D},
            'stride' : 112,
            'patch_size' : 224},
    'generic': {'custom_objects' : None,
            'stride' : 112,
            'patch_size' : 256}
}