# -*- coding: utf-8 -*-
"""
ResNet50 based Fully Convolutional Network (FCN).

Adapted from Azavea: raster-vision 
    https://www.azavea.com/blog/2017/05/30/deep-learning-on-aerial-imagery

# Reference
- [Fully Convolutional Networks for Semantic Segmentation]
"""

from keras.models import Model 
from keras.layers import (
        Input, Activation, Reshape, Conv2D, Lambda, Add)
import tensorflow as tf

from .resnet50 import ResNet50


FCN_RESNET = 'fcn_resnet'


def make_fcn_resnet(input_shape, nb_labels, use_pretraining, freeze_base):
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)
    weights = 'imagenet' if use_pretraining else None
    
    # A ResNet model with weights from training on ImageNet.
    model = ResNet50(
            include_top=False, weights=weights, input_tensor=input_tensor)
    
    if freeze_base:
        for layer in model.layers:
            layer.trainable = False
    
    # Begin decoder module
    # Get final 32x32, 16x16, and 8x8 layers in the original
    # ResNet by the layer names.
    x32 = model.get_layer('act3d').output
    x16 = model.get_layer('act4f').output
    x8 = model.get_layer('act5c').output
    
    # Compress each skip connection so it has nb_labels channels.
    c32 = Conv2D(nb_labels, (1, 1), name='conv_labels_32')(x32)
    c16 = Conv2D(nb_labels, (1, 1), name='conv_labels_16')(x16)
    c8 = Conv2D(nb_labels, (1, 1), name='conv_labels_8')(x8)
    
    # Resize each compressed skip connection using bilinear interpolation.
    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])
    
    r32 = Lambda(resize_bilinear, name='resize_labels_32')(c32)
    r16 = Lambda(resize_bilinear, name='resize_labels_16')(c16)
    r8 = Lambda(resize_bilinear, name='resize_labels_8')(c8)
    
    # Merge the three layers together using summation.
    m = Add(name='merge_labels')([r32, r16, r8])
    
    # Add softmax layer to get probabilities as output.
    x = Reshape((nb_rows * nb_cols, nb_labels))(m)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)
    
    model = Model(inputs=input_tensor, outputs=x)
    
    return model
