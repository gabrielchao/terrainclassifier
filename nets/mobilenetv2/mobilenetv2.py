# -*- coding: utf-8 -*-
"""
MobileNetV2

Adapted from 
    https://github.com/xiaochus/MobileNetV2
See accompanying LICENSE file.

# Reference
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks]
"""

from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, Lambda
from keras.applications.mobilenet import relu6, DepthwiseConv2D

from keras import backend as K
import tensorflow as tf


def _conv_block(inputs, filters, kernel, strides, stage=None):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for 
            all spatial dimensions.
        stage: Integer, current stage label. Used for generating layer names.
    
    # Returns
        Output tensor.
    """
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = Conv2D(filters, kernel, padding = 'same', strides = strides)(inputs)
    x = BatchNormalization(axis = channel_axis)(x)
    if stage is not None:
        x = Activation(relu6, name = 'act' + str(stage))(x) # Named for FCN
    else:
        x = Activation(relu6)(x)
    return x


def _bottleneck(inputs, filters, kernel, t, s, r = False, stage=None):
    """Bottleneck
    This function defines a basic bottleneck structure.
    
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the width and height. Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, whether to use the residuals.
        stage: Integer, current stage label. Used for generating layer names.
    
    # Returns
        Output tensor.
    """
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    
    if stage: 
        # Last block before shape reduction
        x = _conv_block(inputs, tchannel, (1, 1), (1, 1), stage)
    else:
        x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
    
    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)
    
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    
    if r:
        x = Add()([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n, stage=None):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the width and height. Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
        stage: Integer, current stage label. Used for generating layer names.
        
    # Returns
        Output tensor.
    """
    
    x = _bottleneck(inputs, filters, kernel, t, strides, stage=stage)
    
    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)
    
    return x


def MobileNetv2(input_shape, k):
    """MobileNetv2
    This function defines a MobileNetv2 architecture for segmentation.
    
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        
    # Returns
        MobileNetv2 model.
    """
    
    nb_rows, nb_cols, _ = input_shape
    
    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))
    
    x = _inverted_residual_block(x, 16, (3,3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3,3), t=6, strides=2, n=2, stage=1)
    x = _inverted_residual_block(x, 32, (3,3), t=6, strides=2, n=3, stage=2)
    x = _inverted_residual_block(x, 64, (3,3), t=6, strides=2, n=4, stage=3)
    x = _inverted_residual_block(x, 96, (3,3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3,3), t=6, strides=2, n=3, stage=4)
    x = _inverted_residual_block(x, 320, (3,3), t=6, strides=1, n=1)
    
    base = Model(inputs, x) # Feature extractor
    
    # Beginning of decoder module
    # Get final 28x28, 14x14, and 7x7 layers in the base
    # model by the layer names.
    x28 = base.get_layer('act3').output
    x14 = base.get_layer('act4').output
    x7 = x
    
    # Compress each skip connection so it has k channels.
    c28 = Conv2D(k, (1, 1), name='conv_labels_28')(x28)
    c14 = Conv2D(k, (1, 1), name='conv_labels_14')(x14)
    c7 = Conv2D(k, (1, 1), name='conv_labels_7')(x7)
    
    # Resize each compressed skip connection using bilinear interpolation.
    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])
    
    r28 = Lambda(resize_bilinear, name='resize_labels_28')(c28)
    r14 = Lambda(resize_bilinear, name='resize_labels_14')(c14)
    r7 = Lambda(resize_bilinear, name='resize_labels_7')(c7)
    
    # Merge the three layers together using summation.
    m = Add(name='merge_labels')([r28, r14, r7])
    
    # Add softmax layer to get probabilities as output.
    x = Reshape((nb_rows * nb_cols, k))(m)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, k))(x)
    
    model = Model(inputs, x)
    
    """
    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1280))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(k, (1, 1), padding='same')(x)
    
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((k,))(x)
    model = Model(inputs, output)
    """
    
    return model


if __name__ == '__main__':
    model = MobileNetv2((224, 224, 3), 6)
    model.summary()
    
