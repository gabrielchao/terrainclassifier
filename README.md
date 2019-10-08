# Terrain Classifier

![Example predictions](https://raw.githubusercontent.com/gabrielchao/terrainclassifier/master/docs/example_all_models.png)

A semantic segmenation classifier for aerial orthophotos.

Given overhead, birds-eye-view images of terrain, terrainclassifier classifies each pixel of the image by type of terrain.
The output is an image of the same resolution as the original, but with each pixel colour-coded to indicate its terrain class.

terrainclassifier is able to identify the following types of terrain:
- Impervious surfaces (i.e. roads)
- Buildings
- Low vegetation
- Trees
- Cars
- Clutter/background (miscellaneous objects)

## What's Inside

This package consists of preprocessing and training utilities bundled with three Keras CNN (convolutional neural network) models.

The architectures implemented in the models are:
- [Fully Convolutional Network (FCN)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
- [Pyramid Scene Parsing Network (PSPNet)](https://arxiv.org/abs/1612.01105)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

These models were implemented with code from open source repositories (see Acknowledgements).

## Built with

- Keras with Tensorflow backend
- OpenCV

## Acknowledgments

terrainclassifier uses Keras models derived from the following sources:
- Azavea - [Deep Learning for Semantic Segmentation of Aerial Imagery](https://www.azavea.com/blog/2017/05/30/deep-learning-on-aerial-imagery/)
- [Vladkryvoruchko/PSPNet-Keras-tensorflow](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow)
- [xiaochus/MobileNetV2](https://github.com/xiaochus/MobileNetV2)
