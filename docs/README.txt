Terrain Classifier

For training:
If training with the Potsdam dataset, ensure that the dataset has been placed correctly in the datasets/Potsdam folder.
Command line usage with arguments is currently disabled. To enable it, follow the instructions in the comments in train.py.
Instead, train.py can be conveniently used by modifying the code using the convenience training functions train_fcn(),
train_pspnet(), and train_mobilenetv2().

If command line functionality is enabled, run the train.py script with Python in the command line like so:

python train.py

The help documentation is as follows:

usage: train.py [-h] [--model_name {fcn,pspnet,mobilenetv2}]
                [--model_file MODEL_FILE] [--save_path SAVE_PATH]
                [--batch_size BATCH_SIZE] [--steps_per_epoch STEPS_PER_EPOCH]
                [--epochs EPOCHS] [--validation_steps VALIDATION_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {fcn,pspnet,mobilenetv2}
                        Model to train
  --model_file MODEL_FILE
                        Model file as h5 format. If not specified, new model
                        will be created
  --save_path SAVE_PATH
                        Path to save trained model file. If not specified,
                        will save in default path
  --batch_size BATCH_SIZE
                        Number of patches in each training and validation
                        batch
  --steps_per_epoch STEPS_PER_EPOCH
                        Number of training batches in each epoch
  --epochs EPOCHS       Number of epochs
  --validation_steps VALIDATION_STEPS
                        Number of batches for validation after each epoch

For prediction:
Run the predict_tile.py script with Python in the command line like so:

python train.py

The help documentation is as follows:

usage: predict_tile.py [-h] [--model_name {fcn,pspnet,mobilenetv2}]
                       [--model_file {artifacts/model_FCN.h5,artifacts/model_PSPNet.h5,artifacts/model_MobileNetv2.h5}]
                       [--tile_file TILE_FILE] [--batch_size BATCH_SIZE]
                       [--save_name SAVE_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {fcn,pspnet,mobilenetv2}
                        Model to use
  --model_file {artifacts/model_FCN.h5,artifacts/model_PSPNet.h5,artifacts/model_MobileNetv2.h5}
                        Model file as h5 format. If not specified, uses
                        default file for given model name
  --tile_file TILE_FILE
                        Tile file as tif format
  --batch_size BATCH_SIZE
                        Number of patches in each batch
  --save_name SAVE_NAME
                        Filename to save predicted image


For reference, the project directory structure during development is shown below. __pycache__ files
are bytecode generated automatically by Python interpreters.
C:.
¦   models_config.py
¦   one_hot_labels.py
¦   predict_tile.py
¦   preprocess.py
¦   ProbabilityTile.py
¦   README.txt
¦   summary_fcn.html
¦   summary_mobilenetv2.html
¦   summary_pspnet.html
¦   tilepred.png
¦   tileprediction.xcf
¦   train.py
¦
+---artifacts
¦       model_FCN.h5
¦       model_MobileNetv2.h5
¦       model_PSPNet.h5
¦
+---datasets
¦   ¦   sample_minitile_label.tif
¦   ¦   sample_minitile_rgb.tif
¦   ¦   sample_patch_label.png
¦   ¦   sample_patch_rgb.png
¦   ¦
¦   +---DJI
¦   ¦       DJI_0006.JPG
¦   ¦       DJI_0007.JPG
¦   ¦       DJI_0008.JPG
¦   ¦       DJI_0009.JPG
¦   ¦       DJI_0010.JPG
¦   ¦       DJI_0011.JPG
¦   ¦       DJI_0012.JPG
¦   ¦       DJI_0013.JPG
¦   ¦       DJI_0014.JPG
¦   ¦
¦   +---Potsdam
¦       +---Hidden
¦       ¦   +---RGB
¦       ¦           top_potsdam_2_13_RGB.tfw
¦       ¦           top_potsdam_2_13_RGB.tif
¦       ¦           top_potsdam_2_14_RGB.tfw
¦       ¦           top_potsdam_2_14_RGB.tif
¦       ¦           top_potsdam_3_13_RGB.tfw
¦       ¦           top_potsdam_3_13_RGB.tif
¦       ¦           top_potsdam_3_14_RGB.tfw
¦       ¦           top_potsdam_3_14_RGB.tif
¦       ¦           top_potsdam_4_13_RGB.tfw
¦       ¦           top_potsdam_4_13_RGB.tif
¦       ¦           top_potsdam_4_14_RGB.tfw
¦       ¦           top_potsdam_4_14_RGB.tif
¦       ¦           top_potsdam_4_15_RGB.tfw
¦       ¦           top_potsdam_4_15_RGB.tif
¦       ¦           top_potsdam_5_13_RGB.tfw
¦       ¦           top_potsdam_5_13_RGB.tif
¦       ¦           top_potsdam_5_14_RGB.tfw
¦       ¦           top_potsdam_5_14_RGB.tif
¦       ¦           top_potsdam_5_15_RGB.tfw
¦       ¦           top_potsdam_5_15_RGB.tif
¦       ¦           top_potsdam_6_13_RGB.tfw
¦       ¦           top_potsdam_6_13_RGB.tif
¦       ¦           top_potsdam_6_14_RGB.tfw
¦       ¦           top_potsdam_6_14_RGB.tif
¦       ¦           top_potsdam_6_15_RGB.tfw
¦       ¦           top_potsdam_6_15_RGB.tif
¦       ¦           top_potsdam_7_13_RGB.tfw
¦       ¦           top_potsdam_7_13_RGB.tif
¦       ¦
¦       +---Training
¦       ¦   +---Labels
¦       ¦   ¦       top_potsdam_2_10_label.tfw
¦       ¦   ¦       top_potsdam_2_10_label.tif
¦       ¦   ¦       top_potsdam_3_10_label.tfw
¦       ¦   ¦       top_potsdam_3_10_label.tif
¦       ¦   ¦       top_potsdam_3_11_label.tfw
¦       ¦   ¦       top_potsdam_3_11_label.tif
¦       ¦   ¦       top_potsdam_4_10_label.tfw
¦       ¦   ¦       top_potsdam_4_10_label.tif
¦       ¦   ¦       top_potsdam_4_11_label.tfw
¦       ¦   ¦       top_potsdam_4_11_label.tif
¦       ¦   ¦       top_potsdam_5_10_label.tfw
¦       ¦   ¦       top_potsdam_5_10_label.tif
¦       ¦   ¦       top_potsdam_6_10_label.tfw
¦       ¦   ¦       top_potsdam_6_10_label.tif
¦       ¦   ¦       top_potsdam_6_11_label.tfw
¦       ¦   ¦       top_potsdam_6_11_label.tif
¦       ¦   ¦       top_potsdam_6_7_label.tfw
¦       ¦   ¦       top_potsdam_6_7_label.tif
¦       ¦   ¦       top_potsdam_6_8_label.tfw
¦       ¦   ¦       top_potsdam_6_8_label.tif
¦       ¦   ¦       top_potsdam_6_9_label.tfw
¦       ¦   ¦       top_potsdam_6_9_label.tif
¦       ¦   ¦       top_potsdam_7_10_label.tfw
¦       ¦   ¦       top_potsdam_7_10_label.tif
¦       ¦   ¦       top_potsdam_7_11_label.tfw
¦       ¦   ¦       top_potsdam_7_11_label.tif
¦       ¦   ¦       top_potsdam_7_12_label.tfw
¦       ¦   ¦       top_potsdam_7_12_label.tif
¦       ¦   ¦       top_potsdam_7_7_label.tfw
¦       ¦   ¦       top_potsdam_7_7_label.tif
¦       ¦   ¦       top_potsdam_7_8_label.tfw
¦       ¦   ¦       top_potsdam_7_8_label.tif
¦       ¦   ¦       top_potsdam_7_9_label.tfw
¦       ¦   ¦       top_potsdam_7_9_label.tif
¦       ¦   ¦
¦       ¦   +---RGB
¦       ¦           top_potsdam_2_10_RGB.tfw
¦       ¦           top_potsdam_2_10_RGB.tif
¦       ¦           top_potsdam_3_10_RGB.tfw
¦       ¦           top_potsdam_3_10_RGB.tif
¦       ¦           top_potsdam_3_11_RGB.tfw
¦       ¦           top_potsdam_3_11_RGB.tif
¦       ¦           top_potsdam_4_10_RGB.tfw
¦       ¦           top_potsdam_4_10_RGB.tif
¦       ¦           top_potsdam_4_11_RGB.tfw
¦       ¦           top_potsdam_4_11_RGB.tif
¦       ¦           top_potsdam_5_10_RGB.tfw
¦       ¦           top_potsdam_5_10_RGB.tif
¦       ¦           top_potsdam_6_10_RGB.tfw
¦       ¦           top_potsdam_6_10_RGB.tif
¦       ¦           top_potsdam_6_11_RGB.tfw
¦       ¦           top_potsdam_6_11_RGB.tif
¦       ¦           top_potsdam_6_7_RGB.tfw
¦       ¦           top_potsdam_6_7_RGB.tif
¦       ¦           top_potsdam_6_8_RGB.tfw
¦       ¦           top_potsdam_6_8_RGB.tif
¦       ¦           top_potsdam_6_9_RGB.tfw
¦       ¦           top_potsdam_6_9_RGB.tif
¦       ¦           top_potsdam_7_10_RGB.tfw
¦       ¦           top_potsdam_7_10_RGB.tif
¦       ¦           top_potsdam_7_11_RGB.tfw
¦       ¦           top_potsdam_7_11_RGB.tif
¦       ¦           top_potsdam_7_12_RGB.tfw
¦       ¦           top_potsdam_7_12_RGB.tif
¦       ¦           top_potsdam_7_7_RGB.tfw
¦       ¦           top_potsdam_7_7_RGB.tif
¦       ¦           top_potsdam_7_8_RGB.tfw
¦       ¦           top_potsdam_7_8_RGB.tif
¦       ¦           top_potsdam_7_9_RGB.tfw
¦       ¦           top_potsdam_7_9_RGB.tif
¦       ¦
¦       +---Validation
¦           +---Labels
¦           ¦       top_potsdam_2_11_label.tfw
¦           ¦       top_potsdam_2_11_label.tif
¦           ¦       top_potsdam_2_12_label.tfw
¦           ¦       top_potsdam_2_12_label.tif
¦           ¦       top_potsdam_3_12_label.tfw
¦           ¦       top_potsdam_3_12_label.tif
¦           ¦       top_potsdam_4_12_label.tfw
¦           ¦       top_potsdam_4_12_label.tif
¦           ¦       top_potsdam_5_11_label.tfw
¦           ¦       top_potsdam_5_11_label.tif
¦           ¦       top_potsdam_5_12_label.tfw
¦           ¦       top_potsdam_5_12_label.tif
¦           ¦       top_potsdam_6_12_label.tfw
¦           ¦       top_potsdam_6_12_label.tif
¦           ¦
¦           +---RGB
¦                   top_potsdam_2_11_RGB.tfw
¦                   top_potsdam_2_11_RGB.tif
¦                   top_potsdam_2_12_RGB.tfw
¦                   top_potsdam_2_12_RGB.tif
¦                   top_potsdam_3_12_RGB.tfw
¦                   top_potsdam_3_12_RGB.tif
¦                   top_potsdam_4_12_RGB.tfw
¦                   top_potsdam_4_12_RGB.tif
¦                   top_potsdam_5_11_RGB.tfw
¦                   top_potsdam_5_11_RGB.tif
¦                   top_potsdam_5_12_RGB.tfw
¦                   top_potsdam_5_12_RGB.tif
¦                   top_potsdam_6_12_RGB.tfw
¦                   top_potsdam_6_12_RGB.tif
¦
+---examples
¦       fcn 2_12.png
¦       fcn 3_12.png
¦       mobilenetv2 checkpoint 2_12.png
¦       mobilenetv2 checkpoint 3_12.png
¦       mobilenetv2_epoch150 2_12.png
¦       mobilenetv2_epoch1600 2_12.png
¦       mobilenetv2_epoch2350 2_12.png
¦       mobilenetv2_epoch2350 3_12.png
¦       mobilenetv2_epoch350 2_12.png
¦       mobilenetv2_epoch3650 2_12.png
¦       pspnet 2_12.png
¦       pspnet 3_12.png
¦       resized1_prediction.png
¦       resized2_prediction.png
¦
+---logs
¦       training.csv
¦
+---nets
¦   +---fcn
¦   ¦   ¦   fcn.py
¦   ¦   ¦   resnet50.py
¦   ¦   ¦
¦   ¦   +---__pycache__
¦   ¦           fcn.cpython-36.pyc
¦   ¦           resnet50.cpython-36.pyc
¦   ¦
¦   +---mobilenetv2
¦   ¦   ¦   LICENSE
¦   ¦   ¦   mobilenetv2.py
¦   ¦   ¦
¦   ¦   +---__pycache__
¦   ¦           mobilenetv2.cpython-36.pyc
¦   ¦
¦   +---pspnet
¦       ¦   LICENSE
¦       ¦   pspnet.py
¦       ¦
¦       +---__pycache__
¦               pspnet.cpython-36.pyc
¦
+---__pycache__
        fcn.cpython-36.pyc
        loss.cpython-36.pyc
        models_config.cpython-36.pyc
        one_hot_labels.cpython-36.pyc
        preprocess.cpython-36.pyc
        ProbabilityTile.cpython-36.pyc
        Probability_Tile.cpython-36.pyc
        resnet50.cpython-36.pyc