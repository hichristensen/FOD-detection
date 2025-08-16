# FOD-detection
Using Yolo, OAK-D and Python to train and run a system for FOD detection in a aerospace setting. 

The repository contains both code to train a model in ONNX format and convert it to BLOB format 
for run time use on an OAK-D device. All the training is in yolov8_pascal_voc_training.py

The classify_X files are for real-time detection of objects from raw images or from an attached 
OAK-D camera. 

The openvino_model directory contains a trained Yolov8n model trained using the Omaha FOD dataset
https://github.com/FOD-UNOmaha/FOD-data. The model using Yolov8 with 31 classes using 300x300 
images. 

Enjoy
