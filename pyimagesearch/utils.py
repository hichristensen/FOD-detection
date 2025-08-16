# import the necessary packages
from pyimagesearch import config
from pathlib import Path
import numpy as np
import cv2
import depthai as dai

def create_pipeline_images():
    print("[INFO] initializing pipeline...")
    # initialize a depthai pipeline
    pipeline = dai.Pipeline()
    # configure inputs for depthai pipeline
    classifierIN = pipeline.createXLinkIn()
    classifierIN.setStreamName("classifier_in")
    # configure vegetable classifier model and set its input
    print("[INFO] initializing classifier network...")
    classifierNN = pipeline.create(dai.node.NeuralNetwork)
    classifierNN.setBlobPath(
        str(Path(config.CLASSIFIER).resolve().absolute())
    )
    classifierIN.out.link(classifierNN.input)
    # configure outputs for depthai pipeline
    classifierNNOut = pipeline.createXLinkOut()
    classifierNNOut.setStreamName("classifier_nn")
    classifierNN.out.link(classifierNNOut.input)
    # return the pipeline
    return pipeline

def create_pipeline_camera():
    print("[INFO] initializing pipeline...")
    # initialize a depthai pipeline
    pipeline = dai.Pipeline()
    # configure vegetable classifier model and set its input
    print("[INFO] initializing classifier network...")
    classifierNN = pipeline.create(dai.node.NeuralNetwork)
    classifierNN.setBlobPath(
        str(Path(config.CLASSIFIER).resolve().absolute())
    )

    # create and configure the color camera properties
    print("[INFO] Creating Color Camera...")
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(config.CAMERA_PREV_DIM)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_1080_P
    )
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # create XLinkOut node for displaying frames
    cam_xout = pipeline.create(dai.node.XLinkOut)
    # set stream name as rgb
    cam_xout.setStreamName("rgb")
    # link the camera preview to XLinkOut node input
    cam_rgb.preview.link(cam_xout.input)

    # resize the camera frames to dimensions expected by neural network
    print("[INFO] Creating ImageManip node...")
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(config.IMG_DIM)
    # link the camera preview to the manipulation node for resize
    cam_rgb.preview.link(manip.inputImage)
    # link the output of resized frame to input of neural network
    manip.out.link(classifierNN.input)

    # configure outputs for depthai pipeline
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    classifierNN.out.link(xout_nn.input)
    # return the pipeline
    return pipeline

def softmax(x):
    # compute softmax values for each set of scores in x.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    # resize the image array and modify the channel dimensions
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)
