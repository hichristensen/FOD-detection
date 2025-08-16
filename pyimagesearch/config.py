# import the necessary packages
import os
import glob
# define path to the model, test data directory and results
CLASSIFIER = os.path.join(
     "openvino_model","best.blob"
)
TEST_DATA = glob.glob("test_data/*.jpg")
OUTPUT_IMAGES = os.path.join("results", "pred_images")
OUTPUT_VIDEO = os.path.join("results", "pred_camera.mov")
# define image height and width, and camera preview dimensions
IMG_DIM = (320, 320)
CAMERA_PREV_DIM = (480, 480)
# define the class label names list
LABELS = [
    "AdjustableClamp",    "AdjustableWrench",
    "Battery",            "Bolt",
    "BoltNutSet",         "BoltWasher",
    "ClampPart",          "Cutter",
    "FuelCap",            "Hammer",
    "Hose",               "Label",
    "LuggagePart",        "LuggageTag",
    "MetalPart",          "MetalSheet",
    "Nail",               "Nut",
    "PaintChip",          "Pen",
    "PlasticPart",        "Pliers",
    "Rock",               "Screw",
    "Screwdriver",        "SodaCan",
    "Tape",               "Washer",
    "Wire",               "Wood",
    "Wrench",
]
