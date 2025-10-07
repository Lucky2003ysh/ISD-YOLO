import torch
from thop import profile
from ultralytics import YOLO


def val_model():
    # Load a model
    model = YOLO("runs/detect/train225/weights/best.pt") #Place the path to the training weights file
    metrics = model.val(data="data_yaml/neu.yaml")#dataset
if __name__ == '__main__':
    val_model()