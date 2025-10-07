from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def train_model():
    # Load a model
    model = YOLO("ISD-YOLO.yaml").load("yolo11n.pt")
    # Train the model
    results = model.train(data="data_yaml/pcb.yaml", epochs=1800, imgsz=640, plots=True, patience=200, seed=0)

if __name__ == '__main__':
    train_model()