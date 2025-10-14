# 1. requirements and usage

We propose an ISD-YOLO, and more information are in our paper:`**Enhanced Industrial Surface Defect Detection with YOLOv11: A Lightweight Feature Enhancement and Adaptive Weighting Approach**`

## 1.1 requirements

Install the `ultralytics` package, including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml), in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```
pip install ultralytics
```

## 1.2 usage

Our code are in ISD-YOLO document, you can train the model by using train.py, such as :

```python
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def train_model():
    # Load a model
    model = YOLO("ISD-YOLO.yaml").load("yolo11n.pt")
    # Train the model
    results = model.train(data="data_yaml/pcb.yaml", epochs=1200, imgsz=640, plots=True, patience=200, seed=0)

if __name__ == '__main__':
    train_model()
```

------

# 2. Description and Implementation of Key Algorithms

## 2.1 LSCMDC module

**Code located in:** `ultralytics/nn/Conv/LSCMDC.py`

**Description**

We propose a LSCMDC module. Directly using traditional convolution for downsampling results in detail loss, while a purely lightweight design sacrifices feature expression capabilities. Therefore, the module first designs a Space-to-Channel (SPC) mechanism to achieve downsampling without losing learnable information. Considering that surface defects in industrial images follow a certain spatial distribution, multi-directional convolution and feature fusion are used to enhance defect feature extraction.The workflow of the LSCMDC module is shown in Fig. 1.

<img width="865" height="485" alt="image" src="https://github.com/user-attachments/assets/a524d3d5-fdb0-4755-af56-658bc5e84cb9" />


​                                                                                                                 Fig 1. LSCMDC module

## 2.2 GLAM

**Code located in:** `ultralytics/nn/attention/attention.py` 

**Description**

GLAM draws on the sequential channel and spatial attention architecture of CBAM and GAM, and reconstructs their sub-modules to tackle the current problems in industrial defect detection, such as the need for lightweight design and insufficient global information interaction. The overall processing flow can be referred to as shown in Fig. 2.

<img width="865" height="291" alt="image" src="https://github.com/user-attachments/assets/e60af887-d8a2-47a6-96f4-eda9679dd134" />


​                                                                                                                          Fig 2. GLAM

## 2.3 AW loss

**Code located in:** `ultralytics/utils/loss.py/class AWLoss` 

**Description**

The function curve of the AW Loss is shown in Fig. 3.

<img width="263" height="263" alt="image" src="https://github.com/user-attachments/assets/73b711db-2125-4702-b069-fdc046af9f08" />


​                                                                                                                               Fig 3. AW loss

---

# 3. Other Details

All datasets used in our work are open-source and were obtained from:
（1）PCB

```
Huang W, Wei P, Zhang M, et al. HRIPCB: a challenging dataset for PCB defects detection and classification[J]. The Journal of Engineering, 2020, 2020(13): 303-309.
```

（2）NEU

```
Li Z, Wei X, Hassaballah M, et al. A deep learning model for steel surface defect detection[J]. Complex & Intelligent Systems, 2024, 10(1): 885-897.
```

（3）GC10

```
Lv X, Duan F, Jiang J, et al. Deep metallic surface defect detection: The new benchmark and detection network[J]. Sensors, 2020, 20(6): 1562.
```
The datasets used in the training process are available via Baidu Netdisk: https://pan.baidu.com/s/1hm01W3wOpAyVGOvI6CUP5A?pwd=tqxg


Our training results are all saved in the **data validation** folder. You can verify the experimental results in the following way:

```python
import torch
from thop import profile
from ultralytics import YOLO

def val_model():
    # Load a model
    model = YOLO("best.pt") #Place the path to the training weights file
    metrics = model.val(data="data_yaml/neu.yaml")#dataset
if __name__ == '__main__':
    val_model()
```

Please feel free to contact me if you have any questions or require additional information (e.g., weights of experimental results, etc.):

📧 gmail: [ysh3209396834@gmail.com]

# 4. Citation

**Paper Title**:  `Enhanced Industrial Surface Defect Detection with YOLOv11: A Lightweight Feature Enhancement and Adaptive Weighting Approach` 
**Journal**: None 

