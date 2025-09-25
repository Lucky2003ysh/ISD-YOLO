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

Our training results are all saved in the **data validation** folder. You can verify the experimental results in the following way:

```python
import torch
from thop import profile
from ultralytics import YOLO


def val_model():
    # Load a model
    model = YOLO("weights.pt") #Place the path to the training weights file
    metrics = model.val(data="data_yaml.yaml")#dataset
if __name__ == '__main__':
    val_model()
```


