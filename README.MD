# DeCo_YOLO: YOLOs with Deformable Convolution
This code is based on Utralytics(AGPL-3.0) and DCNv4(MIT)

![graph](/images/graph.jpg)

<details>
<summary>Abstract</summary>
In the robot industry, many researchers try to maximize the performance of object detection in limited environments like time and resources. So, in this paper we propose a combination of the most commonly used You Look Only Once (YOLO) in object detection field and deformable convolution network (DCN) modified from original convolution operator. Especially, DCN is accelerated in DCNv4 compared to previous DCN versions by using customized CUDA code. The experimental results show that while the proposed models have parameters and slightly increased inference time, they outperform the original models in performance. For example, compared to their respective base model, the proposed DeCo-YOLO8n, DeCO-YOLO10n and DeCo-YOLO11n achieves 38.6(+1.2) 39.4(+0.9), and 39.1(-0.4) AP<sup>val</sup> , respectively. Based on experiments, we can consider using various existing object detection models with DCN to maximize performance in limited environments like real-time object detection. 
</details>

## Performance
| Model (NMS) |	Params | AP^val^ | Total time | Pre-process time | Inference time | Post-process time | FPS |  
| :---------------| :---: | :------: | :------: | :------: | :------: | :------: | :------: |
| YOLO8n (O) | 3.15 M | 37.4 | 0.8518 ms | 0.0668 ms | 0.5054 ms | 0.2796 ms | 1174 |
| DeCo-YOLO8n (O) | 2.94 M | **38.6** | 0.8532 ms | 0.0658 ms | 0.5196 ms | 0.2678 ms | 1172 |
| YOLO10n (X) | 2.76 M | 38.5 | 0.7283 ms | 0.0674 ms | 0.6296 ms | 0.0313 ms | 1373 |
| DeCo-YOLO10n (X) | 2.98 M | **39.4** | 0.7638 ms | 0.0661 ms | 0.6642 ms | 0.0334 ms | 1309 |
<!-- | YOLO11n (O) | 2.62 M | **39.5** | 0.8970 ms | 0.0693 ms | 0.5641 ms | 0.2637 ms | 1115 |
| DeCo-YOLO11n (O) | 2.78 M | 39.1 | 0.9308 ms | 0.0671 ms | 0.6030 ms | 0.2607 ms | 1074 | -->

## Installation
```conda``` virtual environment is recommended.
```bash
conda create -n deco-yolo python=3.8
conda activate deco-yolo
sh install.sh
```

## Demo
```bash
python train.py --model deco-yolo{8/10/11}n --data coco --gpus 0 --cpus 0-23 --epochs 1000
python val.py --model "*.pt path" --data coco --gpus 0 cpus 0-23
```
or
```bash
sh train.sh
sh val.sh
```
or
```python
from ultralytics import YOLO
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = YOLO("deco-yolo{8/10/11}n").to(device)
model.train(data='coco.yaml', batch=64, imgsz=640)
model.val(data='coco.yaml', batch=64)
```

## Acknowledgement
The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics) and [DCNv4](https://github.com/OpenGVLab/DCNv4).
