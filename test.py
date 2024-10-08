from ultralytics import YOLO
import torch
import argparse

parser = argparse.ArgumentParser(description='DYOLO')
parser.add_argument('--model', dest='model', type=str, default='yolov10n')
parser.add_argument('--data', dest='data', type=str, default='coco')
parser.add_argument('--gpus', dest='gpus', type=str, default='0', help='which device do you want to use')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

model = YOLO(f"{args.model}.yaml").cuda()

args = dict(
    data = f'{args.data}.yaml',
    batch=64,
    epochs=500,
    imgsz=640,
    device='0',
    val=True,
    save_period=1
)
train_results = model.train(**args)

metrics = model.val()
