import argparse
import os

parser = argparse.ArgumentParser(description='DYOLO')
parser.add_argument('--model', dest='model', type=str, default='yolov10n')
parser.add_argument('--load_file', dest='load_file', type=str, default='')
parser.add_argument('--data', dest='data', type=str, default='coco')
parser.add_argument('--gpus', dest='gpus', type=str, default='0', help='which device do you want to use')
parser.add_argument('--epochs', dest='epochs', type=int, default=500)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

from ultralytics import YOLO
import torch

if torch.cuda.is_available():
    device = f"cuda:{args.gpus}"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA is not available. Using CPU.")

model_file = args.load_file if args.load_file else f"{args.model}.yaml"
model = YOLO(f"{model_file}").cuda()

train_args = dict(
    data = f'{args.data}.yaml',
    batch=128,
    epochs=args.epochs,
    imgsz=640,
    device=0,
    val=True,
    save_period=1
)
train_results = model.train(**train_args)

metrics = model.val()
