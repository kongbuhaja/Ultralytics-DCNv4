from ultralytics import YOLO
import time, os, argparse

parser = argparse.ArgumentParser(description='DYOLO')
parser.add_argument('--model', dest='model', type=str, default='yolov10n')
parser.add_argument('--data', dest='data', type=str, default='coco')
parser.add_argument('--gpus', dest='gpus', type=str, default='0', help='which device do you want to use')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

# model = YOLO(f"{args.model}")
model = YOLO('runs/detect/train2/weights/best.pt')
# model = YOLO('yolo11n.pt')


results = model.val(data = f'{args.data}.yaml')
