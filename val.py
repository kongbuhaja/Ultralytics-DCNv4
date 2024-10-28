from ultralytics import YOLO
import time, os, argparse, psutil

parser = argparse.ArgumentParser(description='DYOLO')
parser.add_argument('--model', dest='model', type=str, default='yolov10n')
parser.add_argument('--data', dest='data', type=str, default='coco')
parser.add_argument('--cpus', dest='cpus', type=str, default='0-15', help='How many cpus do you want to use')
parser.add_argument('--gpus', dest='gpus', type=str, default='0', help='which device do you want to use')

args = parser.parse_args()

cpu_range = [int(core) for core in args.cpus.split('-')]
p = psutil.Process()
p.cpu_affinity(cpu_range)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

model = YOLO(f"{args.model}")

results = model.val(data = f'{args.data}.yaml')
