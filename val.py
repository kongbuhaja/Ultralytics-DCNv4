import time, os, argparse, psutil

parser = argparse.ArgumentParser(description='DYOLO')
parser.add_argument('--model', dest='model', type=str, default='yolov10n')
parser.add_argument('--data', dest='data', type=str, default='coco')
parser.add_argument('--cpus', dest='cpus', type=str, default='0-15', help='How many cpus do you want to use')
parser.add_argument('--gpus', dest='gpus', type=str, default='0', help='which device do you want to use')

args = parser.parse_args()

cores = [int(core) for core in args.cpus.split('-')]
p = psutil.Process()
p.cpu_affinity(list(range(cores[0], cores[1]+1)))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

from ultralytics import YOLO
import torch 

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU(s): {args.gpus}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

model_file = args.model if '.pt' in args.model else f'{args.model}.yaml'
model = YOLO(f"{model_file}")

results = model.val(data = f'{args.data}.yaml')

model.info(verbose=1)
print(f"metrics/mAP50-95(B): {results.results_dict['metrics/mAP50-95(B)']}")
print(f"avg pre-processing time: {results.speed['preprocess']:.4f} ms")
print(f"avg inference time: {results.speed['inference']:.4f} ms")
print(f"avg post-processing time: {results.speed['postprocess']:.4f} ms")
print(f'avg total time:{sum([results.speed['preprocess'], 
                             results.speed['inference'],
                             results.speed['postprocess']]):.4f}')
print(f"FPS: {1000/sum([results.speed['preprocess'], 
                        results.speed['inference'],
                        results.speed['postprocess']]), :.2f}")