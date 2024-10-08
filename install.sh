conda create -n dcnv4 python=3.8 -y
conda activate dcnv4

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
sh DCNv4_op/make.sh

pip install ultralytics