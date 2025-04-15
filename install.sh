# conda create -n dcnv4 python=3.8 -y
# conda activate dcnv4

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
cd ultralytics/nn/modules/DCNv4_op/
sh make.sh
cd ../../../../

pip install ultralytics
