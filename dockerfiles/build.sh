sudo docker build --force-rm -f {PATH}/Ultralytics-DCNv4/dockerfiles/env.dockerfile -t ud:latest .
sudo docker run --gpus '"device=0"' --cpuset-cpus=0-15 -m 64g --shm-size=16g -it -v {PATH}:/home/ML--name ud ud:latest
