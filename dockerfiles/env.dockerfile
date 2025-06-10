FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $>TZ > /etc/timezone

ENV DEBIAN_FRONTEND=noninteractive

# apt
RUN echo "== Install Basic Tools ==" &&\
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y --allow-unauthenticated \
        openssh-server vim nano htop tmux sudo \
        git unzip build-essential iputils-ping net-tools ufw \
        python3.8 python3.8-venv python3.8-dev python3-pip \
        curl dpkg libgtk2.0-dev cmake libwebp-dev ca-certificates gnupg git \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
        libatlas-base-dev gfortran \
        libgl1-mesa-glx libglu1-mesa-dev x11-utils x11-apps && \
    apt clean &&\
    rm -rf /var/lib/apt/list/*

# Set Python 3.8 to default python 
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --set python3 /usr/bin/python3.8

WORKDIR /home

