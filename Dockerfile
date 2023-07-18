FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN apt-get upgrade -y
RUN apt-get install -y g++
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential cmake curl ca-certificates lsb-release \
libjpeg-dev libpng-dev ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 git vim sudo \
python3 python3-dev python3-pip && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt update
RUN apt install -y ros-melodic-desktop-full
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
RUN apt install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
RUN rosdep init && rosdep update

ENV FORCE_CUDA="1"
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker
#RUN pip install --no-cache-dir -e .
