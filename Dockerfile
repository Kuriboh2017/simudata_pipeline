# Start from Ubuntu 20.04
FROM ubuntu:20.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Update the package lists
RUN apt-get update && apt-get install -y \
    build-essential tmux htop git time \
    cmake libgflags-dev python3-pip \
    sudo vim

RUN apt-get update && apt-get install -y \
    libegl1-mesa libgles2-mesa-dev \
    libopencv-dev python3-opencv \
    python3-pyqt5

# Create a new user 'sim' with password 'sim'
RUN useradd -m sim && echo "sim:sim" | chpasswd && adduser sim sudo
RUN usermod -s /bin/bash sim

# Switch to the new user
USER sim

# Update pip
RUN python3 -m pip install --upgrade pip

# Install scipy using pip
RUN python3 -m pip install scipy lz4 colour-demosaicing scikit-image numba

RUN echo -e "\"\\e[A\": history-search-backward\n\"\\e[B\": history-search-forward" > /home/sim/.inputrc

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog