## Lens effects

### Summary
This folder uses glsl to generate lens effects on top of real-world images.
- The cpp executable aims to process a single image.
- The python script aims to batch processing images by calling the cpp executable.

### Prerequisite
```bash
sudo apt install -y \
    cmake libgflags-dev libegl1-mesa libgles2-mesa-dev \
    libopencv-dev python3-opencv
python3 -m pip install scipy joblib --user
```

### QuickStart
Compile and run the cpp executable
```bash
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=$PWD ..
make -j 4

# Install the compiled executable to your path.
make install
echo export PATH='$PATH':"$PWD/bin" >> ~/.bashrc
source  ~/.bashrc

# Run a sample
screen_dirt --input-image-path='../samples/input/305212385187.png'
popd
```

For the arguments of the each lens effect, please reference to the README in each lens effect folder.
- [Lens dirt](src/dirt/README.md)
- [Lens flare](src/flare/README.md)
- [Strong lens flare](src/strong_flare/README.md)
- [Fisheye effects](src/fisheye/README.md)
