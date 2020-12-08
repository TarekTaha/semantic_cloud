# semantic_cloud
Semantically identifiy objects and create a 3D color coded cloud

## Installation
This package is tested with TensorFlow 2.0.0, CUDA 10.1, and Pytorch 1.4.  

Pytorch installation in python 2 and python 3 
### Python 2
```sh
RUN pip install torch==1.4.0 torchvision==0.5.0 --no-cache-dir
```
### Python 3
```sh
pip3 install torch==1.4.0 torchvision==0.5.0 --no-cache-dir
```

Some required packges/modules`
```sh
pip install opencv-python==4.2.0.32
pip install Keras==2.3.1
pip install launchpadlib==1.10.6
pip install setuptools==41.0.0
pip install tensorflow==2.1.0
pip install cntk
pip install 'scikit-image<0.15'
pip install Theano

cd ~
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
git checkout tags/v0.6.5 -b v0.6.9
mkdir Build
cd Build
cmake .. -DCMAKE_BUILD_TYPE=Release # or Debug if you are investigating a crash
make
sudo make install
cd ..
# for pygpu
# This must be done after libgpuarray is installed as per instructions above.
python setup.py build
python setup.py install
sudo ldconfig

sudo apt-get install -y python-mako
apt-get install -y libsm6 libxext6 libxrender-dev

# Required for semantic_cloud
echo "arrow" | sudo -S apt-get install ros-melodic-jsk-rviz-plugins -y
```


## Build this package
* First make sure that the neural network model `pspnet_50_ade20k.pth` is copied to `~/catkin_ws/src/semantic_cloud/models_trained/` folder. You can download the model from [this link](https://drive.google.com/drive/folders/1yS92J8LU2PChqqeGUxb0km3lA0yBF7tg?usp=sharing), and extract the model from the `.zip` file.

```sh
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin config --merge-devel

cd ~/catkin_ws/src
# You need access to clone this package
git clone https://github.com/TarekTaha/semantic_cloud.git

cd ..
catkin build
```

## Testing this package
To test this package run the folowing launch file. 

```sh
roslaunch semantic_exploration semantic_explorer.launch
```

# Applications
## Exploration using PX4-powered drone
An application of this package for environmental exploration with semantic mapping using a PX4-powered drone is available in the [semantic_based_exploration](https://github.com/kucars/semantic_based_exploration)
