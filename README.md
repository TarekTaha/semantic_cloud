# semantic_cloud
A ROS package which semantically identifies objects and creates/publishes a 3D color coded cloud

## Installation
This package is tested with Ubuntu 18, ROS melodic, TensorFlow 2.0.0, CUDA 10.1, and Pytorch 1.4.  

Pytorch installation in python 2 and python 3 
### Python 2
```sh
pip install --user torch==1.4.0 torchvision==0.5.0 --no-cache-dir
```
### Python 3
```sh
pip3 install --user torch==1.4.0 torchvision==0.5.0 --no-cache-dir
```

Some required packges/modules`
```sh
pip install setuptools==41.0.0
pip install opencv-python==4.2.0.32

# Required for semantic_cloud, assuming ROS melodic
sudo apt-get install ros-melodic-jsk-rviz-plugins -y
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
roslaunch semantic_cloud semantic_mapping.launch
```

# Applications
## Exploration using PX4-powered drone
An application of this package for environmental exploration with semantic mapping using a PX4-powered drone is available in the [semantic_based_exploration](https://github.com/kucars/semantic_based_exploration)
