#!/usr/bin/env python
"""
Take in an image (rgb or rgb-d)
Use CNN to do semantic segmantation
Out put a cloud point with semantic color registered
\author Xuan Zhang
\date May - July 2018
"""

from __future__ import division
from __future__ import print_function

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np

from sensor_msgs.msg import PointCloud2
from color_pcl_generator import PointType, ColorPclGenerator
import message_filters
import time

from skimage.transform import resize
import cv2

import torch
from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict

from semantic_cloud.msg import *
from semantic_cloud.srv import *
from jsk_rviz_plugins.msg import *
from std_msgs.msg import ColorRGBA, Float32

# SUNRGBD
labels_sunrgbd = ['backgroud', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
        'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
        'floor_mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower_curtain',
        'box', 'whiteboard', 'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag']

# Pascal VOC
labels_pascal = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
        'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

# ADE20K
# Reference: https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.txt
labels_ade20k = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']

def color_map(N=256, normalized=False):
    """
    Return Color Map in PASCAL VOC format (rgb)
    \param N (int) number of classes
    \param normalized (bool) whether colors are normalized (float 0-1)
    \return (Nx3 numpy array) a color map
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255.0 if normalized else cmap
    return cmap

def decode_segmap(temp, n_classes, cmap):
    """
    Given an image of class predictions, produce an bgr8 image with class colors
    \param temp (2d numpy int array) input image with semantic classes (as integer)
    \param n_classes (int) number of classes
    \cmap (Nx3 numpy array) input color map
    \return (numpy array bgr8) the decoded image with class colors
    """
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = cmap[l,0]
        g[temp == l] = cmap[l,1]
        b[temp == l] = cmap[l,2]
    bgr = np.zeros((temp.shape[0], temp.shape[1], 3))
    bgr[:, :, 0] = b
    bgr[:, :, 1] = g
    bgr[:, :, 2] = r
    return bgr.astype(np.uint8)

class SemanticCloud:
    """
    Class for ros node to take in a color image (bgr) and do semantic segmantation on it to produce an image with semantic class colors (chair, desk etc.)
    Then produce point cloud based on depth information
    CNN: PSPNet (https://arxiv.org/abs/1612.01105) (with resnet50) pretrained on ADE20K, fine tuned on SUNRGBD or not
    """
    def __init__(self, gen_pcl = True):
        """
        Constructor
        \param gen_pcl (bool) whether generate point cloud, if set to true the node will subscribe to depth image
        """
        # Get point type
        point_type = rospy.get_param('/semantic_pcl/point_type')
        if point_type == 0:
            self.point_type = PointType.COLOR
            print('Generate color point cloud.')
        elif point_type == 1:
            self.point_type = PointType.SEMANTICS_MAX
            print('Generate semantic point cloud [max fusion].')
        elif point_type == 2:
            self.point_type = PointType.SEMANTICS_BAYESIAN
            print('Generate semantic point cloud [bayesian fusion].')
        else:
            print("Invalid point type.")
            return
        self.real_sense  = rospy.get_param('/semantic_pcl/real_sense')
        self.labels_pub  = rospy.Publisher("/semantic_pcl/labels", OverlayText, queue_size=1)
        self.labels_list = []
        self.text = OverlayText()
        # Get image size
        self.img_width, self.img_height = rospy.get_param('/camera/width'), rospy.get_param('/camera/height')
	self.throttle_rate = rospy.get_param('/semantic_pcl/throttle_rate')
	self.last_time = rospy.Time.now()
        # Set up CNN is use semantics
        if self.point_type is not PointType.COLOR:
            print('Setting up CNN model...')
            # Set device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Get dataset
            self.dataset = rospy.get_param('/semantic_pcl/dataset')
            # Setup model
            model_name ='pspnet'
            model_path = rospy.get_param('/semantic_pcl/model_path')
            if self.dataset == 'sunrgbd': # If use version fine tuned on sunrgbd dataset
                self.n_classes = 38 # Semantic class number
                self.model = get_model(model_name, self.n_classes, version = 'sunrgbd_res50')
                state = torch.load(model_path)
                self.model.load_state_dict(state)
                self.cnn_input_size = (321, 321)
                self.mean = np.array([104.00699, 116.66877, 122.67892]) # Mean value of dataset
            elif self.dataset == 'ade20k':
                self.n_classes = 150 # Semantic class number
                self.model = get_model(model_name, self.n_classes, version = 'ade20k')
                state = torch.load(model_path)
                self.model.load_state_dict(convert_state_dict(state['model_state'])) # Remove 'module' from dictionary keys
                self.cnn_input_size = (473, 473)
                self.mean = np.array([104.00699, 116.66877, 122.67892]) # Mean value of dataset
            self.model = self.model.to(self.device)
            self.model.eval()
            self.cmap = color_map(N = self.n_classes, normalized = False) # Color map for semantic classes
        # Declare array containers
        if self.point_type is PointType.SEMANTICS_BAYESIAN:
            self.semantic_colors = np.zeros((3, self.img_height, self.img_width, 3), dtype = np.uint8) # Numpy array to store 3 decoded semantic images with highest confidences
            self.confidences = np.zeros((3, self.img_height, self.img_width), dtype = np.float32) # Numpy array to store top 3 class confidences
        # Set up ROS
        print('Setting up ROS...')
        self.bridge = CvBridge() # CvBridge to transform ROS Image message to OpenCV image
        # Semantic image publisher
        self.sem_img_pub = rospy.Publisher("/semantic_pcl/semantic_image", Image, queue_size = 1)
        # Set up ros image subscriber
        # Set buff_size to average msg size to avoid accumulating delay
        if gen_pcl:
            # Point cloud frame id
            frame_id = rospy.get_param('/semantic_pcl/frame_id')
            # Camera intrinsic matrix
            fx = rospy.get_param('/camera/fx')
            fy = rospy.get_param('/camera/fy')
            cx = rospy.get_param('/camera/cx')
            cy = rospy.get_param('/camera/cy')
            intrinsic = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype = np.float32)
            self.pcl_pub = rospy.Publisher("/semantic_pcl/semantic_pcl", PointCloud2, queue_size = 1)
            self.color_sub = message_filters.Subscriber(rospy.get_param('/semantic_pcl/color_image_topic'), Image, queue_size = 1, buff_size = 30*480*640)
            self.depth_sub = message_filters.Subscriber(rospy.get_param('/semantic_pcl/depth_image_topic'), Image, queue_size = 1, buff_size = 40*480*640 ) # increase buffer size to avoid delay (despite queue_size = 1)
            self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size = 1, slop = 0.3) # Take in one color image and one depth image with a limite time gap between message time stamps
            self.ts.registerCallback(self.color_depth_callback)
            self.cloud_generator = ColorPclGenerator(intrinsic, self.img_width,self.img_height, frame_id , self.point_type)
        else:
            self.image_sub = rospy.Subscriber(rospy.get_param('/semantic_pcl/color_image_topic'), Image, self.color_callback, queue_size = 1, buff_size = 30*480*640)

        semantic_colored_labels_srv = rospy.Service('get_semantic_colored_labels', GetSemanticColoredLabels, self.get_semantic_colored_labels)
        print('Ready.')

    def get_semantic_colored_labels(self,GetSemanticColoredLabels):
        print("Get Semantic Colored Labels Service called")
        ret  = GetSemanticColoredLabelsResponse()
        scls = SemanticColoredLabels()
        for i in range(0,self.n_classes):
            scl = SemanticColoredLabel()
            if self.dataset == 'sunrgbd':
                label = labels_sunrgbd[i]
            elif self.dataset == 'ade20k':
                label = labels_ade20k[i]
            scl.label = label
            scl.color_r = self.cmap[i,0]
            scl.color_g = self.cmap[i,1]
            scl.color_b = self.cmap[i,2]
            scls.semantic_colored_labels.append(scl)
        ret = scls
        return ret

    def get_label(self,pred_label):
        print(" ============= ")
        unique_labels = np.unique(pred_label)
        count = 0
        
        self.text.width = 200
        self.text.height = 800
        self.text.left = 10
        self.text.top = 10 + 20* count       
        self.text.text_size = 12
        self.text.line_width = 2 
        self.text.font = "DejaVu Sans Mono"
        self.text.fg_color = ColorRGBA(25 / 255.0, 1.0, 240.0 / 255.0, 1.0)
        self.text.bg_color = ColorRGBA(255,255,255, 0.5)
        for color_index in unique_labels:
            label = ''
            if self.dataset == 'sunrgbd':
                label = labels_sunrgbd[color_index]
            elif self.dataset == 'ade20k':
                label = labels_ade20k[color_index]
            count+=1
            if not label in self.labels_list:
                self.labels_list.append(label)
                self.text.text += """<span style="color: rgb(%d,%d,%d);">%s</span>
                """ % (self.cmap[color_index,0],self.cmap[color_index,1], self.cmap[color_index,2],label.capitalize())
            print("Published Label Name with index:" + str(color_index) + " is:" + label)
        self.labels_pub.publish(self.text)

    def color_callback(self, color_img_ros):
        """
        Callback function for color image, de semantic segmantation and show the decoded image. For test purpose
        \param color_img_ros (sensor_msgs.Image) input ros color image message
        """
        print('callback')
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "bgr8") # Convert ros msg to numpy array
        except CvBridgeError as e:
            print(e)
        # Do semantic segmantation
        class_probs = self.predict(color_img)
        confidence, label = class_probs.max(1)
        confidence, label = confidence.squeeze(0).numpy(), label.squeeze(0).numpy()
        label = resize(label, (self.img_height, self.img_width), order = 0, mode = 'reflect', preserve_range = True) # order = 0, nearest neighbour
        label = label.astype(np.int)
        # Add semantic class colors
        decoded = decode_segmap(label, self.n_classes, self.cmap)        # Show input image and decoded image
        confidence = resize(confidence, (self.img_height, self.img_width),  mode = 'reflect', preserve_range = True)
        cv2.imshow('Camera image', color_img)
        cv2.imshow('confidence', confidence)
        cv2.imshow('Semantic segmantation', decoded)
        cv2.waitKey(3)

    def color_depth_callback(self, color_img_ros, depth_img_ros):
        """
        Callback function to produce point cloud registered with semantic class color based on input color image and depth image
        \param color_img_ros (sensor_msgs.Image) the input color image (bgr8)
        \param depth_img_ros (sensor_msgs.Image) the input depth image (registered to the color image frame) (float32) values are in meters
        """
        tic = rospy.Time.now()
        diff = tic -  self.last_time
        if diff.to_sec() < self.throttle_rate:
            return
	
        self.last_time = rospy.Time.now()
        # Convert ros Image message to numpy array
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_ros, "32FC1")
        except CvBridgeError as e:
            print(e)
        # Resize depth
        if depth_img.shape[0] is not self.img_height or depth_img.shape[1] is not self.img_width:
            depth_img = resize(depth_img, (self.img_height, self.img_width), order = 0, mode = 'reflect', preserve_range = True) # order = 0, nearest neighbour
            depth_img = depth_img.astype(np.float32)
            # realsense camera gives depth measurements in mm
            if self.real_sense:
                depth_img = depth_img /1000.0
        if self.point_type is PointType.COLOR:
            cloud_ros = self.cloud_generator.generate_cloud_color(color_img, depth_img, color_img_ros.header.stamp)
        else:
            # Do semantic segmantation
            if self.point_type is PointType.SEMANTICS_MAX:
                semantic_color, pred_confidence = self.predict_max(color_img)
                cloud_ros = self.cloud_generator.generate_cloud_semantic_max(color_img, depth_img, semantic_color, pred_confidence, color_img_ros.header.stamp)

            elif self.point_type is PointType.SEMANTICS_BAYESIAN:
                self.predict_bayesian(color_img)
                # Produce point cloud with rgb colors, semantic colors and confidences
                cloud_ros = self.cloud_generator.generate_cloud_semantic_bayesian(color_img, depth_img, self.semantic_colors, self.confidences, color_img_ros.header.stamp)

            # Publish semantic image
            if self.sem_img_pub.get_num_connections() > 0:
                if self.point_type is PointType.SEMANTICS_MAX:
                    semantic_color_msg = self.bridge.cv2_to_imgmsg(semantic_color, encoding="bgr8")
                    #cvtColor(semantic_color_msg, semantic_color_msg, BGR2RGB);
                else:
                    semantic_color_msg = self.bridge.cv2_to_imgmsg(self.semantic_colors[0], encoding="bgr8")
                self.sem_img_pub.publish(semantic_color_msg)

        # Publish point cloud
        self.pcl_pub.publish(cloud_ros)

    def predict_max(self, img):
        """
        Do semantic prediction for max fusion
        \param img (numpy array rgb8)
        """
        class_probs = self.predict(img)
        # Take best prediction and confidence
        pred_confidence, pred_labels = class_probs.max(1)
        pred_confidence = pred_confidence.squeeze(0).cpu().numpy()
        pred_labels = pred_labels.squeeze(0).cpu().numpy()
        pred_labels = resize(pred_labels, (self.img_height, self.img_width), order = 0, mode = 'reflect', preserve_range = True) # order = 0, nearest neighbour
        pred_labels = pred_labels.astype(np.int)
        # Add semantic color
        semantic_color = decode_segmap(pred_labels, self.n_classes, self.cmap)
        pred_confidence = resize(pred_confidence, (self.img_height, self.img_width),  mode = 'reflect', preserve_range = True)
        self.get_label(pred_labels)
        return (semantic_color, pred_confidence)

    def predict_bayesian(self, img):
        """
        Do semantic prediction for bayesian fusion
        \param img (numpy array rgb8)
        """
        class_probs = self.predict(img)
        # Take 3 best predictions and their confidences (probabilities)
        pred_confidences, pred_labels  = torch.topk(input = class_probs, k = 3, dim = 1, largest = True, sorted = True)
        pred_labels = pred_labels.squeeze(0).cpu().numpy()
        pred_confidences = pred_confidences.squeeze(0).cpu().numpy()
        # Resize predicted labels and confidences to original image size
        for i in range(pred_labels.shape[0]):
            pred_labels_resized = resize(pred_labels[i], (self.img_height, self.img_width), order = 0, mode = 'reflect', preserve_range = True) # order = 0, nearest neighbour
            pred_labels_resized = pred_labels_resized.astype(np.int)
            # Add semantic class colors
            self.semantic_colors[i] = decode_segmap(pred_labels_resized, self.n_classes, self.cmap)
        for i in range(pred_confidences.shape[0]):
            self.confidences[i] = resize(pred_confidences[i], (self.img_height, self.img_width),  mode = 'reflect', preserve_range = True)
        self.get_label(pred_labels)

    def predict(self, img):
        """
        Do semantic segmantation
        \param img: (numpy array bgr8) The input cv image
        """
        img = img.copy() # Make a copy of image because the method will modify the image
        #orig_size = (img.shape[0], img.shape[1]) # Original image size
        # Prepare image: first resize to CNN input size then extract the mean value of SUNRGBD dataset. No normalization
        img = resize(img, self.cnn_input_size, mode = 'reflect', preserve_range = True) # Give float64
        img = img.astype(np.float32)
        img -= self.mean
        # Convert HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Convert to tensor
        img = torch.tensor(img, dtype = torch.float32)
        img = img.unsqueeze(0) # Add batch dimension required by CNN
        with torch.no_grad():
            img = img.to(self.device)
            # Do inference
            since = time.time()
            outputs = self.model(img) #N,C,W,H
            # Apply softmax to obtain normalized probabilities
            outputs = torch.nn.functional.softmax(outputs, 1)
            return outputs

def main(args):
    rospy.init_node('semantic_cloud', anonymous=True)
    seg_cnn = SemanticCloud(gen_pcl = True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
