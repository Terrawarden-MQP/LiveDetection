# ROS2 imports 
# import rclpy
# from rclpy.node import Node

# # CV Bridge and message imports
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# from vision_msgs.msg import ObjectHypothesisWithPose, BoundingBox2D, Detection2D, Detection2DArray
# from cv_bridge import CvBridge, CvBridgeError

# from joisie_vision.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
# from joisie_vision.misc import Timer

import cv2
import numpy as np
import os

from torch2trt import torch2trt
from torch2trt import TRTModule
import torch


def create_TRT_module():
    model_path = os.getenv("HOME")+ '/Terrawarden/TerrawardenVision/joisie_vision/TACO-Trained-Epoch2.pth'
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.load_state_dict(torch.load(model_path))
    model.eval().cuda()

    x = torch.ones((1,3,384,384)).cuda()

    print("Creating TRT version...........")
    model_trt = torch2trt(model, [x])
    print("Created TRT version.......")

    save_location = os.getenv("HOME") + '/ros2_models/TACO_trt.pth'

    print("Saving TRT model......")
    torch.save(model_trt.state_dict(), save_location)

create_TRT_module()