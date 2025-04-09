'''Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'''


# ROS2 imports 
import rclpy
from rclpy.node import Node

# CV Bridge and message imports
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2D, Detection2DArray
from geometry_msgs.msg import Point, PoseWithCovariance, Pose
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO

from joisie_vision.misc import Timer

import cv2
import numpy as np
import os

from torch2trt import torch2trt
from torch2trt import TRTModule
import torch

class YOLODetectionNode(Node):

    def __init__(self):
        super().__init__('yolo_detection_node')

        # Create a subscriber to the Image topic
        self.declare_parameter('topic', 'image')
        topic_name = self.get_parameter('topic')
        self.subscription = self.create_subscription(Image, topic_name.value, self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        # Create a Detection 2D array topic to publish results on
        self.detection_publisher = self.create_publisher(Detection2D, 'joisie_detection', 10)
        self.point_publisher = self.create_publisher(Point, "detected_object_centroid", 10)

        # Create an Image publisher for the results
        self.result_publisher = self.create_publisher(Image,'yolo_detection_image',10)


        self.class_names = ["Cans"]
        self.num_classes = len(self.class_names)
        
        self.declare_parameter('model_path', '/home/joisie/Desktop/ros_ws/src/TerrawardenVision/joisie_vision/joisie_vision/turing_canet_v2.pt')

        model_path =  self.get_parameter('model_path').value
        if (os.path.isfile(model_path)):
            print("Module exists, loading..")        
        else:
            raise FileNotFoundError(f"{model_path} does not exist")
            
        self.declare_parameter('show', "false")
        self.show = self.get_parameter('show').value.lower() == "true"
            
        self.yolo = YOLO(model_path) 
            
        self.timer = Timer()
        self.time_array = []
        self.num_times = 250


    def listener_callback(self, data):
        self.get_logger().info("Received an image! ")
        try:
            # Extract the image from the rosmsg to a regular format using the cv bridge
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.timer.start()
            result = self.yolo.predict(image, 10, 0.4)
            interval = self.timer.end()
            if len(self.time_array) >= self.num_times:
                self.time_array.pop(0)
            self.time_array.append(interval)

            self.get_logger().info('Time: {:.3f}s, Avg Time: {:.3f}s Detect Objects: {:d}.'.format(interval, np.average(self.time_array), result.boxes.size(0)))
            
            if result.boxes.size(0) > 0:
                # Process results list
                boxes = result.boxes  # Boxes object for bounding box outputs
                probs = result.probs  # Probs object for classification outputs

                index_highest_prob = probs.top1
                highest_prob = probs.top1conf

                # Publishing the results onto the the Detection2D vision_msgs format
                # self.detection_publisher.publish(largest_detect[1])
                point = Point()
                x,y,w,h = tuple(boxes.xywhn[index_highest_prob])
                point.x = x
                point.y = y
                self.get_logger().info(f"Publishing object: {point}")
                self.point_publisher.publish(point)

                detection_msg = Detection2D()
                detection_msg.bbox.center.x = x
                detection_msg.bbox.center.y = y
                detection_msg.bbox.size_x = w
                detection_msg.bbox.size_y = h
                detection_msg.source_img = data
                detection_msg.results = [ObjectHypothesisWithPose(id=0, score=highest_prob, pose=PoseWithCovariance(pose=Pose(position=Point(x=x,y=y))))]
                detection_msg.header.time = self.get_clock().now()
                self.detection_publisher.publish(detection_msg)

                results_image = result.plot()
                # Displaying the predictions
                if self.show:
                    results_image.show()
                ros_image = self.bridge.cv2_to_imgmsg(results_image, encoding="bgr8")
                ros_image.header.frame_id = 'camera_frame'
                self.image_publisher.publish(ros_image)
        except CvBridgeError as e:
          print(e)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    yolo_detection_node = YOLODetectionNode()

    rclpy.spin(yolo_detection_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    yolo_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
