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
from vision_msgs.msg import ObjectHypothesisWithPose, ObjectHypothesis, Detection2D, Detection2DArray
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

        self.declare_parameter('centroid_topic', 'detected_object_centroid')
        self.point_publisher = self.create_publisher(Point, self.get_parameter('centroid_topic').value, 10)

        # Create an Image publisher for the results
        self.image_publisher = self.create_publisher(Image,'yolo_detection_image',10)

        self.class_names = ["Cans"]
        self.num_classes = len(self.class_names)
        
        self.declare_parameter('model_path', '/home/joisie/Desktop/ros_ws/src/TerrawardenVision/joisie_vision/joisie_vision/turing_canet_v2.engine')

        model_path =  self.get_parameter('model_path').value
        if (os.path.isfile(model_path)):
            print("Module exists, loading..")        
        else:
            raise FileNotFoundError(f"{model_path} does not exist")
            
        self.declare_parameter('show', False)
        self.show = self.get_parameter('show').value
            
        self.yolo = YOLO(model_path) 
        if ".pt" in model_path:
            self.yolo.cuda()

        self.receiving_info = False
            
        self.timer = Timer()
        self.time_array = []
        self.num_times = 250

        # Detections below this probability will not be communicated over ROS
        self.min_probability = 0.5
        # Detections between this probability and the minimum will log a warning
        # But will still send the centroid
        self.warn_probability = 0.65

        # Minimum size measured from a picture of a can at 4m from camera
        self.min_size = (0.01, 0.035)


    def listener_callback(self, data):
        # self.get_logger().info("Received an image! ")
        if not self.receiving_info:
            self.get_logger().info("Detection online and receiving messages!")
            self.receiving_info = True
        try:
            # Extract the image from the rosmsg to a regular format using the cv bridge
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.timer.start()
            results = self.yolo.predict(source=image) # 10, 0.4
            interval = self.timer.end()
            if len(self.time_array) >= self.num_times:
                self.time_array.pop(0)
            self.time_array.append(interval)

            if len(results) == 0:
                # If this gets printed idek
                self.get_logger().warn(f'Time: {interval:.3f}s, Avg Time: {np.average(self.time_array):.3f}s No Detection Found.')
                return
            result = results[0]

            if interval > 0.05:
                self.get_logger().warn(f'Detection took longer than expected! Time: {interval:.3f}s, Avg Time: {np.average(self.time_array):.3f}s Detect Objects: {result.boxes.data.size(0)}')
            
            results_image = result.plot()
            # Displaying the predictions
            if self.show:
                cv2.imshow("YOLO Detection Node", results_image)
            ros_image = self.bridge.cv2_to_imgmsg(results_image, encoding="rgb8")
            ros_image.header.frame_id = 'camera_frame'
            self.image_publisher.publish(ros_image)
            
            
            if result.boxes.data.size(0) == 0:
                # Don't publish anything if no point seen
                return
            
            # Process results list
            boxes = result.boxes  # Boxes object for bounding box outputs

            index_highest_prob = torch.argmax(boxes.conf)
            highest_prob = float(boxes.conf[index_highest_prob])

            # If the highest prob box has a prob below the minimum
            # do not send it to extract cluster or task manager
            if highest_prob < self.min_probability:
                self.get_logger().warn(f'Detection probability very low ({highest_prob})! Not sending point to extract!')
                return
            elif highest_prob < self.warn_probability:
                self.get_logger().warn(f'Detection probability low ({highest_prob})!')

            # Publishing the results onto the the Detection2D vision_msgs format
            # self.detection_publisher.publish(largest_detect[1])
            point = Point()
            # x, y, w, and h are all normalized
            x, y, w, h = tuple(boxes.xywhn[index_highest_prob])

            if w < self.min_size[0] or h < self.min_size[1]:
                self.get_logger().warn(f"Box with probability {highest_prob} and size {(w,h)} deemed too small to be a can!")
                return
            # self.get_logger().info(f"Most likely box found at {(x,y)} with size {(w,h)}")

            # Point sent is in pixels (but still a float)
            point.x = float(x*data.width)
            point.y = float(y*data.height)
            # self.get_logger().info(f"Publishing object: {point}")
            self.point_publisher.publish(point)

            detection_msg = Detection2D()
            detection_msg.bbox.center.position.x = float(x)
            detection_msg.bbox.center.position.y = float(y)
            detection_msg.bbox.size_x = float(w)
            detection_msg.bbox.size_y = float(h)
            detection_msg.results = [
                ObjectHypothesisWithPose(
                    hypothesis = ObjectHypothesis(class_id = "Can", score = highest_prob), 
                    pose = PoseWithCovariance(pose = Pose(position = Point(x = float(x),y = float(y)))))]
            detection_msg.header.stamp = self.get_clock().now().to_msg()
            self.detection_publisher.publish(detection_msg)
        except CvBridgeError as e:
          print(e)
        cv2.waitKey(1)
    
    def shutdown(self):
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    yolo_detection_node = YOLODetectionNode()

    rclpy.spin(yolo_detection_node)

    yolo_detection_node.shutdown()
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    yolo_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
