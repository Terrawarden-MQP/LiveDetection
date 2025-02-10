# ROS2 imports 
import rclpy
from rclpy.node import Node

# CV Bridge and message imports
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesisWithPose, BoundingBox2D, Detection2D, Detection2DArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from joisie_vision.misc import Timer

import cv2
import numpy as np
import os

class CV2DetectionNode(Node):

    def __init__(self):
        super().__init__('color_detection_node')

        # Create a subscriber to the Image topic
        self.declare_parameter('topic', 'image')
        topic_name = self.get_parameter('topic')
        self.subscription = self.create_subscription(Image, topic_name.value, self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        # Create a Detection 2D array topic to publish results on
        # self.detection_publisher = self.create_publisher(Detection2D, 'color_detection', 10)
        self.point_publisher = self.create_publisher(Point, "detected_object", 10)

        # Create an Image publisher for the results
        self.result_publisher = self.create_publisher(Image,'color_detection_image',10)
            
        self.declare_parameter("color_h", 158)
        h = int(self.get_parameter('color_h').value)
        self.declare_parameter("color_s", 170)
        s = int(self.get_parameter('color_s').value)
        self.declare_parameter("color_v", 183)
        v = int(self.get_parameter('color_v').value)
        self.target_color = np.array([h,s,v], dtype=np.uint8)

        self.declare_parameter('show', False)
        self.show = self.get_parameter('show').value


    def listener_callback(self, data):
        self.get_logger().info("Received an image! ")
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)
        
        # Convert the imageFrame in 
        # BGR(RGB color space) to 
        # HSV(hue-saturation-value) 
        # color space 
        hsvFrame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV) 

        # Set range for red color and 
        # define mask (in HSV)
        color_tolerance = np.array([20, 75, 50], np.uint8)
        color_lower = np.where(self.target_color - color_tolerance >= 0, self.target_color - color_tolerance, 0)
        color_upper = np.where(self.target_color + color_tolerance <= 255, self.target_color + color_tolerance, 255) 
        color_mask = cv2.inRange(hsvFrame, color_lower, color_upper) 
        
        # Morphological Transform, Dilation 
        # for each color and bitwise_and operator 
        # between imageFrame and mask determines 
        # to detect only that particular color 
        kernel = np.ones((5, 5), "uint8") 
        
        # For red color 
        color_mask = cv2.dilate(color_mask, kernel) 
        res_color = cv2.bitwise_and(hsvFrame, hsvFrame, 
                                mask = color_mask) 
        

        # Creating contour to track red color 
        contours, hierarchy = cv2.findContours(color_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        
        detection_array = Detection2DArray()
        largest_detect = (0, None, (0,0))
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 

                # Definition of 2D array message and ading all object stored in it.
                object_hypothesis_with_pose = ObjectHypothesisWithPose()
                object_hypothesis_with_pose.hypothesis.class_id = "Color"
                object_hypothesis_with_pose.hypothesis.score = 1.

                bounding_box = BoundingBox2D()
                bounding_box.center.position.x = float((x + w)/2)
                bounding_box.center.position.y = float((y + h)/2)
                bounding_box.center.theta = 0.0
                
                bounding_box.size_x = float(2*(bounding_box.center.position.x - x))
                bounding_box.size_y = float(2*(bounding_box.center.position.y - y))

                detection = Detection2D()
                detection.header = data.header
                detection.results.append(object_hypothesis_with_pose)
                detection.bbox = bounding_box

                if w*h > largest_detect[0]:
                    largest_detect = (w*h, detection, (float(x + w/2), float(y + h/2)))

                detection_array.header = data.header
                detection_array.detections.append(detection)

                if self.show:
                    cv_image = cv2.rectangle(cv_image, (x, y), 
                                            (x + w, y + h), 
                                            (0, 0, 255), 2) 
                    
                    cv2.putText(cv_image, "Tuned Colour", (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                                (0, 0, 255))
        
        # Displaying the predictions
        if self.show:
            # cv_image = cv2.circle(cv_image, largest_detect[2], 
            #                                 2, (0, 255, 0), 2) 
            cv2.imshow("Real-Time Color Detection", cv_image) 
        # Publishing the results onto the the Detection2D vision_msgs format
        # self.detection_publisher.publish(largest_detect[1])
        point = Point()
        point.x = float(largest_detect[2][0])
        point.y = float(largest_detect[2][1])
        print(f"Publishing largest object: {(largest_detect[2])}")
        self.point_publisher.publish(point)

        ros_image = self.bridge.cv2_to_imgmsg(cv_image)
        ros_image.header.frame_id = 'camera_frame'
        self.result_publisher.publish(ros_image)
        cv2.waitKey(1)
