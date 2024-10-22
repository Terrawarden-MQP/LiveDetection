import torch
import torchvision
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image

class ImageProcessingNode(Node):

    def __init__(self):
        super().__init__('ImageProcessingNode')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            1)
        
        # Step 1: Initialize ResNet50 with ImageNet dataset trained weights
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.eval()
        # Step 2: Initialize the inference transforms
        self.preprocess = self.weights.transforms()

        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):

        tensor_img = torch.Tensor(msg.data, size=(3, msg.height, msg.width))

        # Step 3: Apply inference preprocessing transforms
        batch = self.preprocess(tensor_img).unsqueeze(0)

        # Step 4: Run model on batch
        out = self.model(batch)

        print(out.shape)
        with open('imagenet_classes.txt') as f:
            labels = [line.strip() for line in f.readlines()]
            _, index = torch.max(out, 1)

            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

            print(labels[index[0]], percentage[index[0]].item())


        


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = ImageProcessingNode()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
