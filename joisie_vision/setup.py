from setuptools import setup

package_name = 'joisie_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name + '/launch', ['launch/live_detect.launch.py']),
        ('share/' + package_name + '/launch', ['launch/color_detect.launch.py']),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Terrawarden Drone Cleanup MQP',
    maintainer_email='gr-dronecleanup@wpi.edu',
    description='ROS2 package for object detection using TensorRT',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
	        'live_detection = joisie_vision.trt_detection:main',
	        'color_detection = joisie_vision.color_detection:main',
        ],
    },
)
