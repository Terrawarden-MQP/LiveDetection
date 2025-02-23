from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.actions import ExecuteProcess
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('show_cv', default_value="false", description='Display Raw OpenCV Output'),
        # PREVENT CAMERA MESSAGES FROM BEING SENT OVER NETWORK
        ExecuteProcess(
            cmd=['bash', '-c', 'export ROS_LOCALHOST_ONLY=1'], # Command to execute
            shell=True, # Run in a shell
            name='shell_command', # Name for the process
            output='screen' # Display output in the terminal
        ),
        IncludeLaunchDescription(
            # package="realsense2_camera",
            # launch="rs_launch.py",
            FindPackageShare('realsense2_camera').find('realsense2_camera') + '/launch/rs_launch.py',
            launch_arguments=
                {
                    "depth_module.depth_profile":"480,270,5",
                    "depth_module.exposure":"8000",
                    "enable_sync":"true",
                    "pointcloud.enable":"true",
                    "enable_color":"true",
                    "initial_reset":"true",
                    "rgb_camera.color_profile":"1280,720,15",
                    "align_depth.enable":"true"
                }.items() 
        ),
        Node(
            package="joisie_vision",
            namespace="joisie_vision",
            executable="color_detection",
            parameters=[
                {
                    "topic":"/camera/camera/color/image_raw",
                    "show": LaunchConfiguration('show_cv')
                }
            ]
        )
    ])