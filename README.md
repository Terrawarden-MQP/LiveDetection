## Generic Setup Info:

Intel Realsense Viewer Install Guide (For laptops):
- https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

ROS 2 Install Guide:
- https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

ROS 2 Workspace Setup:
- https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html

Virtual Environment guide (Useful for python version management):
1. `sudo apt install python3.10-venv`
2. `python -m venv ~/venv`
3. `source ~/venv/bin/activate`
	(Needs to be done every time)
	
Vision Stack setup repo setup (after ROS2 Workspace has been set up)
1. `cd ~/ros2_ws/src`
2. `git clone git@github.com:Terrawarden-MQP/TerrawardenVision.git`
3. `cd TerrawardenVision`
4. `pip install -r requirements.txt`

## Jetson Setup

1.  Flash with Jetson 6.0
   - https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/
2.  Check install by running `sudo apt-cache show nvidia-jetpack`
3.  Install Intel RealSense SDK
   - https://github.com/IntelRealSense/librealsense/blob/master/doc/libuvc_installation.md
4.  Install `colcon` (if not already installed)
   - https://colcon.readthedocs.io/en/released/user/installation.html (follow ROS 2 steps)
5.  Install ROS RealSense Wrapper
   - `sudo apt install ros-humble-realsense2-*`
   - `sudo apt remove ros-humble-librealsense2`
   - `mkdir -p ~/realsense2_camera/src && cd ~/realsense2_camera/src`
   - `git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master`
   - `rm -rf realsense2_camera_msgs realsense2_description`
   - `cd ~/realsense2_camera`
   - `colcon build --symlink-install`

OMIT
3.  Install RealSense Drivers
   - https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
   - https://github.com/IntelRealSense/realsense_mipi_platform_driver

### Enable SSH on Jetson
1.  `sudo apt install ufw`
2.  `sudo systemctl enable ssh`
3.  `sudo systemctl start ssh`
4.  `sudo ufw allow ssh`
5.  Modify timeout to 30 minutes `gsettings set org.gnome.desktop.session idle-delay 1800`
