## Generic Setup Info:

Intel Realsense Viewer Install Guide (For laptops):
- https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

ROS 2 Install Guide:
- https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

ROS 2 Workspace Setup:
- https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html
	
Vision Stack setup repo setup (after ROS2 Workspace has been set up)
1. `cd ~/ros2_ws/src`
2. `git clone git@github.com:Terrawarden-MQP/TerrawardenVision.git`
3. `cd TerrawardenVision`
4. `pip install -r requirements.txt`

## Jetson Setup

1.  Flash with Jetpack 6.0
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
6.  Run i455 with ROS
   - `ros2 run rviz2 rviz2` and open topic `/camera/camera/depth/color/points`
   - In folder `realsense2_camera`, run `source install/setup.bash`
   - `ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=480,270,5 depth_module.exposure:=8000 enable_sync:=true pointcloud.enable:=true enable_color:=true initial_reset:=true rgb_camera.color_profile:=1280,720,15`
     (Note: higher resolutions and other configurations are technically supported, as shown by `rs-enumerate-devices`, but have experimentally been shown to drop frames or stop publishing.)
7.  Install ROS 2 Real Time Classification and Detection using PyTorch TensorRT
   - Follow the steps here: `https://github.com/NVIDIA-AI-IOT/ros2_torch_trt`
   - Note, the following modifications may be needed to fix installing the `torch2trt` dependency:
      - Ensure `$CUDA_HOME` is set properly in terminal
      - If `torch` does not show up as being installed in Python properly, modify `cpp_extension.py` (`~/.local/lib/python3.10/site-packages/torch/utils/cpp_extension.py`) so that `CUDA_HOME = _find_cuda_home()` (remove or comment out the rest of the line, `# if torch.cuda._is_compiled() else None`)
      - If there are permission errors accessing files when installing, run `sudo chown -R USERNAME /usr/local/lib/python3.10/dist-packages/`
8. Validate install by testing the live_detectors
      - MobileNET Pretrained model found at: https://drive.google.com/drive/folders/1pKn-RifvJGWiOx0ZCRLtCXM5GT5lAluu

## Package Setup
1. Install PyTorch w/ CUDA Enabled
- https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
2. Install ROS 2 Vision Messages
- `sudo apt install ros-humble-vision-msgs`

FAULTY DRIVERS
3.  Install RealSense Drivers
   - https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
   - https://github.com/IntelRealSense/realsense_mipi_platform_driver

4. Install Intel Wifi Driver (hotfix for Jetpack 6.0 kernel 5.15.136-tegra)
   - Download backports-5.15.148-1.tar.xz from https://cdn.kernel.org/pub/linux/kernel/projects/backports/stable/v5.15.148/
   - `tar Jxfv backports-5.15.148-1.tar.xz`
   - `cd backports-5.15.148-1.tar.xz`
   - `make defconfig-iwlwifi`
   - `make -j8`
   - `sudo make install`
   - reboot the system
   - add a line at the end of `sudo nano /etc/initramfs-tools/modules` to be the `iwlwifi`
   - `sudo update-initramfs -u` to update the interfaces
   - `sudo reboot` or power off, unplug, wait 5 seconds, plug back in

### Enable SSH on Jetson
1.  `sudo apt install ufw`
2.  `sudo systemctl enable ssh`
3.  `sudo systemctl start ssh`
4.  `sudo ufw allow ssh`
5.  Modify timeout to 30 minutes `gsettings set org.gnome.desktop.session idle-delay 1800`

# Running our code
1. `colcon build`
2. `source ~/Desktop/ros_ws/install/setup.bash`
3. `ros2 launch joisie_vision live_detect.launch.py`

## DATASET
We found this boi, good enough for now lmao
- https://github.com/PUTvision/UAVVaste
Wrote code for data augmentation as well - with 10 variants per image of augmentation, that's 7,720 images with 37,180 annotations, which should be more than good enough.
