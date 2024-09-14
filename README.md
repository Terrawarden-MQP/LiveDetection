Intel Realsense Viewer Install Guide:
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

