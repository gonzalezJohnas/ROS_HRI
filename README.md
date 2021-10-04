# ROS_HRI
Python scripts with ROS integration to perform Human robot interaction:
- Head pose estimation
- Skeleton pose estimation
- 2D gestures recognition

Work in progress 

# Dependencies
Python3 with ROS (follow this to setup the environment https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674).

Install the requirements of the project:
  
	pip install -r requirements.txt
	
# Docker installation
Clone the repository

	git clone https://github.com/gonzalezJohnas/ROS_HRI.git

Build the docker image for the ROS_HRI

	docker build - < Dockerfile


Then you can run the Docker image with  --network host to used the host network and with -it to get a bash to run the build the application.

	docker run --network host -it  ros_hri_gpu /bin/bash
	
All the code and dependencies are installed there just need to run the ROS_HRI application

	root@1ec54b63ea34:/catkin_ws# source devel/setup.bash
	root@1ec54b63ea34:/catkin_ws# python3 src/ROS_HRI/src/main.py <name_camera_node> 
					<pose_estimation_model_path>
					<hand_classifier_model_path> 
					<hand_gestures_label_file> 
	
Example of the command

	 python3 src/ROS_HRI/src/main.py /camera/color/image_raw 
	 /home/icub/catkin_build_ws/src/ROS_HRI/src/humanpose/checkpoint/checkpoint_iter_370000.pth
	 /home/icub/Documents/Jonas/HandPose/cnn/models/hand_poses_wGarbage_10.h5 
	 /home/icub/catkin_build_ws/src/ROS_HRI/src/handGesture/poses.txt
	
	

	
