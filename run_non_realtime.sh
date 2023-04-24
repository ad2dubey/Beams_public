#!/bin/bash
#*************For the laptop*************
# source ./devel/setup.bash
# /opt/ros/noetic/bin/roslaunch ./src/retina4sn_viewer/share/srs_4sn.launch output_file:=/home/marga3/work/data/t.csv
# /opt/ros/noetic/bin/rosclean purge -y

#*************For the PC*************
source /home/marga_share/RETINA_4SN_RVIZ_DevMode/rviz/4sn_viewer/catkin_ws/devel/setup.bash
/opt/ros/noetic/bin/roslaunch /home/marga_share/RETINA_4SN_RVIZ_DevMode/rviz/4sn_viewer/catkin_ws/src/retina4sn_viewer/share/srs_4sn.launch output_file:=/home/marga_share/group4/CODEBASE/temp_trash.csv
# cd /home/marga_share/group4/Mar17
# python3 visualize_anim.py $1.csv
# /opt/ros/noetic/bin/rosclean purge -y

# ********** laptop adapted to PC ***************
# source /home/marga_share/RETINA_4SN_RVIZ_DevMode/rviz/4sn_viewer/catkin_ws/devel/setup.bash
# /opt/ros/noetic/bin/roslaunch /home/marga_share/RETINA_4SN_RVIZ_DevMode/rviz/4sn_viewer/catkin_ws/src/retina4sn_viewer/share/srs_4sn.launch output_file:=/home/marga_share/group4/CODEBASE/clusterdata
# /opt/ros/noetic/bin/rosclean purge -y