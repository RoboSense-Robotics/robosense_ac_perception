from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_dir = get_package_share_directory('robosense_ac_perception')
    rviz_config_file_yolov8 = os.path.join(
        package_dir,
        'rviz2_config',
        'rviz2_config_yolov8.rviz'
    )
    config_file = os.path.join(package_dir, "config", "usr_config.yaml")

    yolov8_node = Node(
        package='robosense_ac_perception',
        executable='yolov8_node',
        name='yolov8_node',
        output='screen',
        arguments=['--config', config_file]
    )
    pv_post_process_node = Node(
        package='robosense_ac_perception',
        executable='pv_post_process_node',
        name='pv_post_process_node',
        output='screen',
        arguments=['--config', config_file]
    )
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file_yolov8],
        output='screen'
    )
    ld = LaunchDescription()
    ld.add_action(yolov8_node)
    ld.add_action(pv_post_process_node)
    ld.add_action(rviz2_node)
    return ld