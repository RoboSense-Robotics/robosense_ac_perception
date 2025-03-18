from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_dir = get_package_share_directory('robosense_ac_perception')
    rviz_config_file_ppseg = os.path.join(
        package_dir,
        'rviz2_config',
        'rviz2_config_ppseg.rviz'
    )
    config_file = os.path.join(package_dir, "config", "usr_config.yaml")

    ppseg_node = Node(
        package='robosense_ac_perception',
        executable='ppseg_node',
        name='ppseg_node',
        output='screen',
        arguments=['--config', config_file]
    )
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file_ppseg],
        output='screen'
    )
    ld = LaunchDescription()
    ld.add_action(ppseg_node)
    ld.add_action(rviz2_node)
    return ld