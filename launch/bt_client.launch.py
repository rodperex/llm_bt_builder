from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    pkg_share_dir = get_package_share_directory('llm_bt_builder')
    
    # Default objective file path (relative to package)
    default_objective_file = os.path.join(pkg_share_dir, 'objectives', 'explain.txt')
    default_yaml_file = os.path.join(pkg_share_dir, 'config', 'social_bt_nodes.yaml')

    objective_file_arg = DeclareLaunchArgument(
        'objective_file',
        default_value=default_objective_file,
        description='Path to the objective text file'
    )
    capabilities_yaml_arg = DeclareLaunchArgument(
        'capabilities_yaml',
        default_value=default_yaml_file,
        description='Path to the robot capabilities YAML file'
    )

    bt_client_node = Node(
        package='llm_bt_builder',
        executable='bt_client_node.py',
        name='bt_client_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'objective_file': LaunchConfiguration('objective_file'),
            'capabilities_yaml': LaunchConfiguration('capabilities_yaml')
        }]
    )

    return LaunchDescription([
        objective_file_arg,
        capabilities_yaml_arg,
        bt_client_node
    ])
