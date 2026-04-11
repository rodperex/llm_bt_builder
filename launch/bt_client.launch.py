# Copyright 2026 Rodrigo Pérez-Rodríguez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    default_objective_file = os.path.join(pkg_share_dir, 'objectives', 'take_order.yaml')
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
