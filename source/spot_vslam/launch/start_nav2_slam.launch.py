"""Nav2 bringup (navigation_launch.py with Nav2's default params), RViz2 and a static map -> odom TF.

No SLAM node is started here: run the ORB-SLAM3 node separately (see scripts/README.md).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 1. Nav2's default launch directory
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    # 2. Arguments
    use_sim_time = LaunchConfiguration("use_sim_time", default="True")

    # 3. Nav2 Launch (Bringup)
    # Starts the planner, controller, recoveries, etc.
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": os.path.join(nav2_bringup_dir, "params", "nav2_params.yaml"),  # Nav2's default params
        }.items(),
    )

    # 4. (Optional) RViz2 for viewing
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        # For a custom RViz config, add arguments=['-d', path_to_rviz_file] here
    )

    # 5. (Important) A SLAM node belongs here.
    # With the custom ORB-SLAM3 wrapper, check the SLAM node's name.
    # Without a packaged ROS 2 SLAM node, keep the block below commented out and run Nav2 only
    # slam_node = Node(
    #     package='orb_slam3_ros2_wrapper', # <--- set to the actual package name
    #     executable='orb_slam3_ros2_wrapper_mono', # <--- set to the actual executable
    #     name='orb_slam3',
    #     output='screen',
    #     parameters=[{'use_sim_time': use_sim_time}]
    # )

    # 6. Static TF (to test without SLAM publishing map -> odom)
    # Format: x y z yaw pitch roll parent child
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time", default_value="True", description="Use simulation (Gazebo/Isaac) clock if true"
            ),
            nav2_launch,
            rviz_node,
            static_tf,  # remove this line if your SLAM publishes map->odom, or the two conflict
            # slam_node
        ]
    )
