"""Launch one ORB-SLAM3 RGB-D node for /spot{robot_index}.

ORB_SLAM3_ROOT must be exported so the node finds Vocabulary/ORBvoc.txt.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_index", default_value="0"),
        DeclareLaunchArgument("use_viewer", default_value="false"),
        Node(
            package="spot_orbslam3",
            executable="orbslam3_rgbd_node",
            name=PythonExpression(["'orbslam3_rgbd_' + '", LaunchConfiguration("robot_index"), "'"]),
            output="screen",
            parameters=[{
                "prefix": PythonExpression(["'/spot' + '", LaunchConfiguration("robot_index"), "'"]),
                "use_viewer": LaunchConfiguration("use_viewer"),
                "settings_out": PythonExpression(
                    ["'/tmp/spot_orbslam3_settings_' + '", LaunchConfiguration("robot_index"), "' + '.yaml'"]),
            }],
        ),
    ])
