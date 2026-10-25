"""Launch one ORB-SLAM3 stereo node for /spot{robot_index} (left + right image, Ros2ManagerCfg.right_camera_name).

ORB_SLAM3_ROOT must be exported so the node finds Vocabulary/ORBvoc.txt.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_index", default_value="0"),
            DeclareLaunchArgument("use_viewer", default_value="false"),
            Node(
                package="spot_orbslam3",
                executable="orbslam3_stereo_node",
                name=PythonExpression(["'orbslam3_stereo_' + '", LaunchConfiguration("robot_index"), "'"]),
                output="screen",
                parameters=[
                    {
                        # must match spot_vslam.slam.ros2_manager.TOPIC_PREFIX
                        "prefix": PythonExpression(["'/spot' + '", LaunchConfiguration("robot_index"), "'"]),
                        "use_viewer": LaunchConfiguration("use_viewer"),
                        "settings_out": PythonExpression(
                            [
                                "'/tmp/spot_orbslam3_stereo_settings_' + '",
                                LaunchConfiguration("robot_index"),
                                "' + '.yaml'",
                            ]
                        ),
                    }
                ],
            ),
        ]
    )
