# launch/start_nav2_slam.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 取得 Nav2 的預設 launch 目錄
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    # 2.宣告參數
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    
    # 3. Nav2 Launch (Bringup)
    # 這會啟動規劃器 (Planner), 控制器 (Controller), 恢復行為 (Recoveries) 等
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': os.path.join(nav2_bringup_dir, 'params', 'nav2_params.yaml') # 使用預設參數
        }.items(),
    )

    # 4. (選擇性) 啟動 RViz2 以便觀察
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        # 如果您有自定義的 rviz 設定檔，可以在這裡加入 arguments=['-d', path_to_rviz_file]
    )

    # 5. (關鍵) 這裡應該要有一個 SLAM Node
    # 因為您使用的是自定義的 ORB-SLAM3，您需要確認您的 SLAM Node 名稱
    # 如果您還沒有包裝好的 ROS2 Node，暫時先註解掉下面這段，只跑 Nav2
    # slam_node = Node(
    #     package='orb_slam3_ros2_wrapper', # <--- 請改成您實際的 package name
    #     executable='orb_slam3_ros2_wrapper_mono', # <--- 請改成實際 executable
    #     name='orb_slam3',
    #     output='screen',
    #     parameters=[{'use_sim_time': use_sim_time}]
    # )

    # 6. 靜態 TF 發布 (如果 SLAM 沒有發布 map -> odom，可以用這個測試)
    # 格式: x y z yaw pitch roll parent child
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = ['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation (Gazebo/Isaac) clock if true'),
        
        nav2_launch,
        rviz_node,
        static_tf, # 如果您的 SLAM 會發布 map->odom，請把這一行拿掉，不然會衝突
        # slam_node 
    ])