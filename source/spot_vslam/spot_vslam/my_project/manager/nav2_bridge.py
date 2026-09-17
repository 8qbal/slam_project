#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

class SlamToNav2Bridge(Node):
    def __init__(self):
        super().__init__('slam_to_nav2_bridge')
        # 訂閱 ORB-SLAM3 的 Pose
        self.create_subscription(PoseStamped, '/orbslam3/camera_pose', self.cb, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_br = TransformBroadcaster(self)
        self.get_logger().info("SLAM -> Nav2 Bridge Running...")

    def cb(self, msg):
        t = self.get_clock().now().to_msg()
        # 1. 發布 TF (odom -> base_link)
        tf = TransformStamped()
        tf.header.stamp = t
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = msg.pose.position.x
        tf.transform.translation.y = msg.pose.position.y
        tf.transform.translation.z = 0.0
        tf.transform.rotation = msg.pose.orientation
        self.tf_br.sendTransform(tf)
        
        # 2. 發布 Odom
        odom = Odometry()
        odom.header.stamp = t
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = msg.pose
        self.odom_pub.publish(odom)

def main():
    rclpy.init()
    node = SlamToNav2Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()