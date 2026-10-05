// ORB-SLAM3 RGB-D wrapper for spot_vslam.
//
// Subscribes  /spot{i}/camera/image_raw      (rgb8)
//             /spot{i}/camera/depth          (32FC1, metres)
//             /spot{i}/camera/camera_info    (intrinsics, read once -> generates the ORB-SLAM3 settings file)
//             /spot{i}/orbslam/reset         (std_msgs/Empty)
// Publishes   /spot{i}/orbslam/robot_pose    (PoseStamped)    camera body pose in the map frame
//             /spot{i}/orbslam/tracking_state(Int32)          0 = init, 1 = tracking, 2 = lost
//             /spot{i}/orbslam/local_pc      (PointCloud2)    currently tracked map points, map frame
//
// The map frame is the body-aligned frame (x forward, y left, z up) of the camera at the first tracked frame.

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/int32.hpp>

#include "MapPoint.h"
#include "System.h"

using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;

class OrbSlam3RgbdNode : public rclcpp::Node {
public:
  OrbSlam3RgbdNode() : Node("orbslam3_rgbd") {
    const char *root_env = std::getenv("ORB_SLAM3_ROOT");
    const std::string root = root_env ? root_env : "";
    prefix_ = declare_parameter<std::string>("prefix", "/spot0");
    vocab_ = declare_parameter<std::string>("vocabulary", root.empty() ? "" : root + "/Vocabulary/ORBvoc.txt");
    settings_out_ = declare_parameter<std::string>("settings_out", "/tmp/spot_orbslam3_settings.yaml");
    use_viewer_ = declare_parameter<bool>("use_viewer", false);
    max_depth_ = declare_parameter<double>("max_depth", 10.0);
    n_features_ = declare_parameter<int>("n_features", 1000);
    stereo_b_ = declare_parameter<double>("stereo_baseline", 0.08);
    th_depth_ = declare_parameter<double>("th_depth", 40.0);
    if (vocab_.empty()) {
      RCLCPP_FATAL(get_logger(), "No vocabulary: set ORB_SLAM3_ROOT or the 'vocabulary' parameter");
      throw std::runtime_error("no vocabulary");
    }

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable();

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(prefix_ + "/orbslam/robot_pose", qos);
    state_pub_ = create_publisher<std_msgs::msg::Int32>(prefix_ + "/orbslam/tracking_state", qos);
    pc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(prefix_ + "/orbslam/local_pc", qos);
    reset_sub_ = create_subscription<std_msgs::msg::Empty>(
        prefix_ + "/orbslam/reset", rclcpp::QoS(1).reliable(),
        [this](std_msgs::msg::Empty::ConstSharedPtr) { reset_requested_ = true; });

    // Intrinsics arrive on camera_info; the SLAM system is created lazily on the first message.
    info_sub_ = create_subscription<CameraInfo>(
        prefix_ + "/camera/camera_info", qos, [this](CameraInfo::ConstSharedPtr m) { OnInfo(m); });

    RCLCPP_INFO(get_logger(), "Waiting for %s/camera/camera_info ...", prefix_.c_str());
  }

  ~OrbSlam3RgbdNode() override {
    if (slam_) slam_->Shutdown();
  }

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;

  void OnInfo(CameraInfo::ConstSharedPtr m) {
    if (slam_) return;
    WriteSettings(*m);
    RCLCPP_INFO(get_logger(), "Loading vocabulary %s (takes a few seconds) ...", vocab_.c_str());
    slam_ = std::make_unique<ORB_SLAM3::System>(vocab_, settings_out_, ORB_SLAM3::System::RGBD, use_viewer_);
    info_sub_.reset();

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable();
    rgb_sub_.subscribe(this, prefix_ + "/camera/image_raw", qos.get_rmw_qos_profile());
    depth_sub_.subscribe(this, prefix_ + "/camera/depth", qos.get_rmw_qos_profile());
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), rgb_sub_, depth_sub_);
    sync_->registerCallback(&OrbSlam3RgbdNode::OnFrame, this);
    RCLCPP_INFO(get_logger(), "ORB-SLAM3 RGB-D ready (%ux%u, fx=%.1f fy=%.1f)", m->width, m->height, m->k[0], m->k[4]);
  }

  void WriteSettings(const CameraInfo &ci) {
    std::ofstream f(settings_out_);
    f << std::fixed << std::setprecision(6);  // ORB-SLAM3 rejects "320" where it expects a real number
    f << "%YAML:1.0\n"
      << "File.version: \"1.0\"\n"
      << "Camera.type: \"PinHole\"\n"
      << "Camera1.fx: " << ci.k[0] << "\nCamera1.fy: " << ci.k[4] << "\n"
      << "Camera1.cx: " << ci.k[2] << "\nCamera1.cy: " << ci.k[5] << "\n"
      << "Camera1.k1: 0.0\nCamera1.k2: 0.0\nCamera1.p1: 0.0\nCamera1.p2: 0.0\n"
      << "Camera.width: " << ci.width << "\nCamera.height: " << ci.height << "\n"
      << "Camera.fps: 30\nCamera.RGB: 1\n"
      << "Stereo.ThDepth: " << th_depth_ << "\nStereo.b: " << stereo_b_ << "\n"
      << "RGBD.DepthMapFactor: 1.0\n"
      << "ORBextractor.nFeatures: " << n_features_ << "\nORBextractor.scaleFactor: 1.2\n"
      << "ORBextractor.nLevels: 8\nORBextractor.iniThFAST: 20\nORBextractor.minThFAST: 7\n"
      << "Viewer.KeyFrameSize: 0.05\nViewer.KeyFrameLineWidth: 1.0\nViewer.GraphLineWidth: 0.9\n"
      << "Viewer.PointSize: 2.0\nViewer.CameraSize: 0.08\nViewer.CameraLineWidth: 3.0\n"
      << "Viewer.ViewpointX: 0.0\nViewer.ViewpointY: -0.7\nViewer.ViewpointZ: -1.8\nViewer.ViewpointF: 500.0\n";
  }

  // Optical frame (x right, y down, z forward) -> body frame (x forward, y left, z up).
  static Eigen::Matrix3f OpticalToBody() {
    Eigen::Matrix3f C;
    C << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    return C;
  }

  void OnFrame(Image::ConstSharedPtr rgb, Image::ConstSharedPtr depth) {
    if (reset_requested_.exchange(false)) {
      slam_->Reset();
      RCLCPP_WARN(get_logger(), "ORB-SLAM3 reset");
    }
    if (rgb->encoding != "rgb8" || depth->encoding != "32FC1") {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Unexpected encodings rgb=%s depth=%s",
                            rgb->encoding.c_str(), depth->encoding.c_str());
      return;
    }
    cv::Mat im(rgb->height, rgb->width, CV_8UC3, const_cast<uint8_t *>(rgb->data.data()), rgb->step);
    cv::Mat d(depth->height, depth->width, CV_32FC1, const_cast<uint8_t *>(depth->data.data()), depth->step);
    cv::Mat depth_clean = d.clone();
    for (int y = 0; y < depth_clean.rows; ++y) {  // inf / NaN / out-of-range -> 0 (= no measurement)
      float *row = depth_clean.ptr<float>(y);
      for (int x = 0; x < depth_clean.cols; ++x)
        if (!std::isfinite(row[x]) || row[x] > max_depth_) row[x] = 0.f;
    }
    const double t = rclcpp::Time(rgb->header.stamp).seconds();
    Sophus::SE3f Tcw = slam_->TrackRGBD(im.clone(), depth_clean, t);

    const int orb_state = slam_->GetTrackingState();
    int state = 0;  // NO_IMAGES_YET / NOT_INITIALIZED
    if (orb_state == 2 || orb_state == 5) state = 1;      // OK / OK_KLT
    else if (orb_state == 3 || orb_state == 4) state = 2;  // RECENTLY_LOST / LOST
    std_msgs::msg::Int32 s;
    s.data = state;
    state_pub_->publish(s);
    if (state != 1) return;

    const Eigen::Matrix3f C = OpticalToBody();
    const Sophus::SE3f Twc = Tcw.inverse();
    if (!have_origin_) {  // the first tracked camera pose defines the map frame
      origin_ = Twc;
      have_origin_ = true;
    }
    const Sophus::SE3f rel = origin_.inverse() * Twc;
    Eigen::Vector3f p = C * rel.translation();
    Eigen::Quaternionf q(C * rel.rotationMatrix() * C.transpose());
    q.normalize();

    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = rgb->header.stamp;
    ps.header.frame_id = "orbslam_map";
    ps.pose.position.x = p.x();
    ps.pose.position.y = p.y();
    ps.pose.position.z = p.z();
    ps.pose.orientation.x = q.x();
    ps.pose.orientation.y = q.y();
    ps.pose.orientation.z = q.z();
    ps.pose.orientation.w = q.w();
    pose_pub_->publish(ps);

    PublishPointCloud(rgb->header.stamp, C);
  }

  void PublishPointCloud(const builtin_interfaces::msg::Time &stamp, const Eigen::Matrix3f &C) {
    std::vector<ORB_SLAM3::MapPoint *> mps = slam_->GetTrackedMapPoints();
    std::vector<float> pts;
    pts.reserve(mps.size() * 3);
    const Sophus::SE3f origin_inv = origin_.inverse();
    for (auto *mp : mps) {
      if (!mp || mp->isBad()) continue;
      const Eigen::Vector3f w = C * (origin_inv * mp->GetWorldPos());
      pts.insert(pts.end(), {w.x(), w.y(), w.z()});
    }
    sensor_msgs::msg::PointCloud2 pc;
    pc.header.stamp = stamp;
    pc.header.frame_id = "orbslam_map";
    pc.height = 1;
    pc.width = pts.size() / 3;
    pc.is_bigendian = false;
    pc.is_dense = true;
    pc.point_step = 12;
    pc.row_step = pc.point_step * pc.width;
    for (const char *n : {"x", "y", "z"}) {
      sensor_msgs::msg::PointField f;
      f.name = n;
      f.offset = 4 * pc.fields.size();
      f.datatype = sensor_msgs::msg::PointField::FLOAT32;
      f.count = 1;
      pc.fields.push_back(f);
    }
    pc.data.resize(pts.size() * sizeof(float));
    if (!pts.empty()) std::memcpy(pc.data.data(), pts.data(), pc.data.size());
    pc_pub_->publish(pc);
  }

  std::string prefix_, vocab_, settings_out_;
  bool use_viewer_;
  double max_depth_, stereo_b_, th_depth_;
  int n_features_;

  std::unique_ptr<ORB_SLAM3::System> slam_;
  std::atomic<bool> reset_requested_{false};
  Sophus::SE3f origin_;
  bool have_origin_ = false;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr info_sub_;
  message_filters::Subscriber<Image> rgb_sub_, depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OrbSlam3RgbdNode>());
  rclcpp::shutdown();
  return 0;
}
