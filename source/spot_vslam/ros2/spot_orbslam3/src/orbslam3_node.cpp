// Topic names (prefix, camera/*, orbslam/*) must match spot_vslam/slam/ros2_manager.py (TOPIC_PREFIX, ORB_*_TOPIC).
//
// ORB-SLAM3 wrapper for spot_vslam, built twice (CMakeLists.txt):
//   orbslam3_rgbd_node    RGB-D:  left image + depth
//   orbslam3_stereo_node  stereo: left + right image of a rectified pair (SPOT_ORBSLAM3_STEREO defined)
//
// Subscribes  /spot{i}/camera/image_raw       (rgb8, left / RGB-D camera)
//             /spot{i}/camera/depth           (32FC1, metres; RGB-D)
//             /spot{i}/camera/right/image_raw (rgb8; stereo, same stamp as the left image)
//             /spot{i}/camera/camera_info     (intrinsics, read once -> generates the ORB-SLAM3 settings file)
//             /spot{i}/orbslam/reset         (std_msgs/Empty)
// Publishes   /spot{i}/orbslam/robot_pose    (PoseStamped)    camera body pose in the map frame
//             /spot{i}/orbslam/tracking_state(Int32)          0 = init, 1 = tracking, 2 = lost
//             /spot{i}/orbslam/local_pc      (PointCloud2)    currently tracked map points, map frame
//
// The map frame is the body-aligned frame (x forward, y left, z up) of the camera at the first tracked frame
// (after a reset: the first tracked frame after it).

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

#ifdef SPOT_ORBSLAM3_STEREO
constexpr bool kStereo = true;
#else
constexpr bool kStereo = false;
#endif

class OrbSlam3Node : public rclcpp::Node {
public:
  OrbSlam3Node() : Node(kStereo ? "orbslam3_stereo" : "orbslam3_rgbd") {
    const char *root_env = std::getenv("ORB_SLAM3_ROOT");
    const std::string root = root_env ? root_env : "";
    prefix_ = declare_parameter<std::string>("prefix", "/spot0");
    vocab_ = declare_parameter<std::string>("vocabulary", root.empty() ? "" : root + "/Vocabulary/ORBvoc.txt");
    settings_out_ = declare_parameter<std::string>("settings_out", "/tmp/spot_orbslam3_settings.yaml");
    use_viewer_ = declare_parameter<bool>("use_viewer", false);
    max_depth_ = declare_parameter<double>("max_depth", 10.0);
    // stereo defaults from a sweep on Map A frames (docs/cuvslam.md): with 1000 features / FAST 20 the stereo map
    // needed 16 frames to start (< 500 points) and lost tracking mid-run more often
    n_features_ = declare_parameter<int>("n_features", kStereo ? 2000 : 1000);
    // stereo: the camera baseline (STEREO_BASELINE in locomotion_env_cfg.py); RGB-D: only sets the close / far
    // point threshold th_depth * baseline, as ORB-SLAM3's RGB-D examples do
    stereo_b_ = declare_parameter<double>("stereo_baseline", kStereo ? 0.12 : 0.08);
    th_depth_ = declare_parameter<double>("th_depth", kStereo ? 80.0 : 40.0);
    ini_th_fast_ = declare_parameter<int>("ini_th_fast", kStereo ? 12 : 20);
    min_th_fast_ = declare_parameter<int>("min_th_fast", kStereo ? 5 : 7);
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

  ~OrbSlam3Node() override {
    if (slam_) slam_->Shutdown();
  }

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;

  void OnInfo(CameraInfo::ConstSharedPtr m) {
    if (slam_) return;
    WriteSettings(*m);
    RCLCPP_INFO(get_logger(), "Loading vocabulary %s (takes a few seconds) ...", vocab_.c_str());
    slam_ = std::make_unique<ORB_SLAM3::System>(
        vocab_, settings_out_, kStereo ? ORB_SLAM3::System::STEREO : ORB_SLAM3::System::RGBD, use_viewer_);
    info_sub_.reset();

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable();
    rgb_sub_.subscribe(this, prefix_ + "/camera/image_raw", qos.get_rmw_qos_profile());
    second_sub_.subscribe(this, prefix_ + (kStereo ? "/camera/right/image_raw" : "/camera/depth"),
                          qos.get_rmw_qos_profile());
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), rgb_sub_, second_sub_);
    sync_->registerCallback(&OrbSlam3Node::OnFrame, this);
    RCLCPP_INFO(get_logger(), "ORB-SLAM3 %s ready (%ux%u, fx=%.1f fy=%.1f, baseline %.3f m)",
                kStereo ? "stereo" : "RGB-D", m->width, m->height, m->k[0], m->k[4], stereo_b_);
  }

  void WriteSettings(const CameraInfo &ci) {
    std::ofstream f(settings_out_);
    f << std::fixed << std::setprecision(6);  // ORB-SLAM3 rejects "320" where it expects a real number
    f << "%YAML:1.0\n"
      << "File.version: \"1.0\"\n"
      << "Camera.type: \"PinHole\"\n";
    for (const char *cam : kStereo ? std::vector<const char *>{"Camera1", "Camera2"} : std::vector<const char *>{"Camera1"})
      f << cam << ".fx: " << ci.k[0] << "\n" << cam << ".fy: " << ci.k[4] << "\n"
        << cam << ".cx: " << ci.k[2] << "\n" << cam << ".cy: " << ci.k[5] << "\n"
        << cam << ".k1: 0.0\n" << cam << ".k2: 0.0\n" << cam << ".p1: 0.0\n" << cam << ".p2: 0.0\n";
    if (kStereo) {
      // The sim pair is rectified already (same intrinsics, parallel, no distortion). Camera.type "Rectified" would
      // say so, but ORB-SLAM3 then segfaults printing the settings (its Camera2 calibration stays null), so give
      // both cameras and the left -> right transform: ORB-SLAM3 "rectifies" with the identity. The right camera
      // sits stereo_baseline along the left camera's x axis (optical frame: x right).
      f << "Stereo.T_c1_c2: !!opencv-matrix\n  rows: 4\n  cols: 4\n  dt: f\n  data: [1.0, 0.0, 0.0, " << stereo_b_
        << ", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\n";
    }
    f
      << "Camera.width: " << ci.width << "\nCamera.height: " << ci.height << "\n"
      << "Camera.fps: 30\nCamera.RGB: 1\n"
      << "Stereo.ThDepth: " << th_depth_ << "\nStereo.b: " << stereo_b_ << "\n"
      << "RGBD.DepthMapFactor: 1.0\n"
      << "ORBextractor.nFeatures: " << n_features_ << "\nORBextractor.scaleFactor: 1.2\n"
      << "ORBextractor.nLevels: 8\nORBextractor.iniThFAST: " << ini_th_fast_
      << "\nORBextractor.minThFAST: " << min_th_fast_ << "\n"
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

  // second = depth (RGB-D) or the right image (stereo)
  void OnFrame(Image::ConstSharedPtr rgb, Image::ConstSharedPtr second) {
    if (reset_requested_.exchange(false)) {
      slam_->Reset();
      have_origin_ = false;  // the next tracked frame starts the new map frame
      RCLCPP_WARN(get_logger(), "ORB-SLAM3 reset");
    }
    const char *second_encoding = kStereo ? "rgb8" : "32FC1";
    if (rgb->encoding != "rgb8" || second->encoding != second_encoding) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Unexpected encodings %s / %s", rgb->encoding.c_str(),
                            second->encoding.c_str());
      return;
    }
    cv::Mat im(rgb->height, rgb->width, CV_8UC3, const_cast<uint8_t *>(rgb->data.data()), rgb->step);
    const double t = rclcpp::Time(rgb->header.stamp).seconds();
    Sophus::SE3f Tcw;
    if (kStereo) {
      cv::Mat right(second->height, second->width, CV_8UC3, const_cast<uint8_t *>(second->data.data()),
                    second->step);
      Tcw = slam_->TrackStereo(im.clone(), right.clone(), t);
    } else {
      cv::Mat d(second->height, second->width, CV_32FC1, const_cast<uint8_t *>(second->data.data()), second->step);
      cv::Mat depth_clean = d.clone();
      for (int y = 0; y < depth_clean.rows; ++y) {  // inf / NaN / out-of-range -> 0 (= no measurement)
        float *row = depth_clean.ptr<float>(y);
        for (int x = 0; x < depth_clean.cols; ++x)
          if (!std::isfinite(row[x]) || row[x] > max_depth_) row[x] = 0.f;
      }
      Tcw = slam_->TrackRGBD(im.clone(), depth_clean, t);
    }

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
  int n_features_, ini_th_fast_, min_th_fast_;

  std::unique_ptr<ORB_SLAM3::System> slam_;
  std::atomic<bool> reset_requested_{false};
  Sophus::SE3f origin_;
  bool have_origin_ = false;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr info_sub_;
  message_filters::Subscriber<Image> rgb_sub_, second_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OrbSlam3Node>());
  rclcpp::shutdown();
  return 0;
}
