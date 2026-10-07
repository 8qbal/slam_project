# test_ros2_manager.py
import time
from spot_vslam.managers.ros2_manager import Ros2Manager  # 匯入剛剛的 Ros2Manager

def main():
    # 建立 manager（假設只有 1 台 Spot）
    ros_mgr = Ros2Manager(num_spots=2, device="cpu")

    print("[INFO] 開始接收 ROS2 topic (/spot1/orbslam/local_pc, /spot1/orbslam/tracking_state) ...")

    try:
        while True:
            # 執行 spin_once 處理 subscriber
            ros_mgr.update(timeout_sec=0.1)

            # 把訂閱到的資料轉成 tensor
            obs = ros_mgr.compute_obs(max_pts=128)

            print("local_pc tensor shape:", obs["local_pc"].shape)
            print("tracking_state tensor:", obs["tracking_state"].cpu().numpy())
            print("------")

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("中斷測試，結束程式。")

if __name__ == "__main__":
    main()
