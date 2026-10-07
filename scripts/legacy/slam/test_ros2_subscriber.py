# test_ros2_manager.py
import time
from spot_vslam.managers.ros2_manager import Ros2Manager  # the Ros2Manager from before

def main():
    # Create the manager (assuming a single Spot)
    ros_mgr = Ros2Manager(num_spots=2, device="cpu")

    print("[INFO] Receiving ROS 2 topics (/spot1/orbslam/local_pc, /spot1/orbslam/tracking_state) ...")

    try:
        while True:
            # spin_once to run the subscribers
            ros_mgr.update(timeout_sec=0.1)

            # Convert the received data to tensors
            obs = ros_mgr.compute_obs(max_pts=128)

            print("local_pc tensor shape:", obs["local_pc"].shape)
            print("tracking_state tensor:", obs["tracking_state"].cpu().numpy())
            print("------")

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted, exiting.")

if __name__ == "__main__":
    main()
