from pxr import Usd, UsdGeom, UsdLux, Gf
from spot_vslam.assets import SPOT_VSLAM_USD_DIR

# 開啟原始 USD 檔案
stage = Usd.Stage.Open(f"{SPOT_VSLAM_USD_DIR}/spot.usd")

# -----------------------
# 1️⃣ 新增 DiskLight
# -----------------------
light_path = "/spot/body/MyDiskLight"
disk_light = UsdLux.DiskLight.Define(stage, light_path)

# 設定位置
disk_light.AddTranslateOp().Set(Gf.Vec3d(-0.45, 0.0, 0.0))

# 設定旋轉 (degree -> radian 可用 Gf.Rotation)
disk_light.AddRotateXYZOp().Set(Gf.Vec3d(90.0, -90.0, 0.0))

# 設定強度與半徑
disk_light.CreateIntensityAttr().Set(3000.0)
disk_light.CreateRadiusAttr().Set(3.0)

# -----------------------
# 2️⃣ 新增 Camera
# -----------------------
camera_path = "/spot/body/MyCamera"
camera = UsdGeom.Camera.Define(stage, camera_path)

# 設定位置
camera.AddTranslateOp().Set(Gf.Vec3d(-0.5, 0.0, 0.0))

# 設定旋轉
camera.AddRotateXYZOp().Set(Gf.Vec3d(90.0, -90.0, 0.0))

# -----------------------
# 3️⃣ 匯出新的 USD
# -----------------------
export_path = f"{SPOT_VSLAM_USD_DIR}/spot_with_light_camera.usd"
stage.GetRootLayer().Export(export_path)

print(f"匯出完成：{export_path}")
