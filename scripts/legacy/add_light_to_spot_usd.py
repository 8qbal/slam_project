from pxr import Usd, UsdGeom, UsdLux, Gf
from spot_vslam.assets import SPOT_USD_PATH, SPOT_VSLAM_USD_DIR

# Open the original USD
stage = Usd.Stage.Open(SPOT_USD_PATH)

# -----------------------
# 1. Add a DiskLight
# -----------------------
light_path = "/spot/body/MyDiskLight"
disk_light = UsdLux.DiskLight.Define(stage, light_path)

# Position
disk_light.AddTranslateOp().Set(Gf.Vec3d(-0.45, 0.0, 0.0))

# Rotation (degree -> radian via Gf.Rotation)
disk_light.AddRotateXYZOp().Set(Gf.Vec3d(90.0, -90.0, 0.0))

# Intensity and radius
disk_light.CreateIntensityAttr().Set(3000.0)
disk_light.CreateRadiusAttr().Set(3.0)

# -----------------------
# 2. Add a camera
# -----------------------
camera_path = "/spot/body/MyCamera"
camera = UsdGeom.Camera.Define(stage, camera_path)

# Position
camera.AddTranslateOp().Set(Gf.Vec3d(-0.5, 0.0, 0.0))

# Rotation
camera.AddRotateXYZOp().Set(Gf.Vec3d(90.0, -90.0, 0.0))

# -----------------------
# 3. Export the new USD
# -----------------------
export_path = f"{SPOT_VSLAM_USD_DIR}/spot_with_light_camera.usd"
stage.GetRootLayer().Export(export_path)

print(f"Exported: {export_path}")
