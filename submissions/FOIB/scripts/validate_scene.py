"""
Quick scene validation: load model, run 500 steps, print sensor values and body positions.
No viewer, headless. Should complete in < 5 seconds.
"""
import os, sys
import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco not found — activate conda env: conda activate robothon")

SCENE = os.path.join(os.path.dirname(__file__), "../models/scene.xml")

def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data  = mujoco.MjData(model)

    print(f"Model loaded: {model.nbody} bodies, {model.njnt} joints, "
          f"{model.ngeom} geoms, {model.nsensor} sensors")

    # Print joint names
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(model.njnt)]
    print(f"Joints: {joints}")

    # Print body names and positions at t=0
    mujoco.mj_forward(model, data)
    for name in ["egg", "bowl", "ee", "gripper_base"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            pos = data.xpos[bid]
            print(f"  body '{name}' pos: {pos}")

    # Run 500 steps with zero control, check for NaN / instability
    mujoco.mj_resetData(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)

    egg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "egg")
    egg_pos = data.xpos[egg_id]
    print(f"\nAfter 500 steps (1.0 s):")
    print(f"  Egg position: {egg_pos}")
    print(f"  Sensor readings: {data.sensordata}")

    if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
        print("FAIL: NaN detected in simulation state")
        sys.exit(1)

    if egg_pos[2] < 0.5:
        print("WARN: egg fell off table")
    else:
        print("OK: egg stable on table")

    print("\nScene validation PASSED")

if __name__ == "__main__":
    main()
