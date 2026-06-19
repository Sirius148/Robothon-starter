"""
Interactive run with MuJoCo passive viewer.
Shows the controller running in real time.
Press Ctrl+C to exit.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit("mujoco not found — conda activate robothon")

from controller.phase_controller import PhaseController, Phase

SCENE = os.path.join(os.path.dirname(__file__), "../models/scene.xml")


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    ctrl = PhaseController(model, data)
    ctrl.reset()

    print("Starting interactive viewer... close viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = -45
        viewer.cam.elevation = -20
        viewer.cam.distance  = 1.2
        viewer.cam.lookat[:] = [0.1, 0, 0.85]

        step = 0
        while viewer.is_running():
            phase, fail = ctrl.step()
            mujoco.mj_step(model, data)
            viewer.sync()

            if step % 100 == 0:
                info = ctrl.overlay_info()
                print(f"[{info['step']:5d}] phase={info['phase']:12s} "
                      f"grip_force={info['grip_force']:5.2f}N  "
                      f"egg_z={info['egg_z']:.3f}m  "
                      f"dist_bowl={info['dist_to_bowl']:.3f}m"
                      + (f"  FAIL:{info['fail_reason']}" if info['fail_reason'] else ""))

            if phase in (Phase.DONE, Phase.FAIL):
                info = ctrl.overlay_info()
                result = "SUCCESS" if phase == Phase.DONE else f"FAIL ({info['fail_reason']})"
                print(f"\n=== Episode ended: {result} at step {info['step']} ===\n")
                time.sleep(3)
                break

            step += 1
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
