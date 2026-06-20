"""
Interactive run with MuJoCo passive viewer.

Autonomous mode (default):
  Shows the phase controller running in real time.
  Press Ctrl+C to exit.

Teleop mode (--teleop):
  Direct keyboard control of all joints and gripper.
  Bindings printed on startup.

Camera mode (--camera NAME):
  Lock viewer to a named camera (e.g. gripper_cam).
"""
import os, sys, time, argparse, select, re, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit("mujoco not found — conda activate robothon")

from controller.phase_controller import PhaseController, Phase, _WP

SCENE = os.path.join(os.path.dirname(__file__), "../models/scene.xml")

_OBJECT_DEFS = {
    "egg":      {"type": "ellipsoid", "size": "0.025 0.022 0.032", "rgba": "1.0 0.98 0.85 1", "mass": "0.065"},
    "cylinder": {"type": "cylinder",  "size": "0.025 0.040 0",     "rgba": "0.7 0.9 1.0 1",   "mass": "0.080"},
    "sphere":   {"type": "sphere",    "size": "0.028 0 0",          "rgba": "0.9 0.30 0.30 1", "mass": "0.055"},
}


def _load_object_scene(scene_path, object_type="egg"):
    if object_type == "egg":
        return mujoco.MjModel.from_xml_path(scene_path)
    d = _OBJECT_DEFS[object_type]
    with open(scene_path, encoding="utf-8") as f:
        xml = f.read()
    def _patch(m):
        b = m.group(0)
        b = re.sub(r'\btype="[^"]*"', f'type="{d["type"]}"', b, count=1)
        b = re.sub(r'\bsize="[^"]*"', f'size="{d["size"]}"', b, count=1)
        b = re.sub(r'\bmass="[^"]*"', f'mass="{d["mass"]}"', b, count=1)
        b = re.sub(r'\brgba="[^"]*"', f'rgba="{d["rgba"]}"', b, count=1)
        return b
    xml = re.sub(r'<geom name="egg_geom"[\s\S]*?/>', _patch, xml)
    scene_dir = os.path.dirname(os.path.abspath(scene_path))
    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml", dir=scene_dir, delete=False, mode="w", encoding="utf-8")
    tmp.write(xml)
    tmp_path = tmp.name
    tmp.close()
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    return model


_JOINT_STEP = 0.05   # rad per keypress
_GRIP_STEP  = 0.003  # m   per keypress

_TELEOP_HELP = """
=== TELEOP MODE ===
  a / d    joint1  +/- {j:.2f} rad
  w / s    joint2  +/- {j:.2f} rad
  q / e    joint3  +/- {j:.2f} rad
  [ / ]    gripper  open / close {g:.0f} mm
  r        reset episode
  x        exit
""".format(j=_JOINT_STEP, g=_GRIP_STEP * 1000)


# ---------------------------------------------------------------------------
# Raw-keyboard helper (POSIX — macOS / Linux)
# ---------------------------------------------------------------------------

class _KeyReader:
    """Context manager: switches stdin to raw (no-echo, non-blocking char read)."""
    def __enter__(self):
        try:
            import tty, termios
            self._fd  = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
            self._termios = termios
        except Exception:
            self._fd = None     # Windows fallback: no raw mode
        return self

    def __exit__(self, *_):
        if self._fd is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old)

    def read(self):
        """Return one character if available, else None (non-blocking)."""
        if self._fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# ---------------------------------------------------------------------------
# Teleop loop
# ---------------------------------------------------------------------------

def _run_teleop(model, data, viewer, cam_id):
    act_j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_j1")
    act_j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_j2")
    act_j3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_j3")
    act_ga = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_grip_a")
    act_gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_grip_b")
    act_gc = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_grip_c")
    arm_qpos_adr = [
        model.jnt_qposadr[j] for j in range(model.njnt)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        in ("joint1", "joint2", "joint3")
    ]

    def _reset():
        mujoco.mj_resetData(model, data)
        for i, addr in enumerate(arm_qpos_adr):
            data.qpos[addr] = _WP["retract"][i]
        mujoco.mj_forward(model, data)
        return _WP["retract"].copy(), 0.0

    ctrl_q, grip = _reset()

    def _apply():
        lo = model.actuator_ctrlrange
        data.ctrl[act_j1] = float(np.clip(ctrl_q[0], lo[act_j1, 0], lo[act_j1, 1]))
        data.ctrl[act_j2] = float(np.clip(ctrl_q[1], lo[act_j2, 0], lo[act_j2, 1]))
        data.ctrl[act_j3] = float(np.clip(ctrl_q[2], lo[act_j3, 0], lo[act_j3, 1]))
        g = float(np.clip(grip, 0.0, 0.035))
        data.ctrl[act_ga] = g
        data.ctrl[act_gb] = g
        data.ctrl[act_gc] = g

    print(_TELEOP_HELP)
    if cam_id >= 0:
        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

    step = 0
    with _KeyReader() as keys:
        while viewer.is_running():
            ch = keys.read()
            if ch in ('x', '\x03'):     # x or Ctrl+C
                break
            elif ch == 'a': ctrl_q[0] += _JOINT_STEP
            elif ch == 'd': ctrl_q[0] -= _JOINT_STEP
            elif ch == 'w': ctrl_q[1] += _JOINT_STEP
            elif ch == 's': ctrl_q[1] -= _JOINT_STEP
            elif ch == 'q': ctrl_q[2] += _JOINT_STEP
            elif ch == 'e': ctrl_q[2] -= _JOINT_STEP
            elif ch == '[': grip = max(0.0,   grip - _GRIP_STEP)
            elif ch == ']': grip = min(0.035, grip + _GRIP_STEP)
            elif ch == 'r': ctrl_q, grip = _reset()

            _apply()
            mujoco.mj_step(model, data)
            viewer.sync()

            if step % 200 == 0:
                j1 = float(data.qpos[arm_qpos_adr[0]])
                j2 = float(data.qpos[arm_qpos_adr[1]])
                j3 = float(data.qpos[arm_qpos_adr[2]])
                egg_z = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "egg")][2]
                print(f"\r[{step:6d}]  j1={j1:+.2f} j2={j2:+.2f} j3={j3:+.2f}  "
                      f"grip={grip*1000:.1f}mm  egg_z={egg_z:.3f}m", end="", flush=True)
            step += 1
            time.sleep(model.opt.timestep)

    print()   # newline after inline status


# ---------------------------------------------------------------------------
# Autonomous loop
# ---------------------------------------------------------------------------

def _run_autonomous(model, data, viewer, cam_id, loop):
    ctrl = PhaseController(model, data)

    if cam_id >= 0:
        viewer.cam.type       = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id
    else:
        viewer.cam.azimuth    = -45
        viewer.cam.elevation  = -20
        viewer.cam.distance   = 1.2
        viewer.cam.lookat[:]  = [0.1, 0, 0.85]

    ep = 0
    while viewer.is_running():
        ep += 1
        ctrl.reset()
        step = 0
        print(f"\n--- Episode {ep} ---")

        while viewer.is_running():
            phase, fail = ctrl.step()
            mujoco.mj_step(model, data)
            viewer.sync()

            if step % 100 == 0:
                info = ctrl.overlay_info()
                touch = info.get("finger_touch", (0, 0, 0))
                print(f"[{info['step']:5d}] {info['phase']:12s}  "
                      f"grip={info['grip_force']:5.2f}N  "
                      f"egg_z={info['egg_z']:.3f}  "
                      f"touch=({touch[0]:.1f},{touch[1]:.1f},{touch[2]:.1f})"
                      + (f"  FAIL:{info['fail_reason']}" if info['fail_reason'] else ""))

            if phase in (Phase.DONE, Phase.FAIL):
                info  = ctrl.overlay_info()
                touch = info.get("finger_touch", (0, 0, 0))
                rslt  = "SUCCESS" if phase == Phase.DONE else f"FAIL ({info['fail_reason']})"
                print(f"\n=== {rslt} at step {info['step']} | "
                      f"touch=({touch[0]:.2f},{touch[1]:.2f},{touch[2]:.2f}) | "
                      f"gqual={info['grasp_quality']:.3f} ===")
                time.sleep(3)
                break

            step += 1
            time.sleep(model.opt.timestep)

        if not loop:
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FOIB-Egg interactive viewer")
    parser.add_argument("--teleop",  action="store_true",
                        help="Enable keyboard teleoperation instead of autonomous control")
    parser.add_argument("--camera",  default=None,
                        help="Lock viewer to named camera (e.g. gripper_cam, side_cam, top_cam)")
    parser.add_argument("--loop",    action="store_true",
                        help="Auto-restart episodes after completion (autonomous mode only)")
    parser.add_argument("--two-egg", action="store_true",
                        help="Load two-egg scene variant")
    parser.add_argument("--object", default="egg", choices=["egg", "cylinder", "sphere"],
                        help="target object geometry: egg (default) | cylinder | sphere")
    args = parser.parse_args()

    obj_type = "egg" if args.two_egg else args.object
    model = _load_object_scene(SCENE, obj_type)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    cam_id = -1
    if args.camera:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        if cam_id < 0:
            cams = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    for i in range(model.ncam)]
            print(f"Warning: camera '{args.camera}' not found. Available: {cams}")

    if args.teleop:
        print("Starting viewer in TELEOP mode... close viewer window or press x to exit.")
    else:
        print("Starting viewer in AUTONOMOUS mode... close viewer window to exit.")
        if cam_id < 0:
            print("  Tip: try --camera gripper_cam for a fingertip-eye view during grasp")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if args.teleop:
            _run_teleop(model, data, viewer, cam_id)
        else:
            _run_autonomous(model, data, viewer, cam_id, args.loop)


if __name__ == "__main__":
    main()
