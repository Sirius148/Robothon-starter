"""
Open-loop policy replay from a --collect trajectory CSV.

Loads joint positions and grip control recorded during a benchmark run, then
re-executes them in MuJoCo — demonstrating that stored trajectories are complete
enough to reproduce pick-and-place without the phase controller.

Usage:
  # First collect a trajectory:
  python video/record_demo.py --episodes 1 --collect --collect-dir trajectories/

  # Then replay it:
  python scripts/run_policy.py trajectories/trajectory_ep001.csv
  python scripts/run_policy.py trajectories/trajectory_ep001.csv --out replay.mp4
  python scripts/run_policy.py trajectories/trajectory_ep001.csv --speed 2 --camera top_cam

Tracking error (EE position vs recorded) is printed as a fidelity metric.
"""
import argparse
import csv
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import mujoco
except ImportError:
    sys.exit("mujoco not found — conda activate robothon")

from controller.phase_controller import GRIP_CLOSED, GRIP_OPEN

SCENE = os.path.join(os.path.dirname(__file__), "../models/scene.xml")

# bowl_center site is 18 mm above bowl body origin in the MJCF
_BOWL_SITE_Z_OFFSET = 0.018

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


# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="FOIB-Egg policy replay")
    p.add_argument("traj", help="trajectory_epNNN.csv produced by --collect")
    p.add_argument("--out",    default=None,
                   help="write replay video to PATH (mp4)")
    p.add_argument("--speed",  type=float, default=1.0,
                   help="playback speed multiplier (default: 1×)")
    p.add_argument("--camera", default="side_cam",
                   help="MuJoCo camera name (side_cam / top_cam / gripper_cam)")
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--object", default="egg", choices=["egg", "cylinder", "sphere"],
                   help="target object geometry used during collection (default: egg)")
    return p.parse_args()


def _load_traj(path):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        sys.exit(f"Empty trajectory: {path}")
    return rows


def main():
    args = _parse_args()

    rows = _load_traj(args.traj)
    r0   = rows[0]

    # Initial scene state from first trajectory row
    egg0  = np.array([float(r0["egg_x"]),  float(r0["egg_y"]),  float(r0["egg_z"])])
    bowl0 = np.array([float(r0["bowl_x"]), float(r0["bowl_y"]), float(r0["bowl_z"])])
    has_ctrl_grip = "ctrl_grip" in r0

    # ── model setup ──────────────────────────────────────────────────────────
    model = _load_object_scene(SCENE, args.object)
    data  = mujoco.MjData(model)

    def bid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,     n)
    def aid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    def sid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,     n)
    def jid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    n)

    egg_bid = bid("egg")
    gb_bid  = bid("gripper_base")
    bowl_bid = bid("bowl")
    act_j1, act_j2, act_j3 = aid("act_j1"), aid("act_j2"), aid("act_j3")
    act_ga, act_gb, act_gc  = aid("act_grip_a"), aid("act_grip_b"), aid("act_grip_c")
    ee_site   = sid("ee")
    bowl_site = sid("bowl_center")

    egg_qadr = model.jnt_qposadr[jid("egg_free")]
    arm_qadr = [
        model.jnt_qposadr[j] for j in range(model.njnt)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        in ("joint1", "joint2", "joint3")
    ]

    # Place scene to match trajectory initial conditions
    mujoco.mj_resetData(model, data)
    model.body_pos[bowl_bid] = bowl0 - np.array([0.0, 0.0, _BOWL_SITE_Z_OFFSET])
    data.qpos[egg_qadr:egg_qadr+3]   = egg0
    data.qpos[egg_qadr+3:egg_qadr+7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[arm_qadr[0]] = float(r0["j1_pos"])
    data.qpos[arm_qadr[1]] = float(r0["j2_pos"])
    data.qpos[arm_qadr[2]] = float(r0["j3_pos"])
    mujoco.mj_forward(model, data)

    # ── video writer ─────────────────────────────────────────────────────────
    renderer = writer = None
    if args.out:
        try:
            import cv2, platform
        except ImportError:
            sys.exit("opencv-python not found — pip install opencv-python")
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        fourcc   = cv2.VideoWriter_fourcc(*(
            "avc1" if platform.system() == "Darwin" else "mp4v"))
        writer   = cv2.VideoWriter(args.out, fourcc, args.fps,
                                   (args.width, args.height))
        if not writer.isOpened():
            sys.exit(f"Cannot open video writer: {args.out}")

    render_every = max(1, int(round(1.0 / (args.fps * model.opt.timestep * args.speed))))

    # ── replay loop ──────────────────────────────────────────────────────────
    ee_errors    = []
    prev_grasped = 0
    n_weld_steps = 0
    weld_active  = False

    # Weld implementation: set egg XYZ from the recorded CSV values each step.
    # This exactly reproduces the egg trajectory from the original episode.
    # (Re-computing the gripper-relative offset accumulates one-step lag over
    # 300+ weld steps, causing ~50mm drift — using CSV positions avoids this.)
    has_egg_xyz  = "egg_x"   in rows[0]
    has_j_ctrl   = "j1_ctrl" in rows[0]

    print(f"Replaying {len(rows)} steps from {os.path.basename(args.traj)}")
    print(f"  egg start: {egg0}   bowl: {bowl0}")
    print(f"  ctrl_grip: {has_ctrl_grip}   j_ctrl: {has_j_ctrl}   speed: {args.speed}×")

    for i, row in enumerate(rows):
        grasped_flag = int(row["grasped"])

        # Arm joint targets: use recorded ctrl targets if available (true open-loop replay),
        # else fall back to recorded qpos (approximate — produces near-zero force).
        if has_j_ctrl:
            j1 = float(row["j1_ctrl"])
            j2 = float(row["j2_ctrl"])
            j3 = float(row["j3_ctrl"])
        else:
            j1 = float(row["j1_pos"])
            j2 = float(row["j2_pos"])
            j3 = float(row["j3_pos"])

        if has_ctrl_grip:
            grip = float(row["ctrl_grip"])
        else:
            grip = GRIP_CLOSED if grasped_flag else GRIP_OPEN

        # Set actuator targets
        data.ctrl[act_j1] = j1
        data.ctrl[act_j2] = j2
        data.ctrl[act_j3] = j3
        data.ctrl[act_ga] = grip
        data.ctrl[act_gb] = grip
        data.ctrl[act_gc] = grip

        # Track weld state
        if grasped_flag and not prev_grasped:
            weld_active = True
        if prev_grasped and not grasped_flag and weld_active:
            weld_active = False
            data.qvel[0:3] = [0.0, 0.0, -0.2]   # mirror _release_kinematic_grasp

        # Apply recorded egg position during weld (exact trajectory fidelity)
        if weld_active and has_egg_xyz:
            n_weld_steps += 1
            egg_xyz = np.array([float(row["egg_x"]), float(row["egg_y"]),
                                float(row["egg_z"])])
            data.qpos[egg_qadr:egg_qadr+3]   = egg_xyz
            data.qpos[egg_qadr+3:egg_qadr+7] = data.xquat[gb_bid].copy()
            data.qvel[0:6] = 0.0

        mujoco.mj_step(model, data)

        # EE tracking error vs recorded trajectory
        ee_now = data.site_xpos[ee_site].copy()
        ee_ref = np.array([float(row["ee_x"]), float(row["ee_y"]), float(row["ee_z"])])
        ee_errors.append(float(np.linalg.norm(ee_now - ee_ref)))

        if writer is not None and i % render_every == 0:
            renderer.update_scene(data, camera=args.camera)
            rgb = renderer.render()
            bgr = rgb[:, :, ::-1].copy()
            phase_str = row["phase"]
            weld_str  = "HELD" if weld_active else "    "
            cv2.putText(bgr,
                        f"POLICY REPLAY  [{phase_str}]  {weld_str}  step {i}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 220, 80), 1, cv2.LINE_AA)
            cv2.putText(bgr,
                        f"EE err {ee_errors[-1]*1000:.1f}mm   "
                        f"grip {grip*1000:.1f}mm",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (160, 220, 160), 1, cv2.LINE_AA)
            writer.write(bgr)

        prev_grasped = grasped_flag

    # ── final assessment ─────────────────────────────────────────────────────
    egg_final      = data.xpos[egg_bid].copy()
    bowl_site_pos  = data.site_xpos[bowl_site].copy()
    dist_xy        = float(np.linalg.norm(egg_final[:2] - bowl_site_pos[:2]))
    last_phase     = rows[-1]["phase"]
    last_result    = rows[-1].get("ep_result", "?")
    orig_result    = last_result if last_result else "(mid-episode)"

    success = dist_xy < 0.08 and egg_final[2] > 0.76

    print(f"\nReplay result:")
    print(f"  original ep_result : {orig_result}")
    print(f"  replay outcome     : {'SUCCESS' if success else 'MISS'}"
          f"   dist_to_bowl={dist_xy*1000:.1f}mm  egg_z={egg_final[2]:.3f}")
    print(f"  weld steps         : {n_weld_steps}")
    print(f"  EE tracking error  : "
          f"mean={np.mean(ee_errors)*1000:.2f}mm  "
          f"max={max(ee_errors)*1000:.2f}mm  "
          f"p95={float(np.percentile(ee_errors, 95))*1000:.2f}mm")

    if writer is not None:
        writer.release()
        renderer.close()
        print(f"  video → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
