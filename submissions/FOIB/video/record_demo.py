"""
Headless demo / benchmark recorder — FOIB-Egg

Single-episode demo (backward-compatible, no randomisation):
    python video/record_demo.py --out demo.mp4

Multi-episode benchmark:
    python video/record_demo.py --episodes 10 --tier medium --out benchmark.mp4
    python video/record_demo.py --episodes 10 --tier stress --out stress.mp4 --log results.csv

Tiers:   easy | medium (default) | stress
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import mujoco
except ImportError:
    sys.exit("mujoco not found — conda activate robothon")

try:
    import cv2
except ImportError:
    sys.exit("opencv-python not found — pip install opencv-python")

from controller.phase_controller import PhaseController, Phase
from video.overlay import draw

SCENE = os.path.join(os.path.dirname(__file__), "../models/scene.xml")
MAX_STEPS = 20_000

_EGG_DEFAULT  = np.array([0.26,  0.0,  0.800])
_EGG2_DEFAULT = np.array([0.26,  0.08, 0.800])   # distractor default (8 cm +Y)
_BOWL_DEFAULT = np.array([0.08,  0.22, 0.760])

_HOLD_TERMINAL = 4.0   # s — freeze terminal frame per episode
_HOLD_INTER    = 1.5   # s — inter-episode card
_HOLD_TITLE    = 2.5   # s — opening title card
_HOLD_END      = 4.0   # s — closing results card

# ── Dynamic disturbance constants ──────────────────────────────────────────────
# egg2 is given a random initial rolling velocity at episode start.
# Rolling is modelled as consistent (no-slip): omega = v / R_eff.
# With rolling friction=0.005, a 0.08 m/s kick travels ~65 mm before stopping.
DYNAMIC_VEL_LIN_MAX = 0.08   # m/s  — max initial linear speed for egg2
_EGG_ROLL_RADIUS    = 0.024  # m    — effective rolling radius (avg of Rx=25, Ry=22 mm)
ROLLING_THRESH      = 0.020  # m    — self-displacement threshold for DISTRACTOR_ROLLING

# Randomisation ranges per tier.
# Egg Y is capped at ±3 mm across all tiers: the kinematic-attach approach cannot
# tolerate larger Y offsets without fingers pushing the egg before attach fires.
TIER_PARAMS = {
    "easy": dict(
        egg_x   = 0.005,              # ±5 mm  — near-deterministic reach
        egg_y   = 0.001,              # ±1 mm
        egg_rot = np.radians(5),      # ±5°
        bowl_xy = 0.005,              # ±5 mm
    ),
    "medium": dict(
        egg_x   = 0.020,              # ±2 cm  — current validated range
        egg_y   = 0.003,              # ±3 mm
        egg_rot = np.radians(15),     # ±15°
        bowl_xy = 0.015,              # ±1.5 cm
    ),
    "stress": dict(
        egg_x   = 0.045,              # ±4.5 cm — ee_to_egg ≈49mm, 21mm below capture limit
        egg_y   = 0.003,              # ±3 mm   — locked (finger-closure physical constraint)
        egg_rot = np.radians(30),     # ±30°
        bowl_xy = 0.055,              # ±5.5 cm — max diagonal ≈78mm, near 80mm PLACE_THRESH
    ),
    "extreme": dict(
        egg_x   = 0.062,              # ±6.2 cm — ee_to_egg ≈65mm, 5mm below capture limit
        egg_y   = 0.003,              # ±3 mm   — locked
        egg_rot = np.radians(45),     # ±45°
        bowl_xy = 0.075,              # ±7.5 cm — max diagonal ≈106mm > 80mm → real FAIL zone
    ),
}

# ── Two-egg tier parameters ────────────────────────────────────────────────────
# min_egg_sep: minimum XY distance between target and distractor centre-to-centre.
# Egg diameter ≈ 44mm; sep > 22mm prevents rigid overlap.
# dist_stability_thresh: max allowable XY displacement of distractor (hard fail if exceeded).
# All tiers share the same egg/bowl randomisation (medium) so that only
# min_egg_sep drives difficulty. This gives a clean, reproducible gradient:
# easy 80mm → 10/10, medium 75mm → 7/10, stress 70mm → 4/10, extreme 66mm → 3/10
TWO_EGG_TIER_PARAMS = {
    "easy": dict(
        egg_x                 = 0.020,            # ±20 mm  (same as medium)
        egg_y                 = 0.003,            # ±3 mm
        egg_rot               = np.radians(15),   # ±15°
        bowl_xy               = 0.025,            # ±25 mm
        min_egg_sep           = 0.080,            # 80 mm  — arm never contacts distractor
        dist_stability_thresh = 0.020,            # 20 mm
    ),
    "medium": dict(
        egg_x                 = 0.020,            # ±20 mm
        egg_y                 = 0.003,            # ±3 mm
        egg_rot               = np.radians(15),   # ±15°
        bowl_xy               = 0.025,            # ±25 mm
        min_egg_sep           = 0.075,            # 75 mm  — 3/10 episodes arm contacts distractor
        dist_stability_thresh = 0.020,            # 20 mm
    ),
    "stress": dict(
        egg_x                 = 0.020,            # ±20 mm  (same base — only sep changes)
        egg_y                 = 0.003,            # ±3 mm
        egg_rot               = np.radians(15),   # ±15°
        bowl_xy               = 0.025,            # ±25 mm
        min_egg_sep           = 0.070,            # 70 mm  — 6/10 episodes arm contacts distractor
        dist_stability_thresh = 0.020,            # 20 mm
    ),
    "extreme": dict(
        egg_x                 = 0.020,            # ±20 mm  (same base — only sep changes)
        egg_y                 = 0.003,            # ±3 mm
        egg_rot               = np.radians(15),   # ±15°
        bowl_xy               = 0.025,            # ±25 mm
        min_egg_sep           = 0.066,            # 66 mm  — 7/10 episodes arm contacts distractor
        dist_stability_thresh = 0.020,            # 20 mm
    ),
}


_TRAJ_FIELDS_BASE = (
    "step", "episode_id", "tier", "phase", "ep_result",
    "j1_pos", "j2_pos", "j3_pos",
    "ee_x", "ee_y", "ee_z",
    "egg_x", "egg_y", "egg_z",
    "bowl_x", "bowl_y", "bowl_z",
    "grip_force", "contact_count", "grasped",
)

_TRAJ_FIELDS_TWO_EGG = (
    "egg2_x", "egg2_y", "egg2_z",
    "egg2_disp_mm", "distractor_rolling", "wrong_object_contact",
)


class _TrajWriter:
    """Writes one CSV per episode; each row is one sim step."""

    def __init__(self, collect_dir, episode_id, two_egg=False):
        os.makedirs(collect_dir, exist_ok=True)
        path = os.path.join(collect_dir, f"trajectory_ep{episode_id:03d}.csv")
        self._fh     = open(path, "w", newline="")
        self._w      = csv.writer(self._fh)
        self._fields = _TRAJ_FIELDS_BASE + (_TRAJ_FIELDS_TWO_EGG if two_egg else ())
        self._w.writerow(self._fields)

    def write_step(self, row: dict):
        self._w.writerow([row[f] for f in self._fields])

    def close(self):
        self._fh.close()


def _collect_contacts(model, data):
    """Snapshot all active contacts: returns list of (geom1_name, geom2_name, normal_force_N)."""
    force_buf = np.zeros(6)
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force_buf)
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
        out.append((n1, n2, abs(float(force_buf[0]))))
    return out


def _ee_jac_frob(model, data, site_id):
    """Frobenius norm of the 3×nv translational end-effector Jacobian (mj_jacSite)."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return float(np.linalg.norm(jacp))


def parse_args():
    p = argparse.ArgumentParser(description="FOIB-Egg benchmark recorder")
    p.add_argument("--out",      default="demo.mp4")
    p.add_argument("--fps",      type=int, default=30)
    p.add_argument("--width",    type=int, default=640)
    p.add_argument("--height",   type=int, default=480)
    p.add_argument("--camera",   default="side_cam")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--tier",     default="medium",
                   choices=["easy", "medium", "stress", "extreme"])
    p.add_argument("--log",      default=None, metavar="PATH",
                   help="write per-episode CSV to PATH")
    p.add_argument("--two-egg",  action="store_true",
                   help="two-egg benchmark mode: target + distractor")
    p.add_argument("--dynamic-dist", action="store_true",
                   help="give egg2 a random initial rolling velocity (dynamic disturbance mode)")
    p.add_argument("--collect", action="store_true",
                   help="export per-step trajectory CSV for every episode")
    p.add_argument("--collect-dir", default="trajectories",
                   help="output directory for trajectory CSVs (default: trajectories/)")
    return p.parse_args()


# ── scene randomisation ────────────────────────────────────────────────────────

def _randomize(model, data, ctrl, rng, bowl_bid, tp):
    """Perturb bowl body pos (model) and egg freejoint qpos using tier params tp."""
    bdx, bdy = rng.uniform(-tp["bowl_xy"], tp["bowl_xy"], 2)
    model.body_pos[bowl_bid] = _BOWL_DEFAULT + [bdx, bdy, 0.0]

    # reset() → mj_resetData + arm teleport + mj_forward; captures new bowl_pos
    ctrl.reset()

    edx    = rng.uniform(-tp["egg_x"],   tp["egg_x"])
    edy    = rng.uniform(-tp["egg_y"],   tp["egg_y"])
    dtheta = rng.uniform(-tp["egg_rot"], tp["egg_rot"])
    adr = ctrl.egg_qpos_adr
    data.qpos[adr:adr+3]  = _EGG_DEFAULT + [edx, edy, 0.0]
    data.qpos[adr+3:adr+7] = [np.cos(dtheta / 2), 0.0, 0.0, np.sin(dtheta / 2)]
    mujoco.mj_forward(model, data)


# ── two-egg scene randomisation ───────────────────────────────────────────────

def _randomize_two_egg(model, data, ctrl, rng, bowl_bid, egg2_bid,
                        egg2_qpos_adr, egg2_dof_adr, tp, dynamic=False):
    """Place bowl, target egg, and distractor egg.

    Returns (distractor_init_xy, init_lin_vel_ms, init_ang_vel_rads).
    When dynamic=True, egg2 receives a random rolling kick after placement.
    """
    bdx, bdy = rng.uniform(-tp["bowl_xy"], tp["bowl_xy"], 2)
    model.body_pos[bowl_bid] = _BOWL_DEFAULT + [bdx, bdy, 0.0]
    ctrl.reset()

    edx    = rng.uniform(-tp["egg_x"],   tp["egg_x"])
    edy    = rng.uniform(-tp["egg_y"],   tp["egg_y"])
    dtheta = rng.uniform(-tp["egg_rot"], tp["egg_rot"])
    adr    = ctrl.egg_qpos_adr
    target_pos = _EGG_DEFAULT + np.array([edx, edy, 0.0])
    data.qpos[adr:adr+3]   = target_pos
    data.qpos[adr+3:adr+7] = [np.cos(dtheta / 2), 0.0, 0.0, np.sin(dtheta / 2)]

    # Distractor: offset from target in random XY direction by min_egg_sep
    theta    = rng.uniform(0, 2 * np.pi)
    sep      = tp["min_egg_sep"]
    dist_pos = target_pos + np.array([sep * np.cos(theta), sep * np.sin(theta), 0.0])
    data.qpos[egg2_qpos_adr:egg2_qpos_adr+3]   = dist_pos
    data.qpos[egg2_qpos_adr+3:egg2_qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]

    mujoco.mj_forward(model, data)

    init_lin_vel = 0.0
    init_ang_vel = 0.0
    if dynamic:
        # Consistent rolling kick: omega = v / R_eff so egg rolls without initial slip.
        phi   = rng.uniform(0, 2 * np.pi)
        v     = rng.uniform(DYNAMIC_VEL_LIN_MAX * 0.3, DYNAMIC_VEL_LIN_MAX)
        omega = v / _EGG_ROLL_RADIUS
        # Linear velocity in XY plane
        data.qvel[egg2_dof_adr + 0] = v * np.cos(phi)
        data.qvel[egg2_dof_adr + 1] = v * np.sin(phi)
        data.qvel[egg2_dof_adr + 2] = 0.0
        # Angular velocity perpendicular to roll direction (right-hand rule)
        data.qvel[egg2_dof_adr + 3] =  omega * np.sin(phi)
        data.qvel[egg2_dof_adr + 4] = -omega * np.cos(phi)
        data.qvel[egg2_dof_adr + 5] = rng.uniform(-0.5, 0.5)  # small axial spin
        init_lin_vel = round(v, 4)
        init_ang_vel = round(float(np.linalg.norm(
            data.qvel[egg2_dof_adr+3:egg2_dof_adr+6])), 3)

    return dist_pos[:2].copy(), init_lin_vel, init_ang_vel


# ── title / inter / end cards ─────────────────────────────────────────────────

def _centred_text(img, text, cy, scale, col, thickness=2):
    """Draw horizontally-centred text at row cy. Returns next cy."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cx = max(10, (img.shape[1] - tw) // 2)
    cv2.putText(img, text, (cx, cy), font, scale, col, thickness, cv2.LINE_AA)
    return cy + max(24, int(th * 1.9 + 6))


def _title_card(width, height, tier, n_eps, seed):
    """BGR opening card: project name, tier, task summary."""
    card = np.zeros((height, width, 3), dtype=np.uint8)
    cy = height // 2 - 100
    cy = _centred_text(card, "FOIB-Egg",
                       cy, 1.10, (255, 200, 80), thickness=3)
    cy = _centred_text(card, "Fragile Object Integrity Benchmark",
                       cy, 0.62, (200, 200, 200))
    cy += 14
    cy = _centred_text(card,
                       f"Tier: {tier.upper()}   Episodes: {n_eps}   Seed: {seed}",
                       cy, 0.55, (150, 150, 150))
    cy = _centred_text(card,
                       "Shell integrity is the primary scoring criterion",
                       cy, 0.48, (110, 110, 110))
    return card


def _inter_card(width, height, ep_num, n_eps, tier, fail_str, successes):
    """BGR card shown between episodes."""
    card = np.zeros((height, width, 3), dtype=np.uint8)

    if not fail_str:
        shell_text = "SHELL INTEGRITY :  INTACT"
        shell_col  = (80, 200, 80)
    elif fail_str == "OVER-SQUEEZED":
        shell_text = "SHELL INTEGRITY :  OVER-SQUEEZED"
        shell_col  = (60, 60, 220)
    else:
        shell_text = f"SHELL INTEGRITY :  {fail_str}"
        shell_col  = (40, 180, 220)

    cy = height // 2 - 62
    cy = _centred_text(card,
                       f"EPISODE  {ep_num} / {n_eps}   [{tier.upper()}]",
                       cy, 0.78, (210, 210, 210))
    cy = _centred_text(card, shell_text,   cy, 0.70, shell_col)
    cy = _centred_text(card,
                       f"SCORE  {successes} / {ep_num}",
                       cy, 0.65, (155, 155, 155))
    return card


def _end_card(width, height, tier, records):
    """BGR closing card with benchmark statistics."""
    n         = len(records)
    successes = sum(1 for r in records if r["result"] == "SUCCESS")
    over_sq   = sum(1 for r in records if "OVER-SQUEEZED" in r["result"])
    dropped   = sum(1 for r in records if "DROPPED"       in r["result"])
    timeout   = sum(1 for r in records if "TIMEOUT"       in r["result"])
    max_grip  = max(r["peak_grip"] for r in records)
    avg_steps = int(np.mean([r["steps"] for r in records]))
    pct       = 100 * successes // n if n else 0

    card = np.zeros((height, width, 3), dtype=np.uint8)
    cy   = max(28, height // 2 - 148)

    cy = _centred_text(card, "BENCHMARK RESULTS",
                       cy, 0.88, (255, 200, 80), thickness=2)
    cy = _centred_text(card,
                       f"Tier: {tier.upper()}   |   Episodes: {n}",
                       cy, 0.58, (180, 180, 180))
    cy += 6
    # separator
    cv2.line(card, (width // 5, cy), (4 * width // 5, cy), (70, 70, 70), 1)
    cy += 14

    def _row(text, col, scale=0.60):
        nonlocal cy
        cy = _centred_text(card, text, cy, scale, col)

    _row(f"Shell INTACT      {successes:>3} / {n}  ({pct:3}%)",
         (80, 200, 80))
    _row(f"OVER-SQUEEZED     {over_sq:>3} / {n}",
         (60, 60, 220)   if over_sq  else (90, 90, 90))
    _row(f"Dropped           {dropped:>3} / {n}",
         (40, 180, 220)  if dropped  else (90, 90, 90))
    _row(f"Timeout           {timeout:>3} / {n}",
         (150, 150, 220) if timeout  else (90, 90, 90))

    cy += 6
    cv2.line(card, (width // 5, cy), (4 * width // 5, cy), (70, 70, 70), 1)
    cy += 14

    _row(f"Peak grip (max)   {max_grip:.3f} N",  (180, 180, 180), 0.55)
    _row(f"Avg steps / ep    {avg_steps}",        (180, 180, 180), 0.55)
    return card


def _title_card_two_egg(width, height, tier, n_eps, seed):
    """BGR opening card for two-egg benchmark."""
    card = np.zeros((height, width, 3), dtype=np.uint8)
    cy = height // 2 - 100
    cy = _centred_text(card, "FOIB-Egg v2",
                       cy, 1.05, (255, 200, 80), thickness=3)
    cy = _centred_text(card, "Two-Egg: Target Selection + Distractor Suppression",
                       cy, 0.50, (200, 200, 200))
    cy += 14
    cy = _centred_text(card,
                       f"Tier: {tier.upper()}   Episodes: {n_eps}   Seed: {seed}",
                       cy, 0.55, (150, 150, 150))
    cy = _centred_text(card,
                       "Distractor must not be grasped or significantly displaced",
                       cy, 0.44, (110, 110, 110))
    return card


def _end_card_two_egg(width, height, tier, records, dynamic=False):
    """BGR closing card with two-egg benchmark statistics."""
    n          = len(records)
    successes  = sum(1 for r in records if r["result"] == "SUCCESS")
    rolling    = sum(1 for r in records if "DISTRACTOR_ROLLING"   in r["result"])
    disturbed  = sum(1 for r in records if "DISTRACTOR_DISTURBED" in r["result"])
    dropped    = sum(1 for r in records if "DROPPED"              in r["result"])
    timeout    = sum(1 for r in records if "TIMEOUT"              in r["result"])
    wrong_ct   = sum(1 for r in records if r["wrong_object_contact"])
    avg_disp   = float(np.mean([r["distractor_displacement_mm"] for r in records]))
    max_grip   = max(r["peak_grip"] for r in records)
    avg_steps  = int(np.mean([r["steps"] for r in records]))
    pct        = 100 * successes // n if n else 0

    card = np.zeros((height, width, 3), dtype=np.uint8)
    # Dynamic mode needs one extra result row → start slightly higher
    cy = max(14, height // 2 - (180 if dynamic else 170))

    title = ("FOIB-Egg v2  TWO-EGG [DYNAMIC]" if dynamic
             else "FOIB-Egg v2  TWO-EGG RESULTS")
    cy = _centred_text(card, title, cy, 0.76, (255, 200, 80), thickness=2)
    cy = _centred_text(card,
                       f"Tier: {tier.upper()}   |   Episodes: {n}",
                       cy, 0.54, (180, 180, 180))
    cy += 6
    cv2.line(card, (width // 5, cy), (4 * width // 5, cy), (70, 70, 70), 1)
    cy += 14

    def _row(text, col, scale=0.54):
        nonlocal cy
        cy = _centred_text(card, text, cy, scale, col)

    _row(f"Target SUCCESS        {successes:>3} / {n}  ({pct:3}%)", (80, 200, 80))
    if dynamic:
        _row(f"DISTRACTOR_ROLLING    {rolling:>3} / {n}",
             (180, 100, 255) if rolling else (90, 90, 90))
    _row(f"DISTRACTOR_DISTURBED  {disturbed:>3} / {n}",
         (60, 60, 220)   if disturbed else (90, 90, 90))
    _row(f"Dropped               {dropped:>3} / {n}",
         (40, 180, 220)  if dropped   else (90, 90, 90))
    _row(f"Timeout               {timeout:>3} / {n}",
         (150, 150, 220) if timeout   else (90, 90, 90))
    _row(f"Wrong contact (info)  {wrong_ct:>3} / {n}",
         (200, 180, 100) if wrong_ct  else (90, 90, 90))

    cy += 6
    cv2.line(card, (width // 5, cy), (4 * width // 5, cy), (70, 70, 70), 1)
    cy += 14

    _row(f"Peak grip (max)       {max_grip:.3f} N",   (180, 180, 180), 0.50)
    _row(f"Avg distractor disp   {avg_disp:.1f} mm",  (180, 180, 180), 0.50)
    _row(f"Avg steps / ep        {avg_steps}",         (180, 180, 180), 0.50)

    if dynamic:
        _row("DYNAMIC DIST: egg2 init vel active each episode", (200, 180, 100), 0.44)
    else:
        _row("TWO-EGG: 10->7->4->3  (easy->extreme)",  (200, 180, 100), 0.44)
        _row("FAIL: DISTRACTOR_DISTURBED / DROPPED",    (150, 150, 200), 0.42)
    return card


# ── episode runner ─────────────────────────────────────────────────────────────

def _run_episode(ctrl, model, data, renderer, writer, args,
                 ep_num, n_eps, prior_successes, tier,
                 traj_writer=None):
    """Run one episode, write frames.

    Returns (phase, fail_str, frames_written, peak_grip).
    peak_grip is sampled at every render frame (every ~17 sim steps).
    """
    render_every   = max(1, int(round(1.0 / (args.fps * model.opt.timestep))))
    frames_written = 0
    phase          = Phase.IDLE
    fail           = None
    peak_grip      = 0.0
    max_contacts   = 0
    contacts_end   = []
    jac_frob       = 0.0

    for step_i in range(MAX_STEPS):
        phase, fail = ctrl.step()
        mujoco.mj_step(model, data)

        is_terminal = phase in (Phase.DONE, Phase.FAIL)

        if traj_writer is not None and not is_terminal:
            _bowl = ctrl._bowl_pos if ctrl._bowl_pos is not None else np.zeros(3)
            traj_writer.write_step({
                "step": step_i, "episode_id": ep_num, "tier": tier,
                "phase": phase.name, "ep_result": "",
                "j1_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[0]]), 5),
                "j2_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[1]]), 5),
                "j3_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[2]]), 5),
                "ee_x": round(float(data.site_xpos[ctrl.ee_site][0]), 5),
                "ee_y": round(float(data.site_xpos[ctrl.ee_site][1]), 5),
                "ee_z": round(float(data.site_xpos[ctrl.ee_site][2]), 5),
                "egg_x": round(float(data.xpos[ctrl.egg_id][0]), 5),
                "egg_y": round(float(data.xpos[ctrl.egg_id][1]), 5),
                "egg_z": round(float(data.xpos[ctrl.egg_id][2]), 5),
                "bowl_x": round(float(_bowl[0]), 5),
                "bowl_y": round(float(_bowl[1]), 5),
                "bowl_z": round(float(_bowl[2]), 5),
                "grip_force": round(float(abs(data.sensordata[ctrl.sen_grip])), 4),
                "contact_count": sum(1 for i in range(data.ncon)
                                     if data.contact[i].geom1 in ctrl._egg_geom_ids
                                     or data.contact[i].geom2 in ctrl._egg_geom_ids),
                "grasped": int(ctrl._weld_active),
            })

        if step_i % render_every == 0:
            renderer.update_scene(data, camera=args.camera)
            rgb  = renderer.render()
            info = ctrl.overlay_info()
            peak_grip   = max(peak_grip,   info["grip_force"])
            max_contacts = max(max_contacts, info.get("contact_count", 0))
            info.update({
                "episode_num":  ep_num,
                "n_episodes":   n_eps,
                "ep_successes": prior_successes,
                "tier":         tier,
            })
            writer.write(draw(rgb, info)[:, :, ::-1])
            frames_written += 1

        if is_terminal:
            contacts_end = _collect_contacts(model, data)
            jac_frob     = _ee_jac_frob(model, data, ctrl.ee_site)
            updated_successes = prior_successes + (1 if phase == Phase.DONE else 0)
            if traj_writer is not None:
                _fail_s = fail.value if fail is not None else ""
                _ep_res = "SUCCESS" if phase == Phase.DONE else f"FAIL:{_fail_s}"
                _bowl   = ctrl._bowl_pos if ctrl._bowl_pos is not None else np.zeros(3)
                traj_writer.write_step({
                    "step": step_i, "episode_id": ep_num, "tier": tier,
                    "phase": phase.name, "ep_result": _ep_res,
                    "j1_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[0]]), 5),
                    "j2_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[1]]), 5),
                    "j3_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[2]]), 5),
                    "ee_x": round(float(data.site_xpos[ctrl.ee_site][0]), 5),
                    "ee_y": round(float(data.site_xpos[ctrl.ee_site][1]), 5),
                    "ee_z": round(float(data.site_xpos[ctrl.ee_site][2]), 5),
                    "egg_x": round(float(data.xpos[ctrl.egg_id][0]), 5),
                    "egg_y": round(float(data.xpos[ctrl.egg_id][1]), 5),
                    "egg_z": round(float(data.xpos[ctrl.egg_id][2]), 5),
                    "bowl_x": round(float(_bowl[0]), 5),
                    "bowl_y": round(float(_bowl[1]), 5),
                    "bowl_z": round(float(_bowl[2]), 5),
                    "grip_force": round(float(abs(data.sensordata[ctrl.sen_grip])), 4),
                    "contact_count": sum(1 for i in range(data.ncon)
                                         if data.contact[i].geom1 in ctrl._egg_geom_ids
                                         or data.contact[i].geom2 in ctrl._egg_geom_ids),
                    "grasped": int(ctrl._weld_active),
                })
            renderer.update_scene(data, camera=args.camera)
            rgb  = renderer.render()
            info = ctrl.overlay_info()
            peak_grip    = max(peak_grip,    info["grip_force"])
            max_contacts = max(max_contacts, info.get("contact_count", 0))
            info.update({
                "episode_num":  ep_num,
                "n_episodes":   n_eps,
                "ep_successes": updated_successes,
                "tier":         tier,
            })
            hold_bgr = draw(rgb, info)[:, :, ::-1]
            hold_n   = int(_HOLD_TERMINAL * args.fps)
            for _ in range(hold_n):
                writer.write(hold_bgr)
            frames_written += hold_n
            break

    fail_str = fail.value if fail is not None else ""
    return phase, fail_str, frames_written, peak_grip, max_contacts, contacts_end, jac_frob


def _run_episode_two_egg(ctrl, model, data, renderer, writer, args,
                          ep_num, n_eps, prior_successes, tier,
                          egg2_bid, egg2_geom_ids, finger_geom_ids,
                          distractor_init_xy, dist_stability_thresh,
                          dynamic=False, traj_writer=None):
    """Run one two-egg episode.  Returns metrics dict."""
    render_every         = max(1, int(round(1.0 / (args.fps * model.opt.timestep))))
    frames_written       = 0
    phase                = Phase.IDLE
    fail                 = None
    peak_grip            = 0.0
    max_contacts         = 0
    wrong_object_contact = False
    max_distractor_disp  = 0.0
    target_pick_success  = False
    contacts_end         = []
    jac_frob             = 0.0
    # Dynamic disturbance tracking: self-displacement measured only before
    # any finger contact so that rolling is distinguished from arm-induced disturbance.
    finger_contact_ever  = False
    egg2_self_disp_max   = 0.0

    for step_i in range(MAX_STEPS):
        phase, fail = ctrl.step()
        mujoco.mj_step(model, data)

        # Accumulate self-displacement while no finger has touched egg2 yet
        if dynamic and not finger_contact_ever:
            egg2_xy_now = data.xpos[egg2_bid, :2]
            sd = float(np.linalg.norm(egg2_xy_now - distractor_init_xy))
            egg2_self_disp_max = max(egg2_self_disp_max, sd)

        # wrong-contact: finger geom ↔ egg2 geom, checked every sim step
        if not wrong_object_contact:
            for ci in range(data.ncon):
                g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
                if ((g1 in egg2_geom_ids and g2 in finger_geom_ids) or
                        (g2 in egg2_geom_ids and g1 in finger_geom_ids)):
                    wrong_object_contact = True
                    finger_contact_ever  = True
                    break

        is_terminal = phase in (Phase.DONE, Phase.FAIL)

        if traj_writer is not None and not is_terminal:
            _bowl = ctrl._bowl_pos if ctrl._bowl_pos is not None else np.zeros(3)
            _e2xy = data.xpos[egg2_bid, :2]
            traj_writer.write_step({
                "step": step_i, "episode_id": ep_num, "tier": tier,
                "phase": phase.name, "ep_result": "",
                "j1_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[0]]), 5),
                "j2_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[1]]), 5),
                "j3_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[2]]), 5),
                "ee_x": round(float(data.site_xpos[ctrl.ee_site][0]), 5),
                "ee_y": round(float(data.site_xpos[ctrl.ee_site][1]), 5),
                "ee_z": round(float(data.site_xpos[ctrl.ee_site][2]), 5),
                "egg_x": round(float(data.xpos[ctrl.egg_id][0]), 5),
                "egg_y": round(float(data.xpos[ctrl.egg_id][1]), 5),
                "egg_z": round(float(data.xpos[ctrl.egg_id][2]), 5),
                "bowl_x": round(float(_bowl[0]), 5),
                "bowl_y": round(float(_bowl[1]), 5),
                "bowl_z": round(float(_bowl[2]), 5),
                "grip_force": round(float(abs(data.sensordata[ctrl.sen_grip])), 4),
                "contact_count": sum(1 for i in range(data.ncon)
                                     if data.contact[i].geom1 in ctrl._egg_geom_ids
                                     or data.contact[i].geom2 in ctrl._egg_geom_ids),
                "grasped": int(ctrl._weld_active),
                "egg2_x": round(float(data.xpos[egg2_bid][0]), 5),
                "egg2_y": round(float(data.xpos[egg2_bid][1]), 5),
                "egg2_z": round(float(data.xpos[egg2_bid][2]), 5),
                "egg2_disp_mm": round(float(np.linalg.norm(_e2xy - distractor_init_xy)) * 1000, 2),
                "distractor_rolling": int(dynamic and egg2_self_disp_max > ROLLING_THRESH),
                "wrong_object_contact": int(wrong_object_contact),
            })

        if step_i % render_every == 0:
            renderer.update_scene(data, camera=args.camera)
            rgb  = renderer.render()
            info = ctrl.overlay_info()
            peak_grip    = max(peak_grip,    info["grip_force"])
            max_contacts = max(max_contacts, info.get("contact_count", 0))
            if info.get("grasped"):
                target_pick_success = True

            egg2_xy  = data.xpos[egg2_bid, :2].copy()
            dist_disp = float(np.linalg.norm(egg2_xy - distractor_init_xy))
            max_distractor_disp = max(max_distractor_disp, dist_disp)
            egg_sep  = float(np.linalg.norm(egg2_xy - data.xpos[ctrl.egg_id, :2]))

            info.update({
                "episode_num":     ep_num,
                "n_episodes":      n_eps,
                "ep_successes":    prior_successes,
                "tier":            tier,
                "distractor_disp": dist_disp,
                "egg_sep":         egg_sep,
            })
            writer.write(draw(rgb, info)[:, :, ::-1])
            frames_written += 1

        if is_terminal:
            contacts_end = _collect_contacts(model, data)
            jac_frob     = _ee_jac_frob(model, data, ctrl.ee_site)
            egg2_xy_f  = data.xpos[egg2_bid, :2].copy()
            final_disp = float(np.linalg.norm(egg2_xy_f - distractor_init_xy))
            max_distractor_disp = max(max_distractor_disp, final_disp)
            distractor_stable   = max_distractor_disp < dist_stability_thresh
            distractor_rolling  = dynamic and (egg2_self_disp_max > ROLLING_THRESH)

            fail_str = fail.value if fail is not None else ""
            if phase == Phase.DONE and distractor_rolling:
                # egg2 self-displaced (no arm contact required) — dynamic mode only
                result      = "FAIL:DISTRACTOR_ROLLING"
                updated_suc = prior_successes
            elif phase == Phase.DONE and not distractor_stable:
                result      = "FAIL:DISTRACTOR_DISTURBED"
                updated_suc = prior_successes
            elif phase == Phase.DONE:
                result      = "SUCCESS"
                updated_suc = prior_successes + 1
            else:
                result      = f"FAIL:{fail_str}"
                updated_suc = prior_successes

            if traj_writer is not None:
                _bowl = ctrl._bowl_pos if ctrl._bowl_pos is not None else np.zeros(3)
                traj_writer.write_step({
                    "step": step_i, "episode_id": ep_num, "tier": tier,
                    "phase": phase.name, "ep_result": result,
                    "j1_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[0]]), 5),
                    "j2_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[1]]), 5),
                    "j3_pos": round(float(data.qpos[ctrl._arm_jnt_qposadr[2]]), 5),
                    "ee_x": round(float(data.site_xpos[ctrl.ee_site][0]), 5),
                    "ee_y": round(float(data.site_xpos[ctrl.ee_site][1]), 5),
                    "ee_z": round(float(data.site_xpos[ctrl.ee_site][2]), 5),
                    "egg_x": round(float(data.xpos[ctrl.egg_id][0]), 5),
                    "egg_y": round(float(data.xpos[ctrl.egg_id][1]), 5),
                    "egg_z": round(float(data.xpos[ctrl.egg_id][2]), 5),
                    "bowl_x": round(float(_bowl[0]), 5),
                    "bowl_y": round(float(_bowl[1]), 5),
                    "bowl_z": round(float(_bowl[2]), 5),
                    "grip_force": round(float(abs(data.sensordata[ctrl.sen_grip])), 4),
                    "contact_count": sum(1 for i in range(data.ncon)
                                         if data.contact[i].geom1 in ctrl._egg_geom_ids
                                         or data.contact[i].geom2 in ctrl._egg_geom_ids),
                    "grasped": int(ctrl._weld_active),
                    "egg2_x": round(float(data.xpos[egg2_bid][0]), 5),
                    "egg2_y": round(float(data.xpos[egg2_bid][1]), 5),
                    "egg2_z": round(float(data.xpos[egg2_bid][2]), 5),
                    "egg2_disp_mm": round(final_disp * 1000, 2),
                    "distractor_rolling": int(distractor_rolling),
                    "wrong_object_contact": int(wrong_object_contact),
                })

            renderer.update_scene(data, camera=args.camera)
            rgb  = renderer.render()
            info = ctrl.overlay_info()
            peak_grip    = max(peak_grip,    info["grip_force"])
            max_contacts = max(max_contacts, info.get("contact_count", 0))
            egg_sep_f = float(np.linalg.norm(
                egg2_xy_f - data.xpos[ctrl.egg_id, :2]))
            info.update({
                "episode_num":     ep_num,
                "n_episodes":      n_eps,
                "ep_successes":    updated_suc,
                "tier":            tier,
                "distractor_disp": final_disp,
                "egg_sep":         egg_sep_f,
            })
            hold_bgr = draw(rgb, info)[:, :, ::-1]
            hold_n   = int(_HOLD_TERMINAL * args.fps)
            for _ in range(hold_n):
                writer.write(hold_bgr)
            frames_written += hold_n
            break

    # Final bowl distance (XY)
    bowl_pos = ctrl._bowl_pos
    egg1_pos = data.xpos[ctrl.egg_id]
    target_final_dist = (float(np.linalg.norm(egg1_pos[:2] - bowl_pos[:2]))
                         if bowl_pos is not None else float("nan"))

    # Recompute stable/rolling from max (worst-case, not final)
    distractor_stable  = max_distractor_disp < dist_stability_thresh
    distractor_rolling = dynamic and (egg2_self_disp_max > ROLLING_THRESH)
    fail_str_out = fail.value if fail is not None else ""
    if phase == Phase.DONE and distractor_rolling:
        result = "FAIL:DISTRACTOR_ROLLING"
    elif phase == Phase.DONE and not distractor_stable:
        result = "FAIL:DISTRACTOR_DISTURBED"
    elif phase == Phase.DONE:
        result = "SUCCESS"
    else:
        result = f"FAIL:{fail_str_out}"

    return {
        "phase":                        phase,
        "result":                       result,
        "frames_written":               frames_written,
        "peak_grip":                    peak_grip,
        "contact_count":                max_contacts,
        "target_success":               result == "SUCCESS",
        "wrong_object_contact":         wrong_object_contact,
        "wrong_object_grasp":           False,
        "distractor_displacement_mm":   round(max_distractor_disp * 1000, 1),
        "distractor_stable":            distractor_stable,
        "distractor_rolling":           distractor_rolling,
        "egg2_self_disp_mm":            round(egg2_self_disp_max * 1000, 1),
        "target_pick_success":          target_pick_success,
        "target_place_success":         phase == Phase.DONE,
        "target_final_dist_to_bowl_mm": round(target_final_dist * 1000, 1),
        "contacts_at_end":              contacts_end,
        "ee_jac_frob":                  jac_frob,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    tp   = TWO_EGG_TIER_PARAMS[args.tier] if args.two_egg else TIER_PARAMS[args.tier]

    model    = mujoco.MjModel.from_xml_path(SCENE)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    ctrl     = PhaseController(model, data)
    bowl_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        sys.exit(f"Cannot open video writer for {args.out}")

    csv_fh = csv_writer = None
    if args.log:
        csv_fh     = open(args.log, "w", newline="")
        csv_writer = csv.writer(csv_fh)

    rng          = np.random.default_rng(args.seed)
    n_eps        = args.episodes
    successes    = 0
    total_frames = 0
    records      = []
    render_every = max(1, int(round(1.0 / (args.fps * model.opt.timestep))))

    mode_tag = "two-egg" if args.two_egg else "single-egg"
    print(f"Recording → {args.out}  ({args.fps} fps, camera={args.camera})")
    print(f"  mode={mode_tag}  tier={args.tier}  episodes={n_eps}  seed={args.seed}"
          f"  render_every={render_every}")

    t0 = time.time()

    if args.two_egg:
        # ── Two-egg mode ──────────────────────────────────────────────────────────
        egg2_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "egg2")
        egg2_jnt  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "egg2_free")
        egg2_qpos_adr = model.jnt_qposadr[egg2_jnt]
        egg2_dof_adr  = model.jnt_dofadr[egg2_jnt]
        egg2_geom_ids = frozenset(
            i for i in range(model.ngeom)
            if model.geom_bodyid[i] == egg2_bid
        )
        fl_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_left")
        fr_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_right")
        finger_geom_ids = frozenset(
            i for i in range(model.ngeom)
            if model.geom_bodyid[i] in {fl_bid, fr_bid}
        )

        if csv_writer:
            csv_writer.writerow([
                "episode_id", "tier", "result",
                "target_success", "wrong_object_contact", "wrong_object_grasp",
                "distractor_displacement_mm", "distractor_stable",
                "distractor_rolling", "egg2_self_disp_mm",
                "egg2_init_lin_vel", "egg2_init_ang_vel",
                "target_pick_success", "target_place_success",
                "target_final_dist_to_bowl_mm",
                "steps", "peak_grip", "contact_count",
                "target_egg_id", "distractor_egg_id",
            ])

        if n_eps > 1:
            title   = _title_card_two_egg(args.width, args.height, args.tier, n_eps, args.seed)
            title_n = int(_HOLD_TITLE * args.fps)
            for _ in range(title_n):
                writer.write(title)
            total_frames += title_n

        for ep in range(n_eps):
            ep_num = ep + 1
            print(f"  [EP {ep_num}/{n_eps}]", end=" ", flush=True)

            if n_eps == 1:
                ctrl.reset()
                distractor_init_xy = _EGG2_DEFAULT[:2].copy()
                init_lin_vel = init_ang_vel = 0.0
            else:
                distractor_init_xy, init_lin_vel, init_ang_vel = _randomize_two_egg(
                    model, data, ctrl, rng, bowl_bid,
                    egg2_bid, egg2_qpos_adr, egg2_dof_adr, tp,
                    dynamic=args.dynamic_dist,
                )

            traj_writer = (_TrajWriter(args.collect_dir, ep_num, two_egg=True)
                           if args.collect else None)
            m = _run_episode_two_egg(
                ctrl, model, data, renderer, writer, args,
                ep_num, n_eps, successes, args.tier,
                egg2_bid, egg2_geom_ids, finger_geom_ids,
                distractor_init_xy, tp["dist_stability_thresh"],
                dynamic=args.dynamic_dist,
                traj_writer=traj_writer,
            )
            if traj_writer is not None:
                traj_writer.close()
            total_frames += m["frames_written"]

            if m["result"] == "SUCCESS":
                successes += 1

            rec = {
                "episode_id":                   ep_num,
                "tier":                         args.tier,
                "result":                       m["result"],
                "target_success":               m["target_success"],
                "wrong_object_contact":         m["wrong_object_contact"],
                "wrong_object_grasp":           m["wrong_object_grasp"],
                "distractor_displacement_mm":   m["distractor_displacement_mm"],
                "distractor_stable":            m["distractor_stable"],
                "distractor_rolling":           m["distractor_rolling"],
                "egg2_self_disp_mm":            m["egg2_self_disp_mm"],
                "egg2_init_lin_vel":            init_lin_vel,
                "egg2_init_ang_vel":            init_ang_vel,
                "target_pick_success":          m["target_pick_success"],
                "target_place_success":         m["target_place_success"],
                "target_final_dist_to_bowl_mm": m["target_final_dist_to_bowl_mm"],
                "steps":                        ctrl.step_count,
                "peak_grip":                    m["peak_grip"],
                "contact_count":                m["contact_count"],
            }
            records.append(rec)

            if csv_writer:
                csv_writer.writerow([
                    rec["episode_id"], rec["tier"], rec["result"],
                    rec["target_success"], rec["wrong_object_contact"],
                    rec["wrong_object_grasp"],
                    rec["distractor_displacement_mm"], rec["distractor_stable"],
                    rec["distractor_rolling"], rec["egg2_self_disp_mm"],
                    rec["egg2_init_lin_vel"], rec["egg2_init_ang_vel"],
                    rec["target_pick_success"], rec["target_place_success"],
                    rec["target_final_dist_to_bowl_mm"],
                    rec["steps"], f"{rec['peak_grip']:.4f}", rec["contact_count"],
                    "egg", "egg2",
                ])

            rolling_tag = (f"  rolling={rec['egg2_self_disp_mm']:.1f}mm"
                           if args.dynamic_dist else "")
            print(f"{rec['result']}  steps={rec['steps']}"
                  f"  peak={rec['peak_grip']:.3f}N"
                  f"  dist_disp={rec['distractor_displacement_mm']:.1f}mm"
                  f"  wc={int(rec['wrong_object_contact'])}"
                  f"{rolling_tag}")
            print(f"    JAC_EE ||J||_F={m['ee_jac_frob']:.4f}"
                  f"  contacts@end={len(m['contacts_at_end'])}")
            for cn1, cn2, fn in m["contacts_at_end"]:
                print(f"      {cn1} <-> {cn2}  Fn={fn:.3f}N")

            if ep < n_eps - 1:
                fail_tag = ("" if m["result"] == "SUCCESS"
                            else m["result"].split(":", 1)[-1])
                card    = _inter_card(args.width, args.height,
                                      ep_num, n_eps, args.tier, fail_tag, successes)
                inter_n = int(_HOLD_INTER * args.fps)
                for _ in range(inter_n):
                    writer.write(card)
                total_frames += inter_n

        if n_eps > 1:
            end   = _end_card_two_egg(args.width, args.height, args.tier, records,
                                      dynamic=args.dynamic_dist)
            end_n = int(_HOLD_END * args.fps)
            for _ in range(end_n):
                writer.write(end)
            total_frames += end_n

    else:
        # ── Single-egg mode (original, unchanged) ─────────────────────────────────
        if csv_writer:
            csv_writer.writerow(["ep", "tier", "result", "grip_peak", "contact_max", "steps"])

        if n_eps > 1:
            title = _title_card(args.width, args.height, args.tier, n_eps, args.seed)
            title_n = int(_HOLD_TITLE * args.fps)
            for _ in range(title_n):
                writer.write(title)
            total_frames += title_n

        for ep in range(n_eps):
            ep_num = ep + 1
            print(f"  [EP {ep_num}/{n_eps}]", end=" ", flush=True)

            if n_eps == 1:
                ctrl.reset()
            else:
                _randomize(model, data, ctrl, rng, bowl_bid, tp)

            traj_writer = (_TrajWriter(args.collect_dir, ep_num, two_egg=False)
                           if args.collect else None)
            phase, fail_str, nf, peak_grip, contact_max, contacts_end, jac_frob = \
                _run_episode(ctrl, model, data, renderer, writer, args,
                             ep_num, n_eps, successes, args.tier,
                             traj_writer=traj_writer)
            if traj_writer is not None:
                traj_writer.close()
            total_frames += nf

            ep_result = "SUCCESS" if phase == Phase.DONE else f"FAIL:{fail_str}"
            if phase == Phase.DONE:
                successes += 1

            rec = {
                "ep":          ep_num,
                "tier":        args.tier,
                "result":      ep_result,
                "peak_grip":   peak_grip,
                "contact_max": contact_max,
                "steps":       ctrl.step_count,
            }
            records.append(rec)
            if csv_writer:
                csv_writer.writerow([rec["ep"], rec["tier"], rec["result"],
                                      f"{rec['peak_grip']:.4f}", rec["contact_max"],
                                      rec["steps"]])

            print(f"{ep_result}  steps={ctrl.step_count}"
                  f"  peak_grip={peak_grip:.3f}N  frames={nf}")
            print(f"    JAC_EE ||J||_F={jac_frob:.4f}  contacts@end={len(contacts_end)}")
            for cn1, cn2, fn in contacts_end:
                print(f"      {cn1} <-> {cn2}  Fn={fn:.3f}N")

            if ep < n_eps - 1:
                card    = _inter_card(args.width, args.height,
                                      ep_num, n_eps, args.tier, fail_str, successes)
                inter_n = int(_HOLD_INTER * args.fps)
                for _ in range(inter_n):
                    writer.write(card)
                total_frames += inter_n

        if n_eps > 1:
            end   = _end_card(args.width, args.height, args.tier, records)
            end_n = int(_HOLD_END * args.fps)
            for _ in range(end_n):
                writer.write(end)
            total_frames += end_n

    # ── teardown (shared) ─────────────────────────────────────────────────────
    writer.release()
    renderer.close()
    if csv_fh:
        csv_fh.close()

    elapsed = time.time() - t0
    dur_s   = total_frames / args.fps
    pct     = (100 * successes // n_eps) if n_eps else 0
    print(f"\nDone — {successes}/{n_eps} success ({pct}%)")
    print(f"  total frames: {total_frames}  video: {dur_s:.1f}s"
          f"  wall_time: {elapsed:.1f}s")
    print(f"  Output: {os.path.abspath(args.out)}")
    if args.log:
        print(f"  CSV log: {os.path.abspath(args.log)}")


if __name__ == "__main__":
    main()
