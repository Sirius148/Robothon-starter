"""
Highlight reel for FOIB-Egg benchmark — ~75 s showcase.

Clips (all seed 42, dual-cam layout):
  SE medium  EP1  (713 steps, clean success)
  SE medium  EP6  (601 steps, fastest success)
  TE medium  EP8  (713 steps, 4.3 mm from bowl centre — best placement)
  TE medium  EP5  (810 steps, patient approach)
  TE extreme EP1  (700 steps, success at 66 mm sep)
  TE extreme EP5  (644 steps, DISTRACTOR_DISTURBED — elbow ceiling)

Usage:
    conda run -n robothon python video/record_highlight.py
    conda run -n robothon python video/record_highlight.py --out my_highlight.mp4
"""
import argparse
import os
import platform
import sys
import time
import types

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco

from controller.phase_controller import PhaseController, Phase
from video.overlay import draw

# ── re-import shared helpers from record_demo ────────────────────────────────
# We do a targeted import to avoid running record_demo.main().
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "record_demo",
    os.path.join(os.path.dirname(__file__), "record_demo.py"),
)
_rd = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rd)

_randomize         = _rd._randomize
_randomize_two_egg = _rd._randomize_two_egg
_composite_render  = _rd._composite_render
_draw_phase_sub    = _rd._draw_phase_sub
_centred_text      = _rd._centred_text
_scatter_shards    = _rd._scatter_shards
TIER_PARAMS        = _rd.TIER_PARAMS
TWO_EGG_TIER_PARAMS = _rd.TWO_EGG_TIER_PARAMS
MAX_STEPS          = _rd.MAX_STEPS
_N_SHARDS          = _rd._N_SHARDS

SCENE  = os.path.join(os.path.dirname(__file__), "../models/scene.xml")
W, H   = 640, 480
FPS    = 30
SEED   = 42

_HOLD_TITLE   = int(4.0 * FPS)   # frames — opening title
_HOLD_SECTION = int(2.0 * FPS)   # frames — section cards between tiers
_HOLD_TERM    = int(9.0 * FPS)   # frames — freeze on terminal frame (longer than benchmark)
_HOLD_END     = int(6.0 * FPS)   # frames — closing results card


# ── card generators ───────────────────────────────────────────────────────────

def _title_card():
    card = np.zeros((H, W, 3), dtype=np.uint8)
    cy = H // 2 - 100
    cy = _centred_text(card, "FOIB-Egg", cy, 1.10, (255, 200, 80), thickness=3)
    cy = _centred_text(card, "Fragile Object Integrity Benchmark", cy, 0.60, (200, 200, 200))
    cy += 10
    cy = _centred_text(card, "Pick egg from table  ·  place in bowl  ·  protect shell",
                       cy, 0.46, (150, 150, 150))
    cy = _centred_text(card, "Grip force < 12 N throughout  —  monitored every step",
                       cy, 0.42, (80, 200, 80))
    cy += 10
    cy = _centred_text(card, "seed=42   MuJoCo 3.9.0   3-finger gripper", cy, 0.40, (90, 90, 90))
    return card


def _section_card(title, subtitle=""):
    card = np.zeros((H, W, 3), dtype=np.uint8)
    # Thin accent bar
    cv2.rectangle(card, (0, H // 2 - 60), (W, H // 2 - 56), (255, 200, 80), -1)
    cy = H // 2 - 48
    cy = _centred_text(card, title,    cy, 0.80, (220, 220, 220), thickness=2)
    if subtitle:
        _centred_text(card, subtitle, cy, 0.50, (140, 140, 140))
    return card


def _end_card(clips):
    card = np.zeros((H, W, 3), dtype=np.uint8)
    cy = H // 2 - 148
    cy = _centred_text(card, "Benchmark Results", cy, 0.88, (255, 200, 80), thickness=2)
    cy += 6
    cv2.line(card, (W // 5, cy), (4 * W // 5, cy), (70, 70, 70), 1)
    cy += 14

    def _row(text, col, scale=0.54):
        nonlocal cy
        cy = _centred_text(card, text, cy, scale, col)

    _row("Single-egg  medium   10 / 10  (100 %)",  (80, 200, 80))
    _row("Single-egg  stress    9 / 10   (90 %)",  (80, 200, 80))
    _row("Single-egg  extreme   9 / 10   (90 %)",  (80, 200, 80))
    cy += 4
    _row("Two-egg     medium   10 / 10  (100 %)",  (80, 200, 80))
    _row("Two-egg     stress    9 / 10   (90 %)",  (80, 200, 80))
    _row("Two-egg     extreme   8 / 10   (80 %)",  (40, 180, 220))
    cy += 4
    _row("Dynamic disturbance  (rolling obstacle)",  (180, 140, 255))
    cy += 6
    cv2.line(card, (W // 5, cy), (4 * W // 5, cy), (70, 70, 70), 1)
    cy += 14
    _row("Peak grip (max)  0.54 N     shell limit: 12 N", (160, 160, 160), 0.46)
    _row("Grasp quality (force compliance)  0.956",        (160, 160, 160), 0.46)
    _row("Shell integrity: physical shard scatter on OVER-SQUEEZED", (140, 140, 140), 0.42)
    return card


# ── shell fragility demo clip ─────────────────────────────────────────────────

def _run_clip_shell_demo(ctrl, model, data, renderer, renderer_top, writer, args_ns,
                          shard_qpos, shard_dof, shard_rng, label, tier):
    """Run to LIFT phase (weld fires), then physically scatter shell shards."""
    render_every = max(1, int(round(1.0 / (FPS * model.opt.timestep))))
    phase = Phase.IDLE
    weld_step = -1

    for step_i in range(MAX_STEPS):
        phase, fail = ctrl.step()
        mujoco.mj_step(model, data)

        if ctrl._weld_active and weld_step < 0:
            weld_step = step_i
        # Hold 0.18 s after weld fires so it's clearly visible, then break
        if weld_step >= 0 and step_i > weld_step + 90:
            break

        if step_i % render_every == 0:
            rgb  = _composite_render(renderer, renderer_top, data, args_ns)
            info = ctrl.overlay_info()
            info.update({"episode_num": 1, "n_episodes": 1,
                         "ep_successes": 0, "tier": tier})
            writer.write(draw(rgb, info)[:, :, ::-1])

        if phase in (Phase.DONE, Phase.FAIL):
            break

    # Scatter shards from egg position
    egg_pos = data.xpos[ctrl.egg_id].copy()
    if shard_qpos:
        _scatter_shards(data, egg_pos, shard_qpos, shard_dof, shard_rng)
    # Release kinematic weld: egg drops alongside shards
    ctrl._weld_active = False
    data.qvel[0:3] = [0.0, 0.0, -0.15]

    # Animate for 2.2 s — shards and egg fall/bounce on table
    for shard_step in range(int(2.2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if shard_step % render_every == 0:
            rgb  = _composite_render(renderer, renderer_top, data, args_ns)
            info = ctrl.overlay_info()
            info["fail_reason"] = "OVER-SQUEEZED"   # drives SHELL → red in HUD
            info.update({"episode_num": 1, "n_episodes": 1,
                         "ep_successes": 0, "tier": tier})
            bgr = draw(rgb, info)[:, :, ::-1]
            h, w = bgr.shape[:2]
            txt = "SHELL  CRACKED"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
            cv2.putText(bgr, txt, ((w - tw) // 2, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (40, 40, 220), 2, cv2.LINE_AA)
            lh = 26
            cv2.rectangle(bgr, (0, h - lh), (w, h), (15, 15, 15), -1)
            cv2.putText(bgr, f"{label}   FAIL: OVER-SQUEEZED [demo]",
                        (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 220), 1, cv2.LINE_AA)
            writer.write(bgr)

    return Phase.FAIL, "OVER-SQUEEZED"


# ── episode runner ────────────────────────────────────────────────────────────

def _run_clip(ctrl, model, data, renderer, renderer_top, writer, args_ns,
              label, tier, two_egg=False,
              egg2_bid=None, egg2_geom_ids=None, finger_geom_ids=None,
              distractor_init_xy=None):
    """Run one episode; write frames to `writer`.

    Returns the terminal BGRframe (for the label overlay).
    """
    render_every = max(1, int(round(1.0 / (FPS * model.opt.timestep))))
    phase = Phase.IDLE
    fail  = None
    wrong_object_contact = False
    _prev_pname = None
    _sub_cd = 0
    _sub_frames = max(1, int(1.5 * FPS))

    for step_i in range(MAX_STEPS):
        phase, fail = ctrl.step()
        mujoco.mj_step(model, data)

        if two_egg and not wrong_object_contact and egg2_geom_ids and finger_geom_ids:
            for ci in range(data.ncon):
                g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
                if ((g1 in egg2_geom_ids and g2 in finger_geom_ids) or
                        (g2 in egg2_geom_ids and g1 in finger_geom_ids)):
                    wrong_object_contact = True
                    break

        is_terminal = phase in (Phase.DONE, Phase.FAIL)

        if step_i % render_every == 0:
            rgb  = _composite_render(renderer, renderer_top, data, args_ns)
            info = ctrl.overlay_info()
            info.update({"episode_num": 1, "n_episodes": 1,
                         "ep_successes": 0, "tier": tier})
            if two_egg and distractor_init_xy is not None:
                egg2_pos = data.xpos[egg2_bid]
                info["distractor_disp"] = float(np.linalg.norm(
                    egg2_pos[:2] - distractor_init_xy))
                info["egg_sep"] = float(np.linalg.norm(
                    egg2_pos[:2] - data.xpos[ctrl.egg_id, :2]))
            bgr = draw(rgb, info)[:, :, ::-1]
            if phase.name != _prev_pname:
                _prev_pname = phase.name
                _sub_cd = _sub_frames
            if _sub_cd > 0 and not is_terminal:
                _draw_phase_sub(bgr, phase.name)
                _sub_cd -= 1
            writer.write(bgr)

        if is_terminal:
            rgb  = _composite_render(renderer, renderer_top, data, args_ns)
            info = ctrl.overlay_info()
            info.update({"episode_num": 1, "n_episodes": 1,
                         "ep_successes": int(phase == Phase.DONE), "tier": tier})
            if two_egg and distractor_init_xy is not None:
                egg2_pos = data.xpos[egg2_bid]
                info["distractor_disp"] = float(np.linalg.norm(
                    egg2_pos[:2] - distractor_init_xy))
                info["egg_sep"] = float(np.linalg.norm(
                    egg2_pos[:2] - data.xpos[ctrl.egg_id, :2]))
            hold_bgr = draw(rgb, info)[:, :, ::-1]
            # Bottom label strip
            result_str = "SUCCESS" if phase == Phase.DONE else f"FAIL: {fail.value}"
            result_col = (80, 200, 80) if phase == Phase.DONE else (60, 60, 220)
            lh, lw = 28, W
            cv2.rectangle(hold_bgr, (0, H - lh), (lw, H), (15, 15, 15), -1)
            label_full = f"{label}   {result_str}   steps={ctrl.step_count}"
            (tw, _), _ = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.putText(hold_bgr, label_full,
                        ((W - tw) // 2, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, result_col, 1, cv2.LINE_AA)
            for _ in range(_HOLD_TERM):
                writer.write(hold_bgr)
            break

    fail_str = fail.value if fail is not None else ""
    return phase, fail_str


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="demo_highlight.mp4")
    args = parser.parse_args()

    model    = mujoco.MjModel.from_xml_path(SCENE)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    h_bot    = H - H * 2 // 3
    renderer_top = mujoco.Renderer(model, height=h_bot, width=W)
    ctrl     = PhaseController(model, data)
    bowl_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl")

    # Two-egg IDs
    egg2_bid      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "egg2")
    egg2_jnt      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "egg2_free")
    egg2_qpos_adr = model.jnt_qposadr[egg2_jnt]
    egg2_dof_adr  = model.jnt_dofadr[egg2_jnt]
    egg2_geom_ids = frozenset(
        i for i in range(model.ngeom) if model.geom_bodyid[i] == egg2_bid
    )
    fa_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_a")
    finger_geom_ids = frozenset(
        i for i in range(model.ngeom) if model.geom_bodyid[i] == fa_bid
    )

    fourcc = cv2.VideoWriter_fourcc(*("avc1" if platform.system() == "Darwin" else "mp4v"))
    writer = cv2.VideoWriter(args.out, fourcc, FPS, (W, H))
    if not writer.isOpened():
        sys.exit(f"Cannot open video writer: {args.out}")

    # Shard body addresses for shell-cracking demo
    shard_qpos, shard_dof = [], []
    for _i in range(_N_SHARDS):
        _j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"shard{_i}_free")
        if _j >= 0:
            shard_qpos.append(model.jnt_qposadr[_j])
            shard_dof.append(model.jnt_dofadr[_j])
    shard_rng = np.random.default_rng(SEED + 1000)

    # Shared args namespace — gripper_cam as secondary for immersive fingertip view
    args_ns = types.SimpleNamespace(camera="side_cam", bottom_cam="gripper_cam",
                                     width=W, height=H)

    def _write_card(card_bgr, n_frames):
        for _ in range(n_frames):
            writer.write(card_bgr)

    render_every = max(1, int(round(1.0 / (FPS * model.opt.timestep))))
    t0 = time.time()
    total_frames = 0

    print(f"Recording highlight reel → {args.out}")

    # ── 1. Opening title ─────────────────────────────────────────────────────
    _write_card(_title_card(), _HOLD_TITLE)
    total_frames += _HOLD_TITLE

    # ── 2. Single-Egg Medium ─────────────────────────────────────────────────
    sec = _section_card("Single-Egg — Medium Tier",
                        "Pick & place with no distractor   (seed 42)")
    _write_card(sec, _HOLD_SECTION)
    total_frames += _HOLD_SECTION

    tp_se = TIER_PARAMS["medium"]

    print("  [SE medium EP1]", end=" ", flush=True)
    rng = np.random.default_rng(SEED)
    _randomize(model, data, ctrl, rng, bowl_bid, tp_se)          # EP1
    phase, fstr = _run_clip(ctrl, model, data, renderer, renderer_top,
                             writer, args_ns,
                             label="SE  medium  EP1", tier="medium")
    total_frames += ctrl.step_count // max(1, int(round(1.0 / (FPS * model.opt.timestep)))) + _HOLD_TERM
    print(f"{phase.name}  steps={ctrl.step_count}")

    print("  [SE medium EP6]", end=" ", flush=True)
    for _ in range(5):                                            # advance EP2→6
        _randomize(model, data, ctrl, rng, bowl_bid, tp_se)
    phase, fstr = _run_clip(ctrl, model, data, renderer, renderer_top,
                             writer, args_ns,
                             label="SE  medium  EP6", tier="medium")
    total_frames += ctrl.step_count // max(1, int(round(1.0 / (FPS * model.opt.timestep)))) + _HOLD_TERM
    print(f"{phase.name}  steps={ctrl.step_count}")

    # ── 3. Two-Egg Medium ────────────────────────────────────────────────────
    sec = _section_card("Two-Egg Challenge — Medium Tier",
                        "Distractor at 75 mm separation  (seed 42)")
    _write_card(sec, _HOLD_SECTION)
    total_frames += _HOLD_SECTION

    tp_te = TWO_EGG_TIER_PARAMS["medium"]

    print("  [TE medium EP8 — best placement]", end=" ", flush=True)
    rng = np.random.default_rng(SEED)
    dist_init_xy = None
    for ep in range(8):
        dist_init_xy, _, _ = _randomize_two_egg(
            model, data, ctrl, rng, bowl_bid,
            egg2_bid, egg2_qpos_adr, egg2_dof_adr, tp_te)
    phase, fstr = _run_clip(ctrl, model, data, renderer, renderer_top,
                             writer, args_ns,
                             label="TE  medium  EP8  (dist_to_bowl 4 mm)", tier="medium",
                             two_egg=True, egg2_bid=egg2_bid,
                             egg2_geom_ids=egg2_geom_ids,
                             finger_geom_ids=finger_geom_ids,
                             distractor_init_xy=dist_init_xy)
    total_frames += ctrl.step_count // max(1, int(round(1.0 / (FPS * model.opt.timestep)))) + _HOLD_TERM
    print(f"{phase.name}  steps={ctrl.step_count}")

    print("  [TE medium EP5 — longest approach]", end=" ", flush=True)
    rng = np.random.default_rng(SEED)
    for ep in range(5):
        dist_init_xy, _, _ = _randomize_two_egg(
            model, data, ctrl, rng, bowl_bid,
            egg2_bid, egg2_qpos_adr, egg2_dof_adr, tp_te)
    phase, fstr = _run_clip(ctrl, model, data, renderer, renderer_top,
                             writer, args_ns,
                             label="TE  medium  EP5", tier="medium",
                             two_egg=True, egg2_bid=egg2_bid,
                             egg2_geom_ids=egg2_geom_ids,
                             finger_geom_ids=finger_geom_ids,
                             distractor_init_xy=dist_init_xy)
    total_frames += ctrl.step_count // max(1, int(round(1.0 / (FPS * model.opt.timestep)))) + _HOLD_TERM
    print(f"{phase.name}  steps={ctrl.step_count}")

    # ── 4. Two-Egg Extreme ───────────────────────────────────────────────────
    sec = _section_card("Two-Egg Extreme — 66 mm Separation",
                        "Elbow clearance < 15 mm at tight configs")
    _write_card(sec, _HOLD_SECTION)
    total_frames += _HOLD_SECTION

    tp_tx = TWO_EGG_TIER_PARAMS["extreme"]

    print("  [TE extreme EP1 — success]", end=" ", flush=True)
    rng = np.random.default_rng(SEED)
    dist_init_xy, _, _ = _randomize_two_egg(
        model, data, ctrl, rng, bowl_bid,
        egg2_bid, egg2_qpos_adr, egg2_dof_adr, tp_tx)
    phase, fstr = _run_clip(ctrl, model, data, renderer, renderer_top,
                             writer, args_ns,
                             label="TE  extreme  EP1  (66 mm sep)", tier="extreme",
                             two_egg=True, egg2_bid=egg2_bid,
                             egg2_geom_ids=egg2_geom_ids,
                             finger_geom_ids=finger_geom_ids,
                             distractor_init_xy=dist_init_xy)
    total_frames += ctrl.step_count // max(1, int(round(1.0 / (FPS * model.opt.timestep)))) + _HOLD_TERM
    print(f"{phase.name}  steps={ctrl.step_count}")

    print("  [TE extreme EP5 — elbow contact]", end=" ", flush=True)
    rng = np.random.default_rng(SEED)
    for ep in range(5):
        dist_init_xy, _, _ = _randomize_two_egg(
            model, data, ctrl, rng, bowl_bid,
            egg2_bid, egg2_qpos_adr, egg2_dof_adr, tp_tx)
    phase, fstr = _run_clip(ctrl, model, data, renderer, renderer_top,
                             writer, args_ns,
                             label="TE  extreme  EP5  (elbow ceiling)", tier="extreme",
                             two_egg=True, egg2_bid=egg2_bid,
                             egg2_geom_ids=egg2_geom_ids,
                             finger_geom_ids=finger_geom_ids,
                             distractor_init_xy=dist_init_xy)
    total_frames += ctrl.step_count // max(1, int(round(1.0 / (FPS * model.opt.timestep)))) + _HOLD_TERM
    print(f"{phase.name}  steps={ctrl.step_count}")

    # ── 5. Shell fragility demo ──────────────────────────────────────────────
    sec = _section_card("Shell Integrity Demo",
                        "Force over-squeeze → physical shell cracking")
    _write_card(sec, _HOLD_SECTION)
    total_frames += _HOLD_SECTION

    print("  [Shell fragility demo]", end=" ", flush=True)
    rng = np.random.default_rng(SEED)
    _randomize(model, data, ctrl, rng, bowl_bid, TIER_PARAMS["medium"])
    phase, fstr = _run_clip_shell_demo(
        ctrl, model, data, renderer, renderer_top, writer, args_ns,
        shard_qpos, shard_dof, shard_rng,
        label="SE  medium  EP1  [fragility demo]", tier="medium",
    )
    total_frames += ctrl.step_count // render_every + int(2.2 * FPS)
    print(f"{phase.name} → {fstr}")

    # ── 6. Dynamic disturbance ───────────────────────────────────────────────
    sec = _section_card("Dynamic Disturbance",
                        "Distractor rolling — robot tracks target only  (seed 42)")
    _write_card(sec, _HOLD_SECTION)
    total_frames += _HOLD_SECTION

    tp_dyn = TWO_EGG_TIER_PARAMS["medium"]
    print("  [TE dynamic EP1]", end=" ", flush=True)
    rng = np.random.default_rng(SEED)
    dist_init_xy, _, _ = _randomize_two_egg(
        model, data, ctrl, rng, bowl_bid,
        egg2_bid, egg2_qpos_adr, egg2_dof_adr, tp_dyn, dynamic=True)
    phase, fstr = _run_clip(ctrl, model, data, renderer, renderer_top,
                             writer, args_ns,
                             label="TE  dynamic  EP1  (rolling distractor)", tier="medium",
                             two_egg=True, egg2_bid=egg2_bid,
                             egg2_geom_ids=egg2_geom_ids,
                             finger_geom_ids=finger_geom_ids,
                             distractor_init_xy=dist_init_xy)
    total_frames += ctrl.step_count // render_every + _HOLD_TERM
    print(f"{phase.name}  steps={ctrl.step_count}")

    # ── 7. End card ──────────────────────────────────────────────────────────
    _write_card(_end_card([]), _HOLD_END)
    total_frames += _HOLD_END

    writer.release()
    renderer.close()
    renderer_top.close()

    elapsed = time.time() - t0
    dur_s   = total_frames / FPS
    print(f"\nDone — {dur_s:.1f}s video  ({total_frames} frames)  wall={elapsed:.1f}s")
    print(f"  Output: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
