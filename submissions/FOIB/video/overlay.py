"""
Overlay renderer for the FOIB-Egg benchmark.

Left-strip HUD (top → bottom):
  EP N/M  [TIER]  SCORE K   — gold header (multi-episode + tier)
  PHASE  : …
  STEP   : …
  OBS    : Z=…  D=…m        — egg height and distance to bowl
  SHELL  : …                 — primary benchmark indicator
  GRIP   : … N  [HELD]
  STATUS : …

Bottom-right corner: gate checklist [GRASP / LIFT / PLACE / SHELL].
"""
import cv2
import numpy as np


_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.60
_THICKNESS  = 2
_LINE_H     = 28
_MARGIN     = 12

# BGR palette
_COLOR_OK   = (80,  200,  80)   # green
_COLOR_FAIL = (60,   60, 220)   # red
_COLOR_WARN = (40,  180, 220)   # amber-orange
_COLOR_INFO = (220, 220, 220)   # white-ish
_COLOR_GRAY = (110, 110, 110)   # inactive / not-yet
_COLOR_EP   = (255, 200,  80)   # gold — episode / tier header
_COLOR_OBS  = (180, 210, 255)   # pale blue — observation values

# Phases where shell monitoring is not yet active
_PRE_GRASP_PHASES = {"IDLE", "RETRACT", "APPROACH"}


def _shell_state(phase: str, fail_reason: str, grasped: bool) -> tuple:
    """Return (label, color_BGR) for the SHELL integrity badge."""
    if fail_reason == "OVER-SQUEEZED":
        return "OVER-SQUEEZED", _COLOR_FAIL
    if fail_reason in ("DROPPED", "TIMEOUT"):
        return fail_reason, _COLOR_WARN
    if phase == "DONE":
        return "INTACT", _COLOR_OK
    if phase in _PRE_GRASP_PHASES:
        return "--", _COLOR_GRAY
    if phase == "GRASP" and not grasped:
        return "CLOSING", _COLOR_WARN   # fingers moving in, monitoring begins
    return "OK", _COLOR_OK              # held and force within safe limits


def draw(frame: np.ndarray, info: dict) -> np.ndarray:
    """
    Draw benchmark HUD on an RGB H×W×3 uint8 frame.

    Required info keys : phase, step, fail_reason, grip_force, grasped, gates,
                         egg_z, dist_bowl
    Optional info keys : episode_num, n_episodes, ep_successes, tier
    Returns a new frame (does not modify in-place).
    """
    bgr = frame[:, :, ::-1].copy()

    phase        = info.get("phase",       "?")
    step         = info.get("step",        0)
    fail_reason  = info.get("fail_reason", "")
    grip_force   = info.get("grip_force",  0.0)
    grasped      = info.get("grasped",     False)
    gates        = info.get("gates",       {})
    egg_z         = info.get("egg_z",         None)
    dist_bowl     = info.get("dist_bowl",    None)
    peak_grip        = info.get("peak_grip",        None)
    contact_count    = info.get("contact_count",    None)
    distractor_disp  = info.get("distractor_disp",  None)
    egg_sep          = info.get("egg_sep",           None)
    episode_num      = info.get("episode_num",       None)
    n_episodes    = info.get("n_episodes",   None)
    ep_successes  = info.get("ep_successes", None)
    tier          = info.get("tier",         None)
    finger_touch  = info.get("finger_touch",  None)
    grip_b_force  = info.get("grip_b_force",  None)
    grip_c_force  = info.get("grip_c_force",  None)
    grasp_quality = info.get("grasp_quality", None)
    object_type   = info.get("object_type",   "egg")

    shell_label, shell_color = _shell_state(phase, fail_reason, grasped)

    if fail_reason:
        status_str, status_color = "FAIL",    _COLOR_FAIL
    elif phase == "DONE":
        status_str, status_color = "SUCCESS", _COLOR_OK
    else:
        status_str, status_color = "RUNNING", _COLOR_INFO

    # ── build line list ───────────────────────────────────────────────────────
    lines = []

    if episode_num is not None:
        tier_tag  = f"  [{tier.upper()}]" if tier else ""
        score_str = f"  SCORE {ep_successes}" if ep_successes is not None else ""
        obj_tag   = f"  {object_type.upper()}" if object_type not in ("egg", None) else ""
        lines.append((f"EP {episode_num}/{n_episodes}{tier_tag}{score_str}{obj_tag}",
                      _COLOR_EP))

    lines.append((f"PHASE  : {phase}", _COLOR_INFO))
    lines.append((f"STEP   : {step}",  _COLOR_INFO))

    # observation line — egg height and horizontal distance to bowl
    if egg_z is not None and dist_bowl is not None:
        lines.append((f"OBS    : Z={egg_z:.3f}m  D={dist_bowl:.3f}m", _COLOR_OBS))

    # peak grip and contact count (only when controller provides them)
    if peak_grip is not None or contact_count is not None:
        pk_str  = f"PK={peak_grip:.3f}N" if peak_grip is not None else ""
        cts_str = f"CTS={contact_count}" if contact_count is not None else ""
        sep     = "  " if pk_str and cts_str else ""
        lines.append((f"MEAS   : {pk_str}{sep}{cts_str}", _COLOR_OBS))

    # two-egg distractor metrics (only when provided)
    if distractor_disp is not None or egg_sep is not None:
        sep_str  = f"SEP={int(egg_sep * 1000)}mm"          if egg_sep         is not None else ""
        disp_str = f"DISP={int(distractor_disp * 1000)}mm" if distractor_disp is not None else ""
        sep2     = "  " if sep_str and disp_str else ""
        lines.append((f"DIST2  : {sep_str}{sep2}{disp_str}", _COLOR_OBS))

    lines.append((f"SHELL  : {shell_label}", shell_color))
    lines.append((f"GRIP   : {grip_force:.2f} N  {'[HELD]' if grasped else '      '}",
                  _COLOR_OK if grasped else _COLOR_INFO))
    if grip_b_force is not None and grip_c_force is not None:
        # Per-finger effort: B/C close independently; symmetry if |B-C|<0.5N
        sym_ok = abs(grip_b_force - grip_c_force) < 0.5
        frc_col = _COLOR_OK if sym_ok else _COLOR_WARN
        lines.append((f"FGRIP  : B={grip_b_force:.2f}N C={grip_c_force:.2f}N",
                      frc_col))

    if finger_touch is not None:
        ta, tb, tc = finger_touch
        n_active = sum(v > 0.05 for v in (ta, tb, tc))
        if grasped:
            # Post-weld: all 3 fingers should fire → green if 3/3, warn if 1-2/3
            touch_col = _COLOR_OK if n_active == 3 else (_COLOR_WARN if n_active > 0 else _COLOR_FAIL)
        else:
            touch_col = _COLOR_OK if ta > 0.05 else _COLOR_GRAY
        lines.append((f"TOUCH  : A={ta:.2f} B={tb:.2f} C={tc:.2f}", touch_col))
        if grasped and n_active > 0:
            sym = n_active / 3.0
            sym_col = _COLOR_OK if sym >= 1.0 else _COLOR_WARN
            lines.append((f"GSYM   : {sym:.2f}  ({n_active}/3 fingers)", sym_col))
    if grasp_quality is not None:
        gq_col = _COLOR_OK if grasp_quality > 0.8 else _COLOR_WARN
        lines.append((f"GQUAL  : {grasp_quality:.3f}", gq_col))

    lines.append((f"STATUS : {status_str}", status_color))

    # ── semi-transparent background strip ────────────────────────────────────
    strip_h = _MARGIN + len(lines) * _LINE_H + _MARGIN
    strip_w = 400   # wide enough for FGRIP line
    overlay_bg = bgr.copy()
    cv2.rectangle(overlay_bg, (0, 0), (strip_w, strip_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay_bg, 0.55, bgr, 0.45, 0, bgr)

    y = _MARGIN + _LINE_H - 6
    for text, col in lines:
        cv2.putText(bgr, text, (_MARGIN, y), _FONT, _FONT_SCALE, col,
                    _THICKNESS, cv2.LINE_AA)
        y += _LINE_H

    # ── gate checklist bottom-right (SAFE → SHELL) ───────────────────────────
    h, w = bgr.shape[:2]
    gx, gy = w - 160, h - 10
    gate_labels = {"SAFE": "SHELL", "GRASP": "GRASP",
                   "LIFT": "LIFT",  "PLACE": "PLACE"}
    for name, ok in gates.items():
        label = gate_labels.get(name, name)
        sym   = "+" if ok else "-"
        gcol  = _COLOR_OK if ok else _COLOR_FAIL
        cv2.putText(bgr, f"[{sym}] {label}", (gx, gy), _FONT, 0.48, gcol, 1,
                    cv2.LINE_AA)
        gy -= 22

    return bgr[:, :, ::-1]
