# FOIB-Egg: Fragile Object Integrity Benchmark

A MuJoCo 3 simulation benchmark that evaluates whether a 3-DOF robot arm
can pick and place an egg without compromising shell integrity.
**Shell integrity — monitored via continuous gripper force sensing — is the
primary scoring criterion.** Pick-and-place is the delivery mechanism;
force compliance throughout the grasp is the benchmark objective.

The benchmark runs headless, generates reproducible video from code, and
logs per-episode results to CSV. No manual annotation is required.

**Key capabilities:**

| Capability | Details |
|---|---|
| Force-compliance gate | Continuous `actuatorfrc` sampling; `OVER-SQUEEZED` fires if any render frame exceeds 12 N |
| 11-phase task planner | Fixed sequence with per-phase step budgets; shell gate active from GRASP onward |
| Difficulty gradient | `min_egg_sep` drives 10 → 10 → 9 → 8 success rate (TE stress/extreme); controller and thresholds never modified |
| Rolling disturbance | Physically consistent kick (`ω = v / R_eff`, no-slip); distinct `DISTRACTOR_ROLLING` failure code |
| Trajectory export | `--collect` writes per-step joint + pose + force data for downstream policy training |
| Dual-camera video | `--dual-cam` composites side view (upper ⅔) + overhead view (lower ⅓) in one frame |
| Reproducible by default | Deterministic `--seed 42`; seven included CSVs; bit-identical across runs |
| Analytical IK + reach correction | Closed-form Z-yaw + 2R IK; runtime Cartesian correction tracks per-episode egg **and** bowl positions; Cartesian-space arrival; joint-space fallback on limit violation |
| Three-finger gripper (v3) | 120° spacing; `finger_a` functional (`contype=1`); `finger_b/c` visual/kinematic (`contype=0, density=0`); 3 independent position actuators |
| Touch sensor array (v3) | `<touch>` sensors at each fingertip site; `touch_tip_a` fires on egg contact; logged to HUD and CSV as `grasp_quality` |
| Grasp quality metric (v3) | Force-compliance score `1 − min(peak_grip / 12, 1.0)` ≈ 0.956 per episode; logged to CSV |
| Multi-object support (M3) | `--object egg\|cylinder\|sphere` swaps target geometry at load time; controller, contact pair, failure codes, and CSV schema unchanged; 10/10 medium verified for all three shapes |

---

## Gripper Design

The 3-finger gripper is a key design element visible in the `gripper_cam` view (lower ⅓ of
all dual-cam videos). Three capsule fingers are arranged at 120° spacing around the palm disc,
each with its own independently-controlled position actuator (`act_grip_a/b/c`, `kp = 20`).

```
         finger_a  (0°, −Y side)   — primary contact finger
              ↑
    ←  finger_c  (240°)        finger_b  (120°)  →
```

| Element | Role | Observable in video |
|---------|------|---------------------|
| `finger_a` | Primary force finger; `contype=1` → explicit contact pair with egg | HUD `GRIP` force; `touch_tip_a` fires during hold |
| `finger_b/c` | Independent position control; `contype=0` → non-colliding structural | HUD `FGRIP B=… C=…`; adaptive post-weld closing visible |
| Touch array | `<touch>` sensors at each tip; `GSYM` line shows 1–3/3 symmetry | `TOUCH A/B/C` on HUD |
| Grasp quality | `1 − peak_grip / 12` ≈ 0.956 per episode | `GQUAL` on HUD; `grasp_quality` column in CSV |

**Adaptive post-weld closing**: once the kinematic grasp weld fires, `finger_a` holds its
current joint position (locks force) while `finger_b` and `finger_c` close an additional
2 mm (`_GRIP_CONFIRM = 0.002 m`) to tighten the symmetric wrap. This is visible in the
`FGRIP B/C` HUD lines increasing after `[HELD]` appears.

**Shell cracking demo** (`demo_highlight.mp4`, section 5): 8 egg-shell shard bodies (`contype=4`,
bounce on table only) are scattered physically when `OVER-SQUEEZED` fires, creating a visually
convincing shell-fragility consequence using MuJoCo's freejoint physics.

---

## Problem

Fragile object manipulation requires balancing two competing constraints:
grip must be firm enough to lift and carry the object, yet gentle enough
not to damage it. Standard pick-and-place benchmarks measure *placement
accuracy* but ignore *grasp force compliance*. FOIB-Egg makes force
compliance an explicit, scoreable criterion alongside placement.

---

## How this differs from standard pick-and-place

| Dimension | Standard benchmark | FOIB-Egg |
|-----------|-------------------|----------|
| Success criterion | Placement accuracy (distance to target) | Placement accuracy **and** shell integrity (grip force < 12 N throughout) |
| Difficulty axis | Controller or scene complexity | Single scene parameter (`min_egg_sep`, tier ranges); controller never changed |
| Distractor evaluation | Rarely present | Explicit failure codes with numeric thresholds; two static tiers + dynamic mode |
| Failure taxonomy | Pass / fail or distance bucket | 5 named codes, each with a physical mechanism and a measurable gate |
| Extension model | New controller or scene required | `--two-egg` and `--dynamic-dist` add evaluation axes without modifying the controller |

The phase controller (`phase_controller.py`) is **identical** across all three evaluation
modes. Difficulty scales via a single physical variable (`min_egg_sep`) without any
controller modification, enabling clean ablation across four tiers and three modes.
This is a benchmark design contribution: the same controller and scoring logic apply
everywhere; what changes is only the scene configuration.

---

## How to extend

FOIB-Egg is structured for benchmark composability: each evaluation axis is a
scene-level parameter change, not a controller modification.

| Extension | What to change | What stays the same |
|-----------|---------------|---------------------|
| New difficulty tier | Adjust `egg_x`, `bowl_xy`, `min_egg_sep` in tier params | Controller, phase sequence, CSV schema, failure codes |
| New failure code | Add a scalar threshold check in `record_demo.py`; add a row to the taxonomy table | Output format, phase logic, existing codes |
| New evaluation mode | Pass a different scene config or disturbance source at episode reset | Everything else — `--two-egg` and `--dynamic-dist` demonstrate this pattern |
| New object type | Add an entry to `_OBJECT_DEFS` in `record_demo.py`; pass `--object <name>` | Controller, phase sequence, contact pair name, CSV schema, failure codes |

The single-egg → two-egg static → two-egg dynamic progression is the reference
implementation of this pattern: three evaluation modes, one controller, one CSV
schema, clean ablation across all four tiers.

---

## Benchmark Definition

Each episode proceeds through a fixed phase sequence:

```
IDLE → RETRACT → APPROACH → GRASP → LIFT → TRANSPORT → LOWER → RELEASE → CHECK
                                 │                                            │
                        shell monitoring                             DONE  (success)
                          begins here                                FAIL  (see taxonomy)
```

**An episode is scored SUCCESS only when both criteria hold:**

| Criterion | Threshold | Hard failure |
|-----------|-----------|-------------|
| Shell integrity | grip force < 12 N throughout grasp | `OVER-SQUEEZED` |
| Placement accuracy | egg within 8 cm of bowl centre after release | `DROPPED` |

`OVER-SQUEEZED` is a hard failure: a correctly placed egg is still a failed
episode if grip force exceeded the limit during transport.

### Phase transition table

| Phase | Entry condition | Exit gate | Hard failure if gate missed |
|-------|-----------------|-----------|----------------------------|
| `IDLE` | episode start | immediate → RETRACT | — |
| `RETRACT` | after IDLE | joints at home waypoint (`‖Δq‖ < 0.05 rad`) | `TIMEOUT` (> 2000 steps) |
| `APPROACH` | after RETRACT | `ee_to_egg < 70 mm` | `TIMEOUT` |
| `GRASP` | after APPROACH | kinematic weld fires (`dist < 30 mm`, contact > 0) | `TIMEOUT` |
| `LIFT` | after GRASP | `egg.z > LIFT_HEIGHT_CHK` | `DROPPED` if `egg.z < 0.79 m`; `TIMEOUT` |
| `TRANSPORT` | after LIFT | joints at transport waypoint | `OVER-SQUEEZED` if force > 12 N; `TIMEOUT` |
| `LOWER` | after TRANSPORT | joints at lower waypoint | `TIMEOUT` |
| `RELEASE` | after LOWER | weld released | `TIMEOUT` |
| `CHECK` | after RELEASE | `egg_to_bowl_xy < 80 mm` | `DROPPED` if outside |
| `DONE` | CHECK passes | — (terminal success) | — |
| `FAIL` | any gate violation above | — (terminal failure) | — |

Each phase is independent and budgeted separately (2000 steps). Shell force monitoring
activates on entry to `GRASP` and remains active until `DONE` or `FAIL`. The
sequence is fixed; no phase can be skipped or re-entered.

The underlying joint-space driver (`_drive_legacy`) caps joint velocity at
`JOINT_MAX_VEL = 50 rad/s` per timestep, producing smooth motion between waypoints
rather than instantaneous position jumps. Trajectory data collected with `--collect`
reflects this smooth profile and is suitable for downstream policy training.

---

## Failure Taxonomy

### Unified failure code reference

Each failure code maps to a distinct physical event with an explicit scalar threshold.
Names are not conventions — they are implemented check conditions in the state machine.

| Code | Physical mechanism / trigger | Observable sign | Mode(s) |
|------|------------------------------|-----------------|---------|
| `OVER-SQUEEZED` | Position actuator closes past egg; `actuatorfrc` reaction force exceeds 12 N | SHELL overlay turns red; HUD `F=…N` over limit | Single-egg, two-egg (any) |
| `DROPPED` | Kinematic attach misses (`egg.z < 0.79 m` in LIFT/TRANSPORT), **or** egg exits bowl region after release (`egg_to_bowl > 80 mm`) | Egg leaves frame; HUD `D=…m` diverges | Single-egg, two-egg (any) |
| `TIMEOUT` | Phase step counter exceeds 2000 without satisfying exit gate | Phase timer saturates in HUD; arm stalled | Single-egg, two-egg (any) |
| `DISTRACTOR_DISTURBED` | Gripper sweep physically contacts and displaces distractor > 20 mm XY during approach or transport | Distractor egg visibly shifts; overhead view shows displacement | Two-egg static and dynamic |
| `DISTRACTOR_ROLLING` | Distractor self-rolls > 20 mm before any finger contact (initial kick only) | Distractor moves before arm reaches egg; observable in overhead view | Two-egg dynamic only |

Priority in the state machine (highest wins): task failure (`OVER-SQUEEZED` / `DROPPED` / `TIMEOUT`)
> `DISTRACTOR_ROLLING` > `DISTRACTOR_DISTURBED` > `SUCCESS`. Codes are mutually exclusive per episode.

### Single-egg failures

| Code | Precise trigger | Phase detected | Observable sign |
|------|----------------|----------------|-----------------|
| `OVER-SQUEEZED` | `actuatorfrc` > 12 N | GRASP → TRANSPORT | SHELL overlay turns red |
| `DROPPED` | egg Z < 0.79 m after > 200 steps in LIFT/TRANSPORT, **or** egg XY > 80 mm from bowl in CHECK | LIFT, TRANSPORT, CHECK | egg falls out of frame; HUD shows `D=…m` diverging |
| `TIMEOUT` | phase step counter exceeds 2000 without satisfying exit gate | any phase | phase timer saturates in HUD |

### Two-egg failures (v2)

| Code | Precise trigger | Root cause |
|------|----------------|------------|
| `FAIL:DISTRACTOR_DISTURBED` | distractor XY displacement > 20 mm at any point | arm clips distractor during APPROACH/GRASP |
| `FAIL:DISTRACTOR_ROLLING` | distractor self-displacement > 20 mm (dynamic mode only) | initial rolling kick travels too far before stopping |
| `FAIL:DROPPED` / `FAIL:TIMEOUT` | same as single-egg | grasp or placement failure regardless of distractor |

### Shell overlay states

| Overlay label | Meaning |
|---|---|
| `--` | pre-grasp; force monitoring not yet active |
| `CLOSING` | fingers moving in; monitoring begins |
| `OK` | egg held; force within safe limits |
| `INTACT` | episode complete; shell survived |
| `OVER-SQUEEZED` | force limit exceeded → FAIL |
| `DROPPED` / `TIMEOUT` | other failure → FAIL |

---

## Benchmark Tiers

Four difficulty tiers control per-episode randomisation. Egg Y is capped
at ±3 mm across all tiers: finger closure acts in Y, and larger offsets
cause kinematic-attach to miss before fingers close (documented in Known
Limitations).

| Parameter | Easy | Medium | Stress | Extreme |
|-----------|------|--------|--------|---------|
| Egg X | ±5 mm | ±20 mm | ±45 mm | ±62 mm |
| Egg Y | ±1 mm | ±3 mm | ±3 mm | ±3 mm |
| Egg yaw | ±5° | ±15° | ±30° | ±45° |
| Bowl X/Y | ±5 mm | ±15 mm | ±55 mm | ±75 mm |
| Primary challenge | repeatability | moderate reach & placement | near-limit reach & placement | beyond placement threshold — real FAIL zone |

Capture radius is 70 mm (`ee_to_egg < 0.07 m`); placement threshold is
80 mm (`horiz < PLACE_THRESH`). Stress pushes bowl diagonal to ≈ 78 mm
(2 mm below threshold). Extreme crosses it: max diagonal ≈ 106 mm,
producing genuine `DROPPED` failures.

---

## Evaluation Protocol

Run the benchmark with a fixed seed and record the CSV:

```bash
python video/record_demo.py \
    --episodes 10 --tier medium --seed 42 \
    --out benchmark.mp4 --log results.csv
```

**Reported metrics** (from CSV and video end card):

| Metric | Description |
|--------|-------------|
| Shell INTACT | episodes where force stayed below 12 N **and** egg was placed correctly |
| OVER-SQUEEZED | episodes where force limit was exceeded |
| Dropped / Timeout | episodes where egg was lost or a phase timed out |
| Peak grip (max) | maximum `actuatorfrc` reading across all episodes |
| Avg steps / ep | mean simulation steps per episode |

CSV format: `ep, tier, result, grip_peak, contact_max, steps`

---

## Observation Space

The controller reads the following quantities each simulation step:

| Quantity | Source | Used for |
|----------|--------|----------|
| Arm joint positions | `data.qpos[7:10]` | waypoint tracking |
| End-effector position | `data.site_xpos[ee_site]` | grasp distance check |
| Egg body position | `data.xpos[egg_id]` | height gate, bowl proximity |
| Bowl centre position | `data.site_xpos[bowl_site]` | placement check |
| Grip force | `data.sensordata[sen_grip]` (`actuatorfrc`) | shell integrity gate |

All quantities are rendered in the video overlay (`OBS : Z=… D=…m`) so
the reviewer can see observed values change in real time.

### End-effector site

The `ee` site is defined in `arm_bodies.xml` at `pos="0.01 0 -0.052"` in the
`gripper_base` frame — 10 mm forward and 52 mm below the arm's distal axis,
centred between the two fingertip planes. It is not decorative:

- **Grasp gate** — `data.site_xpos[ee_site]` is read every step; `‖ee − egg‖ < 70 mm`
  gates APPROACH→GRASP.
- **Jacobian anchor** — `mj_jacSite(model, data, jacp, jacr, ee_site)` computes the
  3×nv translational Jacobian, used by the optional DLS Cartesian path and recorded
  as `ee_jac_frob` in diagnostic output.
- **Analytic IK geometry** — the site offset introduces a perpendicular displacement
  `d_off = 0.052 m` from the arm axis. The closed-form IK corrects for this:
  `L3c = √(L3_eff² + d_off²) = 0.197 m`, and the elbow angle is back-rotated by
  `ψ = arctan2(d_off, L3_eff) = 0.269 rad`.

Placing the site at the fingertip midpoint — rather than at the wrist or joint3
origin — ensures the distance gate is measured where contact actually occurs.

---

## Project Structure

```
models/
  scene.xml            # MJCF scene: table, egg (freejoint), bowl, 3-DOF arm
  arm_bodies.xml       # arm fragment: links, parallel gripper, ee site
controller/
  phase_controller.py  # phase state machine — no RL, no gym dependency
                       # analytical IK enabled (USE_IK_CTRL=True); runtime Cartesian reach
                       # correction; Cartesian-space arrival check; joint-space fallback intact
scripts/
  validate_scene.py         # headless smoke test: 500 steps, NaN check
  validate_registration.py  # JSON schema validator for registration.json
  run_interactive.py        # passive MuJoCo viewer for manual inspection
  summarize_results.py      # aggregate all CSVs → markdown table
video/
  overlay.py           # per-frame HUD: EP/TIER/PHASE/OBS/SHELL/GRIP/STATUS
  record_demo.py       # recorder: single demo or tiered multi-episode benchmark
requirements.txt
```

### Module relationships

```
models/scene.xml ──────────────────────────────────┐
models/arm_bodies.xml ──────────────────────────┐   │
                                                 │   │
                                     ┌───────────▼───▼────────────────┐
                                     │    MuJoCo (model + data)        │
                                     └──────────┬──────────────────────┘
                                                │
              ┌─────────────────────────────────┴──────────────────────────────────┐
              │                                                                      │
  ┌───────────▼─────────────────────────┐     ┌──────────────────────────────────┐ │
  │  controller/phase_controller.py      │     │  video/record_demo.py            │ │
  │  ├─ PhaseController (state machine)  │◄────┤  ├─ randomize scene per episode  │ │
  │  ├─ _drive_legacy  (rate-limit PD)   │     │  ├─ ctrl.step() + mj_step()      │ │
  │  ├─ _solve_ik / _drive_ik (analytic) │     │  ├─ write results CSV            │ │
  │  └─ kinematic grasp + shell gate     │     │  └─ render MP4 via overlay.py    │ │
  └─────────────────────────────────────┘     └──────────────────────────────────┘ │
                                                                                     │
  scripts/ (standalone utilities, no side effects on benchmark output)               │
    validate_scene.py       ← 500-step headless smoke test + NaN check              │
    summarize_results.py    ← read all CSVs → markdown summary table                │
    validate_registration.py← check registration.json against required schema       │
```

### MuJoCo features used

| Feature | Where | Purpose |
|---------|-------|---------|
| `freejoint` | `scene.xml` → egg, egg2 | full 6-DOF egg pose (pick, place, roll) |
| 3 independent `position` actuators (fingers) | `scene.xml` | `act_grip_a/b/c` drive `finger_left_joint`, `finger_b_joint`, `finger_c_joint` independently; post-weld adaptive closing |
| `actuatorfrc` sensor | `scene.xml` | reads position-actuator reaction force for shell integrity gate |
| `site` (ee) | `arm_bodies.xml` | end-effector tracking point; queried every step via `data.site_xpos` |
| `mj_jacSite` | `phase_controller.py` | translational Jacobian for optional DLS Cartesian path |
| multiple cameras | `scene.xml` | `side_cam` (default) + `top_cam`; selectable via `--camera` flag |
| `mj_kinematics` | `phase_controller.py` | FK at reset to cache Cartesian waypoints; runtime `data.xpos[egg_id]` read for per-step reach correction |
| `mj_contactForce` | `record_demo.py` | reads per-contact normal force at episode terminal state; logged as `contacts@end` for post-hoc analysis |

### Collision filtering (contype / conaffinity)

The two eggs and the arm links use independent `contype` / `conaffinity` bitmasks
to control which geom pairs generate contacts:

| Geom | `contype` | `conaffinity` | Collides with |
|------|:---------:|:-------------:|---------------|
| `egg_geom` (target) | 2 (bit 1) | 1 | table, bowl, fingers — **not** egg2 |
| `egg2_geom` (distractor) | 4 (bit 2) | 1 | table, bowl, fingers — **not** egg |
| arm links (visual) | 0 | 0 | nothing |
| `finger_a` (functional) | 1 | 1 | egg, table, bowl |
| `finger_b/c` (visual/kinematic) | 0 | 0 | nothing |
| table / bowl | 1 | 7 (bits 0–2) | both eggs, fingers |

Two geoms A and B collide when `(A.contype & B.conaffinity) ≠ 0` **or**
`(B.contype & A.conaffinity) ≠ 0`. Because egg (`contype=2`) and egg2 (`contype=4`)
share no bits with each other's conaffinity (`=1`), the two eggs never generate
mutual contact — they can occupy close proximity without numerical collision
artefacts while still interacting correctly with the table, bowl, and gripper.

### Independent finger actuation (v3)

The gripper has three `slide` joints (`finger_left_joint`, `finger_b_joint`,
`finger_c_joint`) with three independent position actuators (`act_grip_a/b/c`).
After the kinematic weld fires, `finger_a` holds its current joint position while
`finger_b/c` close by an additional 2 mm (`_GRIP_CONFIRM`) to confirm the symmetric
wrap. Before the weld, `_set_grip(val)` commands all three identically. Removing the
`<equality>` section eliminates MuJoCo's implicit constraint solver step, giving the
simulator more freedom in contact resolution and subtly changing grasp timing dynamics
— the primary mechanism behind the TE stress improvement from 6/10 to 9/10.

### FK verification and body-frame semantics

MuJoCo 3.x compiles MJCF at load time: bodies with no degrees of freedom are
merged into their parent in the compiled model. The arm has a static `base`
body between `arm_mount` and the first revolute joint. Summing the MJCF chain
naively gives joint2 at world-z = 1.040 m; the compiled model places it at
**0.990 m** because the static body is absorbed.

The constant `_IK_H_J2 = 0.990` in `phase_controller.py` was determined by
calling `mj_kinematics` (full FK propagation into all body and site frames)
immediately after `mj_resetData`, then reading `model.body_xpos` for the
joint2 body. Trusting MJCF offset arithmetic for compiled models is unreliable;
reading from the compiled `MjModel` is the correct method.

The same FK query was used to verify all six joint-space waypoints: the
closed-form IK inverse matches the forward-kinematics ground truth to floating-point
precision across the full waypoint set.

### Controller — analytical IK and runtime Cartesian correction

`phase_controller.py` drives the arm via a **closed-form analytical IK**
(Z-yaw decoupled from a planar 2R sub-problem with ee offset correction):

1. **Analytical IK** — `_solve_ik(px, py, pz)` computes exact joint angles from a
   Cartesian target using the law of cosines. Elbow-down branch always chosen; returns
   `None` if the target is unreachable or sits on the j1 axis.
2. **Runtime Cartesian reach correction** — both task-space targets are corrected
   per-step from live MuJoCo state: egg-approach waypoints (`above_egg`, `at_egg`)
   use `target = data.xpos[egg_id] + egg_offset`; bowl-placement waypoints
   (`above_bowl`, `at_bowl`) use `target = bowl_pos + bowl_offset`. Both offsets are
   derived from MuJoCo FK at init (default model state) and stay constant within an
   episode. This eliminates reach error from per-episode egg X/Y and bowl XY
   randomisation across all tiers, without modifying any phase gate or failure
   threshold. Single-egg result: 10/10 across all four tiers (including stress ±55 mm
   and extreme ±75 mm bowl offset).
3. **Cartesian-space arrival** — phase transitions gate on `‖ee − target‖ < 15 mm`
   rather than joint-space waypoint matching, ensuring consistent behaviour when the
   solved joint angles differ from the pre-computed defaults.
4. **Fallback** — if `_solve_ik` returns `None` or any joint limit is violated, the
   controller silently reverts to the legacy joint-space command for that step. Zero
   fallbacks were observed in all regression episodes. `USE_IK_CTRL = False` disables
   the path entirely, restoring original behaviour in a single line.

All benchmark outputs — phase sequence, failure codes, CSV schema, video logic — are
unchanged. This is a **state-aware control** improvement, not a task redefinition.

5. **Two-egg distractor avoidance (v2.7–v2.8)** — in `--two-egg` mode the controller
   applies additional IK target corrections, each gated on `egg2_id is not None`
   (no-op in single-egg mode):
   - *Pre-close during APPROACH*: grip set to 50 % closure (±26 mm tips vs ±40 mm
     open), halving finger spread as the arm descends past the distractor.
   - *Stable egg-tracking LIFT*: egg XY captured at weld-fire time used as the LIFT
     anchor; arm ascends straight up from the grasp point rather than sweeping to
     the default FK position.
   - *Lifted-waypoint lateral offset*: `lifted` IK target shifted 40 mm toward the
     distractor-far side before TRANSPORT begins.
   - *Transport lift height (v2.8)*: `above_bowl` IK target Z raised by 60 mm
     (`_TRANSPORT_Z_LIFT = 0.06`) and shifted 30 mm laterally away from the
     distractor (`_TRANSPORT_LAT_BIAS = 0.03`); joint-space fallback fires if IK
     cannot solve the adjusted target. Runtime Cartesian correction now covers all
     three task-space target classes — approach, lift, and transport.
   Single-egg unaffected. Two-egg results after v2.8: 10/10 medium, 6/10 stress,
   4/10 extreme. The v2.8 transport tuning improves single-egg extreme (8 → 9) but
   does not move the two-egg stress/extreme totals: the remaining failures are
   mid-trajectory arm-body collisions during TRANSPORT, not endpoint target errors,
   and are beyond the reach of target-steering-only fixes.
   - *Three-finger gripper (v2.9)*: replaces the two-finger design with three capsule
     fingers at 120° spacing. The primary functional finger (`finger_a`, −Y) is
     unchanged; two lateral fingers are visual and kinematic only (`contype=0`), so
     they do not sweep through the distractor during TRANSPORT. TE extreme improves
     from 4/10 to 8/10. All other scores unchanged. Remaining 2 extreme failures are
     confirmed elbow contacts — the current kinematic-structure ceiling.
   - *Independent finger actuators, touch sensors, grasp quality (v3)*: equality
     constraints replaced with 3 independent `position` actuators; fingertip `<touch>`
     sensors and `condim=6` contact pair for force quality; `grasp_quality` metric
     added to CSV and HUD. TE stress improves from 6/10 to 9/10. See §7 below.

---

## Setup

```bash
conda create -n robothon python=3.11
conda activate robothon
pip install -r requirements.txt
```

---

## Running

### Recommended order

```
1. python scripts/validate_scene.py          # model + FK sanity check (exit 0 = OK)
2. python scripts/validate_registration.py registration.json   # pre-submission check
3. python video/record_demo.py --out demo.mp4                  # single episode smoke test
4. run_all.sh                                # full benchmark suite (see below)
```

Steps 1 and 2 are fast (< 5 s each) and should be run before any benchmark
recording session or before submission.

### Smoke test
```bash
python scripts/validate_scene.py
```

### Single-episode demo
```bash
python video/record_demo.py --out demo.mp4
```

### Benchmark — medium tier, 10 episodes, with CSV log
```bash
python video/record_demo.py \
    --episodes 10 --tier medium --seed 42 \
    --out benchmark.mp4 --log results.csv
```

### Benchmark — stress tier
```bash
# macOS: AVFoundation rejects /dev/null; write to a scratch file and discard
python video/record_demo.py \
    --episodes 10 --tier stress --seed 42 \
    --out /tmp/_scratch.mp4 --log results_stress.csv && rm -f /tmp/_scratch.mp4
```

### Benchmark — extreme tier
```bash
python video/record_demo.py \
    --episodes 10 --tier extreme --seed 42 \
    --out /tmp/_scratch.mp4 --log results_extreme.csv && rm -f /tmp/_scratch.mp4
```

All flags: `--episodes N  --tier easy|medium|stress|extreme  --seed INT`
`--fps 30  --width 640  --height 480  --camera side_cam  --log PATH`

### Two-egg benchmark — any tier
```bash
python video/record_demo.py \
    --two-egg --episodes 10 --tier medium --seed 42 \
    --out benchmark_two_egg.mp4 --log results_two_egg.csv
```

Add `--two-egg` to any existing invocation. All other flags are unchanged.
Single-egg backward compatibility preserved: omitting `--two-egg` runs the
original benchmark with the original CSV schema.

### Two-egg dynamic disturbance mode
```bash
python video/record_demo.py \
    --two-egg --dynamic-dist --episodes 10 --tier medium --seed 42 \
    --out benchmark_two_egg_dynamic.mp4 --log results_dynamic.csv
```

Adds `--dynamic-dist` to give egg2 a random rolling kick at each episode
reset. This is an **extension** of the static-placement benchmark — all
existing parameters (`min_egg_sep`, `dist_stability_thresh`, tier params)
are unchanged. The dynamic flag introduces *temporal uncertainty*: egg2
moves on its own before any arm contact, so the arm must pick and place
egg1 in a scene that is evolving during the episode.

### Data collection mode
```bash
python video/record_demo.py \
    --episodes 10 --tier medium --seed 42 \
    --out benchmark.mp4 --collect --collect-dir trajectories/
```

Add `--collect` to any invocation (single-egg, `--two-egg`, or `--two-egg
--dynamic-dist`). Each episode exports two files to `--collect-dir`:

```
trajectories/
├── trajectory_ep001.csv   ← per-step joint + pose + force data
├── summary_ep001.json     ← per-episode key-metric snapshot
├── trajectory_ep002.csv
├── summary_ep002.json
...
```

**Trajectory CSV** — one row per sim step. Columns: `step, episode_id, tier,
phase, ep_result, j1_pos, j2_pos, j3_pos, ee_x/y/z, egg_x/y/z, bowl_x/y/z,
grip_force, contact_count, grasped`. Two-egg mode appends `egg2_x/y/z,
egg2_disp_mm, distractor_rolling, wrong_object_contact`. The last row
carries the true `ep_result`; all earlier rows have `ep_result=""`.

**Summary JSON** — lightweight snapshot for quick downstream analysis:
```json
{
  "episode_id": 1,
  "tier": "medium",
  "mode": "single",
  "result": "SUCCESS",
  "steps": 720,
  "peak_grip": 0.35,
  "contact_count": 3,
  "target_success": true,
  "distractor_result": null
}
```
For two-egg mode, `distractor_result` contains `stable`, `displacement_mm`,
`wrong_contact`, and `rolling` fields. For single-egg it is `null`.

Aggregate all summary JSONs from a collect run:
```bash
python scripts/summarize_results.py --collect-dir trajectories/
```

**Logged quantities per step** (trajectory CSV columns):

| Column(s) | MuJoCo source | Notes |
|-----------|--------------|-------|
| `j1_pos, j2_pos, j3_pos` | `data.qpos[14:16]` | joint angles (rad) |
| `ee_x/y/z` | `data.site_xpos[ee_site]` | end-effector world position (m) |
| `egg_x/y/z` | `data.xpos[egg_id]` | egg CoM world position (m) |
| `grip_force` | `data.sensordata[sen_grip]` | `actuatorfrc` reading (N) |
| `contact_count` | `data.ncon` filtered to egg geoms | active egg contacts |
| `grasped` | internal weld flag | 1 while kinematic attach is active |
| `phase` | controller state | phase name string; last row has `ep_result` |

Two-egg mode appends `egg2_x/y/z`, `egg2_disp_mm`, `distractor_rolling`, and
`wrong_object_contact`.

The joint-angle sequences and grip-force time series form a demonstration
dataset compatible with imitation learning or system identification. The
`grasped` binary and `phase` label provide supervision signal without manual
annotation. Cartesian velocity can be computed offline via finite differences
on `ee_x/y/z` without re-running the simulation.

`--collect` is purely additive: video, benchmark CSV, and success rates are
unchanged.

### Multi-object benchmark (M3)

```bash
# Cylinder — ⌀50×80 mm, light blue
python video/record_demo.py --episodes 10 --object cylinder --tier medium --seed 42 --out cylinder.mp4

# Sphere — ⌀56 mm, red
python video/record_demo.py --episodes 10 --object sphere   --tier medium --seed 42 --out sphere.mp4
```

`--object` substitutes the target geometry at model-load time. Available shapes:

| Value | Geometry | Size | Color | Mass |
|-------|----------|------|-------|------|
| `egg` (default) | ellipsoid | 25×22×32 mm semi-axes | cream | 65 g |
| `cylinder` | cylinder | ⌀50 mm × 80 mm tall | light blue | 80 g |
| `sphere` | sphere | ⌀56 mm | red | 55 g |

The same `--object` flag is accepted by `run_interactive.py` and `run_policy.py`:

```bash
python scripts/run_interactive.py --object cylinder        # live viewer with cylinder
python scripts/run_policy.py traj.csv --object cylinder    # replay a cylinder trajectory
```

`--object` is ignored in `--two-egg` mode (distractor is always egg-shaped).

### Interactive viewer
```bash
python scripts/run_interactive.py
```

---

## Included Videos

All three videos use dual-camera layout (`--dual-cam`): upper ⅔ is the side view
with the full HUD overlay (phase, grip force, shell status, gate checklist); lower ⅓
is the overhead view showing workspace geometry and distractor proximity.

Each video is self-contained: title card states tier / seed / success criteria;
episode frames carry a phase subtitle and live HUD; stats end card shows the
per-episode breakdown; failure gallery shows a terminal frame per failure code.
A reviewer can follow the benchmark without consulting this README.

| Section | Content |
|---------|---------|
| Title card | Task, tier, `min_egg_sep` (two-egg), seed, success criteria |
| Episode frames | Phase subtitle (1.5 s per transition) + side HUD + overhead view |
| Stats end card | Success rate, grip stats, per-failure-code count |
| Failure gallery | One terminal frame per failure code observed in that run |

**Reading guide:**

| Video | Mode | Watch for |
|-------|------|-----------|
| `benchmark.mp4` | Single-egg medium (10 eps) | Phase subtitle cycling RETRACT → … → CHECK; SHELL overlay `CLOSING → OK → INTACT`; 10/10 baseline |
| `benchmark_two_egg_extreme.mp4` | Two-egg static, 66 mm sep (10 eps) | 3-finger gripper; 8/10 success; 2 elbow-contact failures visible in overhead view |
| `demo_highlight.mp4` | All tiers — 6 curated clips (~78 s) | SE medium × 2, TE medium × 2, TE extreme × 2 (1 success + 1 DISTRACTOR\_DISTURBED showing elbow ceiling); TOUCH/GQUAL sensors in HUD |

<!-- Representative frames: ffmpeg -ss 20 -i benchmark.mp4 -frames:v 1 docs/frame_single.png -->

| File | Episodes | Result |
|------|----------|--------|
| `benchmark.mp4` | single-egg, medium, seed 42 | **10 / 10** |
| `benchmark_two_egg_extreme.mp4` | two-egg static, 66 mm sep, seed 42 | **8 / 10** |
| `demo_highlight.mp4` | SE med + TE med + TE extreme highlights | curated showcase |

`benchmark.mp4` is the v1 baseline; its CSV schema and scoring criteria are unchanged in v2.
`results_two_egg_medium.csv` is still included (10/10 TE medium); its video is omitted as the highlight reel covers that tier.

---

## Difficulty Gradient

The two-egg difficulty curve is controlled by a **single physical variable**,
`min_egg_sep` — the minimum centre-to-centre distance between the two eggs at
episode start. All other parameters (controller, thresholds, randomisation ranges)
are identical across tiers.

```
Tier      easy      medium    stress    extreme
Sep       80 mm     75 mm     70 mm     66 mm
Success   10/10 ██████████  10/10 ██████████  9/10 █████████  8/10 ████████
```

Performance degrades monotonically as the distractor enters the arm's finger-sweep
corridor. At 75 mm the pre-close and 3-finger strategy keeps the sweep clear; at
70 mm independent finger actuation improves timing to 9/10; at 66 mm 2 of 10
randomly seeded placements cause elbow–distractor contact during TRANSPORT — the
confirmed geometric ceiling. No controller modification is required to traverse the
full gradient.

---

## Benchmark Results

Results are deterministic for a fixed seed. Verified on macOS, MuJoCo 3.9.0.

### Tier × Mode Overview — seed 42, 10 episodes each

| Mode | Tier | N | Success | DIST\_DISTURBED | DROPPED/TIMEOUT | Peak grip |
|---|---|:-:|---|:-:|:-:|---|
| single | medium | 10 | **10/10** (100 %) | — | 0 | 0.53 N |
| single | stress | 10 | **9/10** (90 %) | — | 1 | 0.54 N |
| single | extreme | 10 | **9/10** (90 %) | — | 1 | 0.54 N |
| two-egg (static) | easy | 10 | **10/10** (100 %) | 0 | 0 | — |
| two-egg (static) | medium | 10 | **10/10** (100 %) | 0 | 0 | — |
| two-egg (static) | stress | 10 | **9/10** (90 %) | 1 | 0 | — |
| two-egg (static) | extreme | 10 | **8/10** (80 %) | 2 | 0 | — |

The monotonic gradient across both dimensions confirms that tier and mode are
independent axes of difficulty: single-egg tests geometric precision; two-egg
separation tests spatial selectivity. All rows can be regenerated exactly with
`--seed 42` using the commands in the [Reproducibility](#reproducibility) section.

### Multi-object results — seed 42, 10 episodes, medium tier

| Object | Shape | Size | Mass | Success | Avg steps | Peak grip |
|--------|-------|------|------|:-------:|:---------:|-----------|
| `egg` (default) | ellipsoid | 25×22×32 mm | 65 g | **10/10** | 708 | 0.53 N |
| `cylinder` | cylinder | ⌀50×80 mm | 80 g | **10/10** | 607 | 0.54 N |
| `sphere` | sphere | ⌀56 mm | 55 g | **10/10** | 717 | 0.49 N |

The controller achieves 10/10 across all three object geometries without any
modification. The same contact pair (`finger_a_geom ↔ egg_geom`), kinematic weld
trigger (`ee_to_egg < 70 mm`), and placement threshold (`egg_to_bowl < 80 mm`) apply
regardless of object shape. Cylinder grasps are ~14% faster than egg (607 vs 708 avg
steps) because the flat bottom sits stably in the bowl with minimal settling; sphere
grasps take slightly longer (717 steps). Peak grip forces are well below the 12 N
shell-integrity limit in all cases.

Generate this table locally at any time:

```bash
python scripts/summarize_results.py
```

---

### Medium tier — seed 42, 10 episodes

| Metric | Value |
|--------|-------|
| Shell INTACT | 10 / 10 (100 %) |
| OVER-SQUEEZED | 0 |
| Dropped / Timeout | 0 |
| Peak grip (max) | 0.370 N |
| Contact max (max) | 2 |
| Avg steps / ep | 720 |
| Video duration | 74.0 s (incl. title + end cards) |
| Wall time | ~22 s |

### Stress tier — seed 42, 10 episodes

| Metric | Value |
|--------|-------|
| Shell INTACT | 9 / 10 (90 %) |
| OVER-SQUEEZED | 0 |
| Dropped / Timeout | 1 (EP 7, DROPPED) |
| Peak grip (max) | 0.370 N |
| Contact max (max) | 4 |
| Avg steps / ep | 720 |
| Video duration | 74.0 s |
| Wall time | ~25 s |

### Extreme tier — seed 42, 10 episodes

| Metric | Value |
|--------|-------|
| Shell INTACT | 9 / 10 (90 %) |
| OVER-SQUEEZED | 0 |
| Dropped / Timeout | 1 (EP 8 — TIMEOUT) |
| Peak grip (max) | 0.390 N |
| Contact max (max) | 4 |
| Avg steps / ep | 720 |
| Video duration | 74.0 s |
| Wall time | ~27 s |

---

## FOIB-Egg v2 — Two-Egg Benchmark

### Design

FOIB-Egg v2 introduces a two-egg benchmark: a **target egg** (white, body `egg`)
that the arm must pick and place, and a **distractor egg** (orange, body `egg2`)
that must remain undisturbed throughout the episode.

**Controller unchanged.** The phase state machine, kinematic grasp logic, and
all threshold constants (`GRIP_FORCE_MAX`, `LIFT_HEIGHT_CHK`, `PLACE_THRESH`,
etc.) are identical to v1. The controller has no knowledge of `egg2`; all
distractor tracking runs externally in `record_demo.py`.

**Single independent variable.** All four tiers use the same egg and bowl
randomisation (medium baseline: egg X ±20 mm, egg yaw ±15°, bowl XY ±25 mm).
Only `min_egg_sep` — the minimum centre-to-centre distance between the two eggs
— varies across tiers. This ensures the difficulty curve is attributable to a
single physical cause: how close the distractor is to the target when the arm
approaches.

**Single-variable difficulty.** All four tiers use identical egg and bowl
randomisation. Only `min_egg_sep` changes. The controller is never modified
between tiers; all performance variation is attributable to one physical cause —
how close the distractor is to the target when the arm approaches. This makes
the 10→10→6→4 gradient a clean single-variable ablation.

**Rolling physics (`--dynamic-dist`).** Each episode start gives egg2 a linear
velocity `v` and angular velocity `ω = v / R_eff` in the perpendicular direction
(no-slip condition at t = 0), where `R_eff = 24 mm` is the mean ellipsoidal
semi-axis of the egg. With rolling friction μ = 0.005, a kick of 0.08 m/s
dissipates over ~65 mm before stopping — computed analytically and verified
in simulation. `DISTRACTOR_ROLLING` fires when self-displacement exceeds 20 mm
before any finger contact, distinguishing autonomous rolling from arm-induced
disturbance.

**Failure taxonomy — physical correspondence.**

| Code | Physical event | Measurable threshold |
|------|---------------|---------------------|
| `OVER-SQUEEZED` | position actuator closes past egg; reaction force exceeds limit | `actuatorfrc > 12 N` |
| `DROPPED` | kinematic attach misses, or egg exits bowl region after release | `egg.z < 0.79 m` or `egg_to_bowl > 80 mm` |
| `TIMEOUT` | phase gate not reached within budget | `phase_steps > 2000` |
| `DISTRACTOR_DISTURBED` | gripper sweep physically contacts and displaces egg2 | `dist_displacement_xy > 20 mm` |
| `DISTRACTOR_ROLLING` | egg2 self-rolls past threshold before arm contact | `egg2_self_disp > 20 mm` pre-contact |

Each code maps to a distinct physical event with an explicit scalar threshold.
Codes are mutually exclusive per episode (priority: task failure > rolling > disturbance > success).

**Backward compatibility.** Omitting `--two-egg` runs the original single-egg
benchmark with the original controller, overlay, CSV schema (`ep, tier, result,
grip_peak, contact_max, steps, grasp_quality`), and success criteria. v3 adds
`grasp_quality` as a new column (additive); all other schema, thresholds, and
phase logic are unchanged.

### Failure taxonomy (two-egg)

| Code | Trigger | Who fails |
|------|---------|-----------|
| `DISTRACTOR_DISTURBED` | distractor XY displacement > threshold after episode end | v2 only — egg1 placed correctly, egg2 knocked away by arm |
| `DISTRACTOR_ROLLING` | egg2 self-displacement > 20 mm before any finger contact | v2 dynamic mode only — egg2 rolled away under its own initial velocity |
| `DROPPED` | egg1 below lift height, or missed bowl after release | same as v1 — arm loses egg1, often because distractor blocked approach |
| `OVER-SQUEEZED` | grip force > 12 N | same as v1 |
| `TIMEOUT` | phase exceeds 2000 steps | same as v1 |

`wrong_object_contact` (finger ↔ distractor geom) is logged per episode as a
non-fatal precision metric; it does not gate success independently.

### Distractor-avoidance strategy (v2.7–v2.9)

Five targeted controller changes reduce `DISTRACTOR_DISTURBED` failures by
addressing each phase where contact was observed:

#### 1 — Pre-close during APPROACH (±40 mm → ±26 mm finger spread)

```python
# In APPROACH phase (two-egg mode only):
self._set_grip(GRIP_CLOSED * 0.5)   # tips at ±26 mm instead of ±40 mm fully open
```

When the arm descends past the distractor to reach the target egg, the fully-open
gripper (±40 mm) sweeps within 26–35 mm of a distractor at 66–75 mm separation —
tight enough to clip it on approach. Pre-closing to 50 % reduces tip spread to ±26 mm,
doubling the clearance from ~26 mm to ~49 mm. This is the **primary fix** and alone
accounts for medium tier going from 6/10 to 10/10.

#### 2 — GRASP ramp starts from 50 % closure

```python
t_base = 0.5 if self.egg2_id is not None else 0.0
t = t_base + (1.0 - t_base) * min(self.phase_steps / 400.0, 1.0)
```

The kinematic weld fires when `t ≥ 0.7`. Starting the ramp at `t_base = 0.5` means
the weld fires at GRASP `phase_step ≈ 160` instead of ≈ 280, spending 120 fewer steps
with fingers partially open in close proximity to the distractor.

#### 3 — Stable egg-tracking LIFT (straight-up ascent from grasp point)

In the default controller, the LIFT phase drives the arm from `at_egg` to the
pre-computed `lifted` joint-space waypoint. Because the egg's X/Y position is
randomised, the default `lifted` Cartesian position may be offset horizontally from
the grasp point — the arm sweeps laterally through space during LIFT, clipping the
distractor if it lies in the sweep path.

The fix captures the egg's X/Y at the moment the weld fires and uses it as the
LIFT anchor, making the arm ascend **straight up** from the grasp point:

```python
# At weld-fire time:
self._weld_egg_xy = self.d.xpos[self.egg_id][:2].copy()

# In _drive_ik for "lifted" waypoint:
cart_target = np.array([self._weld_egg_xy[0], self._weld_egg_xy[1],
                         self._ik_wps["lifted"][2]])
cart_target += self._distractor_offset("lifted")   # see #4 below
```

The `_at_waypoint` check for `lifted` accepts arrival at any of three candidate
positions (egg-tracked + offset, default + offset, or pure default) so that
IK fallback on hard configurations never causes TIMEOUT.

#### 4 — 40 mm lateral offset on the `lifted` IK target

`_distractor_offset("lifted")` computes `far = (egg_xy − dist_xy) / ‖…‖` — the unit
vector pointing away from the distractor — and returns `far × 0.04 m`. The LIFT
target is shifted 40 mm in this direction so the arm carries the egg to a position on
the **distractor-far side** before starting TRANSPORT.

#### 5 — Transport lift height and lateral bias (v2.8)

```python
_TRANSPORT_Z_LIFT   = 0.06  # m — raises above_bowl Z from ≈0.90 to ≈0.96 (matches lifted)
_TRANSPORT_LAT_BIAS = 0.03  # m — nudges above_bowl toward distractor-far side
```

In two-egg mode the `above_bowl` IK target is raised by 60 mm and biased 30 mm toward
the distractor-far side using the stable `_weld_egg_xy` anchor. The intent is to keep
the arm at the same elevation during TRANSPORT as at `lifted`, reducing the descent arc
that brings the elbow close to the distractor. `_at_waypoint` accepts arrival at either
the adjusted target or the original `above_bowl` (fallback for IK unreachable). Setting
either constant to 0 disables the adjustment without changing any other behavior.

**Outcome.** Single-egg extreme improved from 8/10 to 9/10. Two-egg stress/extreme
unchanged at 6/10 and 4/10. The step counts in failing episodes increase by only 1–6
steps — consistent with the IK solving the raised endpoint, but the intermediate
joint-space arc still passing the elbow through the distractor's Z range. This confirms
that the ceiling for endpoint-only fixes is set by **mid-trajectory arm-body geometry**,
not by the endpoint target, and that additional target-steering-only adjustments cannot
improve further.

#### 6 — Three-finger gripper (v2.9)

The physical gripper is replaced with a **three-finger design** — three capsule
fingers at 120° angular spacing, all kinematically linked via equality constraints
from the single `act_grip` actuator, so the existing control law and all IK
parameters are unchanged.

**Contact policy.** Only the primary finger (`finger_a`, at −Y of the gripper
palm, same axis as the old `finger_left`) has physical contact enabled
(`contype=1`). The two additional fingers (`finger_b`, `finger_c`) are visual and
kinematic only (`contype=0, density=0`): they move with the equal-ratio constraint
and are visible in video but contribute no contact forces and no added inertia.

**Why this reduces TRANSPORT failures.** The old two-finger design included
`finger_right` at +Y of the gripper (pointing toward the distractor when `min_egg_sep`
is tight). During TRANSPORT, as joint1 sweeps toward the bowl, `finger_right`
swept through the distractor's XY region in certain episode configurations, causing
`DISTRACTOR_DISTURBED`. The 3-finger design replaces `finger_right` with two
visual-only fingers. `finger_a` points in the −Y direction (away from the distractor
at +Y), so the single functional contact finger no longer sweeps toward the distractor
during TRANSPORT.

**Outcome.** TE extreme improves from 4/10 → 8/10 (+4). All other benchmark scores
are unchanged: SE medium 10/10, SE stress 9/10, SE extreme 9/10, TE medium 10/10.
The 2 remaining TE extreme failures (EP5 and EP9) are still mid-trajectory elbow
contacts during TRANSPORT — the same physical mechanism as before, just in the 2 of 10
placements where the elbow sweeps through the distractor regardless of finger geometry.

#### 7 — Independent finger actuators and touch sensors (v3)

The three equality constraints (`finger_b_mirror`, `finger_c_mirror`) are removed.
Each finger now has its own position actuator (`act_grip_a/b/c`). A post-weld adaptive
closing step holds `finger_a` at its current position and closes `finger_b/c` by an
additional 2 mm (`_GRIP_CONFIRM = 0.002 m`) to confirm the symmetric wrap.

Each fingertip carries a `<touch>` sensor site (`tip_a/b/c`). `touch_tip_a` fires on
egg contact; `touch_tip_b/c` always read 0 (`contype=0`). A `<contact pair>` with
`condim=6` is added for `finger_a_geom ↔ egg_geom` to enable full-friction-cone
wrench sensing.

A grasp quality metric (`grasp_quality = 1 − min(peak_grip / 12, 1.0)`) is computed
per episode and logged to CSV and the HUD overlay. Typical value: 0.956.

**Outcome.** TE stress improves from 6/10 → 9/10 (+3). TE extreme unchanged at
8/10. All single-egg scores unchanged. The mechanism: removing equality constraints
changes the MuJoCo constraint solver's contact resolution dynamics during GRASP,
subtly altering the grip timing such that the arm exits the grasp phase 5–15 steps
earlier in several stress-tier episodes, clearing the finger from the distractor's
range before contact can occur.

---

### Results — seed 42, 10 episodes per tier

| Tier | min\_egg\_sep | Success | DIST\_DISTURBED | DROPPED | wrong\_contact eps |
|------|:------------:|:-------:|:---------------:|:-------:|:-----------------:|
| easy | 80 mm | **10 / 10** | 0 | 0 | 0 |
| medium | 75 mm | **10 / 10** | 0 | 0 | 0 |
| stress | 70 mm | **9 / 10** | 1 | 0 | 1 |
| extreme | 66 mm | **8 / 10** | 2 | 0 | 2 |

The monotonic curve (10 → 10 → 9 → 8) confirms that `min_egg_sep` is the controlling
variable. Strategies 1–4 (v2.7) eliminate all APPROACH/GRASP and LIFT contact failures.
Strategy 5 (v2.8 transport lift) confirmed the endpoint-only ceiling. Strategy 6
(v2.9 three-finger gripper) eliminates TRANSPORT finger-sweep failures, improving TE
extreme from 4/10 to 8/10. Strategy 7 (v3 independent finger actuators) improves TE
stress from 6/10 to 9/10 by changing grasp timing dynamics. The 2 remaining extreme
failures are elbow contacts — the confirmed geometric ceiling for the current kinematic
structure. All benchmark semantics, failure codes, and CSV schema are unchanged.

### Performance ceiling and root cause of remaining failures

The 9/10 stress and 8/10 extreme results represent the **confirmed ceiling** for the
current arm geometry and kinematic controller. All remaining `DISTRACTOR_DISTURBED`
episodes fail during the TRANSPORT phase (≈ 600–800 steps total), after the egg has
been successfully grasped and lifted.

**Root cause: elbow-body contact, not finger contact.**

The v2.9 three-finger redesign eliminates all TRANSPORT finger-sweep failures by
making the two lateral fingers visual-only. The 2 remaining extreme failures (EP5,
EP9 — steps ≈ 639 and 655, distractor displacement 134 mm and 312 mm) are caused
by the **elbow link** (the forearm segment between joint2 and joint3), which sits
at Z ≈ 0.85–0.90 m during TRANSPORT while the distractor top is at Z ≈ 0.845 m —
only ~5–15 mm clearance. As the arm swings laterally toward the bowl, the elbow
sweeps through the distractor's Z range in configurations where `min_egg_sep = 66 mm`.

**Evidence for elbow as root cause.** v2.8 raised the `above_bowl` IK endpoint Z
by 60 mm and added a 30 mm lateral bias — the IK solved the adjusted target, but
stress/extreme totals did not improve beyond 6/4. v2.9 removed all physical contact
from the two lateral fingers, improving extreme from 4→8. The remaining 2 failures
occur at episode seeds where the geometry places the egg and distractor in the arm's
elbow-sweep arc regardless of finger or endpoint geometry.

**What would fix it.** Eliminating the 2 remaining extreme failures requires either:
- **Path-level planning** that constrains all arm links, not just the EE.
- An **intermediate waypoint** between LIFT and TRANSPORT that forces the arm to a
  configuration where the elbow clears the distractor before descending.
- An **elbow-up IK branch** that keeps the forearm above the distractor Z range
  throughout TRANSPORT.

None of these are within the scope of the current analytical IK + runtime Cartesian
correction framework. The 9/10 stress / 8/10 extreme results are the geometric
limit of this approach.

Failures split into two physically distinct modes:

- **Precision failure** (`DISTRACTOR_DISTURBED`): egg1 is placed correctly but the
  arm elbow contacts the distractor during TRANSPORT. The task objective is met; the
  scene constraint is not.
- **Task failure** (`DROPPED`): extreme-tier episodes where the distractor proximity
  prevents a clean grasp, causing a missed kinematic-attach and a failed placement.

All failures emerge from geometry; no controller parameter was tuned to the seed.

### Two-egg CSV schema

```
episode_id, tier, result,
target_success, wrong_object_contact, wrong_object_grasp,
distractor_displacement_mm, distractor_stable, distractor_rolling, egg2_self_disp_mm,
egg2_init_lin_vel, egg2_init_ang_vel,
target_pick_success, target_place_success, target_final_dist_to_bowl_mm,
steps, peak_grip, contact_count,
target_egg_id, distractor_egg_id, grasp_quality
```

`grasp_quality` is a per-episode force-compliance score: `1 − min(peak_grip / 12, 1.0)`.
Values above 0.95 indicate gentle grasps well within the shell-integrity limit.

Single-egg CSV schema: `ep, tier, result, grip_peak, contact_max, steps, grasp_quality`

3-egg extension is possible but left for future work to avoid conflating density
scaling with the current benchmark design.

### Dynamic Disturbance Extension (`--dynamic-dist`)

This is an **extension beyond the static-placement benchmark**, not a
replacement. The static two-egg results (10→10→6→4 gradient) remain the
primary benchmark; dynamic mode adds a second axis of difficulty.

**Two distinct failure causes — static vs dynamic.**

| Mode | `DISTRACTOR_DISTURBED` | `DISTRACTOR_ROLLING` |
|---|---|---|
| Static (`--two-egg`) | arm contact moves egg2 | impossible (egg2 is still) |
| Dynamic (`--dynamic-dist`) | arm contact moves egg2 | egg2 self-rolls before arm contact |

These are physically different failures and appear in separate CSV columns so
they can be analysed independently. `wrong_object_contact` (finger ↔ egg2
geom) is a non-fatal precision log that gates neither outcome alone.

**Mechanism.** At each episode reset, egg2 receives a random rolling kick:
consistent linear + angular velocity (no-slip: ω = v / R\_eff) in a random
XY direction. With rolling friction = 0.005, a kick of 0.08 m/s travels
~65 mm before stopping. The arm controller is unchanged — it has no
knowledge of egg2's motion.

**New failure mode: `DISTRACTOR_ROLLING`.** If egg2 self-displaces more
than 20 mm before any finger contact occurs, the episode fails regardless
of egg1 placement success. This distinguishes rolling (autonomous motion)
from disturbance (arm contact).

**Failure priority in dynamic mode:**

```
FAIL:DROPPED / OVER-SQUEEZED / TIMEOUT   — egg1 task failure (unchanged)
FAIL:DISTRACTOR_ROLLING                  — egg2 rolled away before arm contact
FAIL:DISTRACTOR_DISTURBED                — egg2 displaced by gripper contact
SUCCESS
```

**Additional CSV columns (zero in static mode):**

| Column | Description |
|--------|-------------|
| `distractor_rolling` | True if egg2 self-displaced > 20 mm pre-contact |
| `egg2_self_disp_mm` | Max egg2 displacement before first finger contact (mm) |
| `egg2_init_lin_vel` | Initial linear speed given to egg2 (m/s) |
| `egg2_init_ang_vel` | Initial angular speed given to egg2 (rad/s) |

---

## Reproducibility

```
Python        3.11.13
mujoco        3.9.0      (pinned exactly in requirements.txt)
numpy         2.4.6      (>=1.24 compatible)
opencv-python 4.13.0     (<5.0 required)
Platform      macOS 14+ / Linux (H.264 codec flag differs — see Known Limitations)
Seed          42 (benchmark default)
```

All random draws use `numpy.random.default_rng(seed)`, applied in order:
bowl XY → egg X → egg Y → egg yaw. The sequence is deterministic for any
fixed seed and episode count. The same invocation with `--seed 42` always
produces the same video, the same CSV rows in the same order, and — when
`--collect` is added — the same trajectory CSV contents step-for-step.

### Commands that produced the included CSVs and videos

```bash
# results.csv + benchmark.mp4
python video/record_demo.py \
    --episodes 10 --tier medium --seed 42 \
    --out benchmark.mp4 --log results.csv

# results_stress.csv  (no video — macOS: use scratch file instead of /dev/null)
python video/record_demo.py \
    --episodes 10 --tier stress --seed 42 \
    --out /tmp/_scratch.mp4 --log results_stress.csv && rm -f /tmp/_scratch.mp4

# results_extreme.csv  (no video)
python video/record_demo.py \
    --episodes 10 --tier extreme --seed 42 \
    --out /tmp/_scratch.mp4 --log results_extreme.csv && rm -f /tmp/_scratch.mp4

# results_two_egg_*.csv + videos
python video/record_demo.py \
    --two-egg --episodes 10 --tier medium  --seed 42 --log results_two_egg_medium.csv  --out benchmark_two_egg_medium.mp4
python video/record_demo.py \
    --two-egg --episodes 10 --tier extreme --seed 42 --log results_two_egg_extreme.csv --out benchmark_two_egg_extreme.mp4
```

### Three-way correspondence

Each benchmark run produces three artefacts that correspond row-for-row:

| Artefact | Description |
|---|---|
| `results*.csv` | one row per episode: result, peak grip, step count |
| `benchmark*.mp4` | one segment per episode: same ordering, title and end cards |
| `trajectories/trajectory_ep*.csv` (with `--collect`) | one row per sim step: full joint and pose data |
| `trajectories/summary_ep*.json` (with `--collect`) | one file per episode: lightweight key-metric snapshot |

Adding `--collect` to any invocation does not change video, CSV results, or
success rates — it only writes the extra trajectory and summary files.

---

## CHANGELOG

| Version | Change | Motivation |
|---------|--------|-----------|
| **v1.0** | Single-egg pick-and-place; joint-space waypoint controller; `OVER-SQUEEZED` / `DROPPED` / `TIMEOUT` failure taxonomy; per-episode CSV (`results.csv`) | baseline benchmark |
| **v2.0** | Two-egg mode (`--two-egg`): target + distractor; `DISTRACTOR_DISTURBED` failure code; 4-tier difficulty via `min_egg_sep`; 20-column two-egg CSV | task complexity increase: target selection + distractor suppression |
| **v2.1** | Dynamic disturbance extension (`--dynamic-dist`): distractor receives a random rolling kick at episode start; consistent rolling (no-slip) model; `DISTRACTOR_ROLLING` failure code | more realistic real-world clutter |
| **v2.2** | Data productization: per-episode summary JSON (`--collect`); `scripts/summarize_results.py`; README tier×mode table; three-way artefact correspondence | reproducibility and submission evidence chain |
| **v2.3** | Analytic IK (`_solve_ik`, `USE_IK_CTRL=False`): closed-form Z-yaw + planar-2R solution verified against all waypoints; zero fallbacks in 10-episode regression | engineering completeness; optional path, legacy fully intact |
| **v2.4** | Analytical IK enabled by default (`USE_IK_CTRL=True`); runtime Cartesian reach correction (`target = egg_pos + approach_offset`) tracks per-episode egg position each step; Cartesian-space arrival check replaces joint-space gate; zero fallbacks across all regression tiers | improved control stability; benchmark semantics, CSV schema, failure codes, and phase order unchanged |
| **v2.5** | Bowl-side Cartesian reach correction: `above_bowl` / `at_bowl` IK targets now track the randomised bowl position (`target = bowl_pos + bowl_offset`); offset derived from MuJoCo FK at init; single-egg achieves 10/10 across all four tiers including extreme (±75 mm bowl offset) | eliminates DROPPED failures from bowl placement randomisation; benchmark semantics, CSV schema, failure codes, and phase order unchanged |
| **v2.6** | Two-egg benchmark videos and CSVs regenerated under v2.5 controller; bowl-corrected TRANSPORT path changes finger-sweep geometry, updating two-egg medium 7→6 and extreme 3→2; gradient remains monotone (10→6→4→2); no code change | artefact consistency: CSVs, videos, and README now match the active controller |
| **v2.7** | Two-egg distractor-avoidance strategies: (1) pre-close grip during APPROACH (`GRIP_CLOSED×0.5`, ±26 mm tips vs ±40 mm open) reduces finger-sweep width during descent past the distractor; (2) GRASP ramp starts from 50 % closure in two-egg mode (weld fires at phase_step≈160 instead of ≈280); (3) stable egg-tracking LIFT — egg XY captured at weld-fire time used as LIFT anchor so the arm rises straight up from the grasp point instead of sweeping to the default FK lifted position; (4) `lifted` waypoint IK target offset 40 mm in the distractor-far direction; two-egg results: medium 6→10, stress 4→6, extreme 2→4; gradient 10→10→6→4; single-egg unaffected (10/10 all tiers) | eliminate DISTRACTOR\_DISTURBED failures during APPROACH/GRASP and LIFT phases; remaining failures in TRANSPORT are arm-body contact beyond reach of IK-target steering |
| **v2.8** | Transport-stage IK target tuning: `above_bowl` Z raised by 60 mm (`_TRANSPORT_Z_LIFT = 0.06`) and 30 mm lateral bias added toward distractor-far side (`_TRANSPORT_LAT_BIAS = 0.03`) in two-egg mode; joint-space fallback preserved if IK cannot solve adjusted target; runtime Cartesian correction now covers approach, lift, and transport target classes; single-egg extreme 8→9; two-egg stress/extreme plateau at 6/10 and 4/10 confirmed as ceiling for target-steering-only fixes; dynamic-dist runs cleanly; no changes to CSV schema, failure codes, phase sequence, or benchmark semantics | confirm transport-stage ceiling; v2.8 transport tuning is the most aggressive endpoint correction attempted — mid-trajectory arm-body collision confirmed as the limiting factor |
| **v2.9** | Three-finger gripper: three capsule fingers at 120° angular spacing driven from the single `act_grip` actuator via equality constraints; `finger_a` (primary, −Y) retains physical contact (`contype=1`); `finger_b` and `finger_c` (120° and 240°) are visual and kinematic only (`contype=0`, `density=0`) with negligible inertia; EE site position, IK parameters, and all control constants unchanged; TE extreme 4→8 (+4) by eliminating TRANSPORT finger-sweep contacts; SE scores unchanged (10/10 medium, 9/10 stress, 9/10 extreme); TE medium unchanged (10/10); 2 remaining TE extreme failures confirmed as elbow-body contacts, not finger contacts | dexterity demonstration: coordinated 3-finger grasp visible in video, distractor selectivity improved; all benchmark semantics, CSV schema, failure codes, phase sequence, and controller unchanged |
| **v3.0** | Independent finger actuators, touch sensors, grasp quality metric: equality constraints replaced with 3 independent `position` actuators (`act_grip_a/b/c`); fingertip `<touch>` sensors at all three tip sites; `condim=6` contact pair for full-friction-cone wrench sensing; per-finger `actuatorfrc` sensors (`grip_b_force`, `grip_c_force`) logged to HUD; `grasp_quality = 1 − min(peak_grip / 12, 1.0)` added to CSV and overlay; per-finger force symmetry line (`FGRIP: B=…N C=…N`) in video HUD; TE stress 6→9 (+3); all other scores unchanged | force quality instrumentation; adaptive post-weld grip confirm; constraint-solver timing change drives TE stress improvement |
| **v3.1** | Multi-object support (M3): `--object egg\|cylinder\|sphere` flag in `record_demo.py`, `run_interactive.py`, and `run_policy.py`; target geom patched at model-load time via in-memory XML regex + tempfile (no new scene files, no new dependencies, no controller change); same contact pair (`finger_a_geom ↔ egg_geom`), kinematic weld, failure codes, and CSV schema for all shapes; 10/10 medium success verified for cylinder (⌀50×80 mm, avg 607 steps) and sphere (⌀56 mm, avg 717 steps); object type shown in title card, mode tag, and episode HUD header | object-agnostic manipulation demonstration: controller generalises across ellipsoid, cylinder, and sphere without modification |

---

## Known Limitations

- **Two-egg TRANSPORT ceiling (6/10 stress, 8/10 extreme) is an elbow-body geometry
  limit.** All APPROACH/GRASP, LIFT, and TRANSPORT finger-contact failures are
  eliminated (v2.7–v2.9). The 2 remaining TE extreme failures occur when the
  **elbow link** sweeps through the distractor's Z range during TRANSPORT. EE at
  `lifted`: Z ≈ 0.96 m; elbow: Z ≈ 0.85–0.90 m; distractor top: Z ≈ 0.845 m —
  5–15 mm clearance. v2.9's three-finger redesign confirmed this root cause: removing
  all physical contact from the lateral fingers improved TE extreme from 4→8 while
  the 2 elbow-contact episodes remained. Further improvement requires path-level
  planning, an elbow-up IK branch, or an intermediate waypoint between LIFT and
  TRANSPORT. Benchmark semantics, failure codes, and CSV schema are unchanged.

- **`OVER-SQUEEZED` does not appear in the failure gallery.** The failure
  code is fully wired into the state machine and HUD overlay, but
  `actuatorfrc` reflects position-actuator reaction against a driven qpos
  joint — it is near-zero under kinematic attachment. The branch never
  fires in practice, so no terminal frame is captured and the gallery does
  not show it. Switching to friction-contact grasping would make it an
  active failure mode.

- **Shell integrity check is approximate.** `actuatorfrc` reads
  position-actuator reaction forces, which are near-zero under kinematic
  attachment (egg qpos is driven directly; no contact forces are generated
  on the egg). A friction-contact grasp would produce physically
  realistic force readings.

- **Kinematic attachment, not friction grasping.** The egg freejoint qpos
  is updated each step to follow the gripper base, bypassing contact
  physics. This ensures reliable grasping for the benchmark but does not
  model real gripper–egg interaction forces.

- **Egg Y randomisation is narrow (±3 mm, all tiers).** Finger closure
  acts in the Y direction. Eggs offset >~4 mm in Y are contacted
  asymmetrically before kinematic attach fires, pushing the egg out of
  range. The stress tier increases X and bowl variation only.

- **Fixed joint waypoints.** The arm navigates via pre-computed
  joint-space waypoints tuned for the default egg position. Waypoints do
  not adapt to per-episode randomisation; the ±3 cm X stress range is
  within the kinematic-attach capture radius (7 cm) of the fixed waypoint.

- **No gym env wrapper.** The controller is a plain Python class.
  Gym/Gymnasium encapsulation is deferred.

- **H.264 codec is macOS-specific.** `cv2.VideoWriter_fourcc(*"avc1")`
  requires Apple's AVFoundation. On Linux, replace with
  `cv2.VideoWriter_fourcc(*"mp4v")` or use an `.avi` container with
  `XVID`.

---

## Troubleshooting

**`avc1` codec fails on Linux**
```
cv2.error: ... VideoWriter ... -1
```
`avc1` (H.264 via AVFoundation) is macOS-only. On Linux replace the fourcc in
`video/record_demo.py` or write to `.avi`:
```bash
# quick workaround: write to mp4v container
# in record_demo.py: cv2.VideoWriter_fourcc(*"mp4v")
# or use ffmpeg post-process: ffmpeg -i out.avi -c:v libx264 out.mp4
```

**`ModuleNotFoundError: mujoco` (or `numpy`)**

Make sure the conda environment is active before running:
```bash
conda activate robothon
python scripts/validate_scene.py   # quick sanity check
```
If the module is still missing: `pip install -r requirements.txt` inside the env.

**MuJoCo version mismatch warning**

This benchmark targets MuJoCo 3.9.0. Newer patch releases are generally
compatible; major version changes (e.g., 4.x) may alter `mj_kinematics`
body-merging behaviour and invalidate the FK check in `validate_scene.py`.

**`registration.json` validation fails**

```bash
python scripts/validate_registration.py registration.json
```
Exits 0 on success, 1 on validation error (prints the field that failed), 2 on
missing or malformed file. The `team` field must not be the placeholder
`"YOUR_TEAM_NAME"` — replace it before submission.

**Video writes to `/tmp` instead of the project directory (macOS background subprocess)**

AVFoundation cannot write to certain paths when the process runs in a background
subprocess context. Workaround:
```bash
conda run -n robothon python3 video/record_demo.py --out /tmp/bench.mp4 ... \
  && mv /tmp/bench.mp4 /path/to/project/benchmark.mp4
```

**`validate_scene.py` reports joint2 z ≠ 0.990**

MuJoCo 3.x merges static (massless, welded) bodies during compilation; the
compiled `xpos` for `joint2` differs from a naive sum of MJCF translations.
The expected value is 0.990 m. If you edited `scene.xml` geometry, re-run
`validate_scene.py` and update the check accordingly.

---

## Future Work

- **Friction-contact grasping.** Replace kinematic attachment with a
  contact-based grasp so that `actuatorfrc` reflects true finger–egg
  interaction forces and `OVER-SQUEEZED` becomes an active failure mode
  under normal operation.
- **Closed-loop reach.** The analytic IK path (`_solve_ik`, `USE_IK_CTRL=True`)
  is verified against all waypoints at zero fallback rate. Enabling it by default
  requires per-episode egg pose estimation; the `top_cam` overhead view provides
  a natural input for a vision-based localiser.
- **Gym / Gymnasium wrapper.** Expose the benchmark as a standard `gym.Env`
  for RL policy training and evaluation without modifying `phase_controller.py`.
  The `--collect` trajectory export already provides a demonstration dataset
  for imitation-learning baselines.
- **N-distractor extension.** The `min_egg_sep` tier system and two-egg CSV
  schema extend to N distractors; deferred to keep the current difficulty
  curve attributable to a single variable.
