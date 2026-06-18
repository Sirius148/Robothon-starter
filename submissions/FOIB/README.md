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
| Difficulty gradient | `min_egg_sep` drives 10 → 7 → 4 → 3 success rate; controller and thresholds never modified |
| Rolling disturbance | Physically consistent kick (`ω = v / R_eff`, no-slip); distinct `DISTRACTOR_ROLLING` failure code |
| Trajectory export | `--collect` writes per-step joint + pose + force data for downstream policy training |
| Dual-camera video | `--dual-cam` composites side view (upper ⅔) + overhead view (lower ⅓) in one frame |
| Reproducible by default | Deterministic `--seed 42`; seven included CSVs; bit-identical across runs |

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
                       # analytic IK optional path (USE_IK_CTRL), default off; legacy intact
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
| `equality / joint` (finger mirror) | `scene.xml` | right finger mirrors left via `polycoef="0 1 0 0 0"` — one actuator drives both fingers |
| `actuatorfrc` sensor | `scene.xml` | reads position-actuator reaction force for shell integrity gate |
| `site` (ee) | `arm_bodies.xml` | end-effector tracking point; queried every step via `data.site_xpos` |
| `mj_jacSite` | `phase_controller.py` | translational Jacobian for optional DLS Cartesian path |
| multiple cameras | `scene.xml` | `side_cam` (default) + `top_cam`; selectable via `--camera` flag |
| `mj_kinematics` | `phase_controller.py` | FK at reset to cache Cartesian waypoints for IK/DLS paths |
| `mj_contactForce` | `record_demo.py` | reads per-contact normal force at episode terminal state; logged as `contacts@end` for post-hoc analysis |

### Collision filtering (contype / conaffinity)

The two eggs and the arm links use independent `contype` / `conaffinity` bitmasks
to control which geom pairs generate contacts:

| Geom | `contype` | `conaffinity` | Collides with |
|------|:---------:|:-------------:|---------------|
| `egg_geom` (target) | 2 (bit 1) | 1 | table, bowl, fingers — **not** egg2 |
| `egg2_geom` (distractor) | 4 (bit 2) | 1 | table, bowl, fingers — **not** egg |
| arm links (visual) | 0 | 0 | nothing |
| fingers | 1 | 1 | both eggs, table |
| table / bowl | 1 | 7 (bits 0–2) | both eggs, fingers |

Two geoms A and B collide when `(A.contype & B.conaffinity) ≠ 0` **or**
`(B.contype & A.conaffinity) ≠ 0`. Because egg (`contype=2`) and egg2 (`contype=4`)
share no bits with each other's conaffinity (`=1`), the two eggs never generate
mutual contact — they can occupy close proximity without numerical collision
artefacts while still interacting correctly with the table, bowl, and gripper.

### Equality constraint (finger mirror)

The gripper has two `slide` joints (`finger_left_joint`, `finger_right_joint`) but
only one actuator (`act_grip`). The `<equality><joint>` constraint with
`polycoef="0 1 0 0 0"` enforces `q_right = q_left` at every timestep, so a single
position command symmetrically closes both fingers. This eliminates a control
degree of freedom without requiring a second actuator or a custom constraint solver.

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
python video/record_demo.py \
    --episodes 10 --tier stress --seed 42 \
    --out benchmark_stress.mp4 --log results_stress.csv
```

### Benchmark — extreme tier
```bash
python video/record_demo.py \
    --episodes 10 --tier extreme --seed 42 \
    --out benchmark_extreme.mp4 --log results_extreme.csv
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
| `benchmark.mp4` | Single-egg (mode 1) | Phase subtitle cycling RETRACT → … → CHECK; SHELL overlay `CLOSING → OK → INTACT`; 10/10 baseline |
| `benchmark_two_egg_medium.mp4` | Two-egg static (mode 2), 75 mm sep | Distractor robustness at intended operating point (7/10); failure gallery: `DISTRACTOR_DISTURBED` + `DROPPED` |
| `benchmark_two_egg_extreme.mp4` | Two-egg static (mode 2), 66 mm sep | Full failure taxonomy across 7 failures; overhead view shows why proximity drives the drop from 7 → 3 |

<!-- Representative frames: ffmpeg -ss 20 -i benchmark.mp4 -frames:v 1 docs/frame_single.png -->

| File | Episodes | Result |
|------|----------|--------|
| `benchmark.mp4` | single-egg, medium, seed 42 | **10 / 10** |
| `benchmark_two_egg_medium.mp4` | two-egg static, 75 mm sep, seed 42 | **7 / 10** |
| `benchmark_two_egg_extreme.mp4` | two-egg static, 66 mm sep, seed 42 | **3 / 10** |

`benchmark.mp4` is the v1 baseline; its CSV schema and scoring criteria are unchanged in v2.

---

## Difficulty Gradient

The two-egg difficulty curve is controlled by a **single physical variable**,
`min_egg_sep` — the minimum centre-to-centre distance between the two eggs at
episode start. All other parameters (controller, thresholds, randomisation ranges)
are identical across tiers.

```
Tier      easy      medium    stress    extreme
Sep       80 mm     75 mm     70 mm     66 mm
Success   10/10 ██████████  7/10 ███████  4/10 ████  3/10 ███
```

Performance degrades monotonically as the distractor enters the arm's finger-sweep
corridor. At 80 mm the sweep never reaches egg2; at 66 mm it does in 7 of 10
randomly seeded placements. Failures split into two physically distinct modes:
geometry-induced disturbance (`DISTRACTOR_DISTURBED`) and blocked approach leading
to a missed grasp (`DROPPED`). No controller modification is required to traverse
the full gradient.

---

## Benchmark Results

Results are deterministic for a fixed seed. Verified on macOS, MuJoCo 3.9.0.

### Tier × Mode Overview — seed 42, 10 episodes each

| Mode | Tier | N | Success | DIST\_DISTURBED | DROPPED/TIMEOUT | Peak grip |
|---|---|:-:|---|:-:|:-:|---|
| single | medium | 10 | **10/10** (100 %) | — | 0 | 0.36 N |
| single | stress | 10 | **9/10** (90 %) | — | 1 | 0.41 N |
| single | extreme | 10 | **8/10** (80 %) | — | 2 | 0.39 N |
| two-egg (static) | easy | 10 | **10/10** (100 %) | 0 | 0 | — |
| two-egg (static) | medium | 10 | **7/10** (70 %) | 2 | 1 | — |
| two-egg (static) | stress | 10 | **4/10** (40 %) | 4 | 2 | — |
| two-egg (static) | extreme | 10 | **3/10** (30 %) | 5 | 2 | — |

The monotonic gradient across both dimensions confirms that tier and mode are
independent axes of difficulty: single-egg tests geometric precision; two-egg
separation tests spatial selectivity. All rows can be regenerated exactly with
`--seed 42` using the commands in the [Reproducibility](#reproducibility) section.

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
| Shell INTACT | 8 / 10 (80 %) |
| OVER-SQUEEZED | 0 |
| Dropped / Timeout | 2 (EP 5, 7 — DROPPED) |
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
the 10→7→4→3 gradient a clean single-variable ablation.

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
grip_peak, contact_max, steps`), and success criteria. v2 is an additive
extension: no existing test path, constant, or threshold was modified.

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

### Results — seed 42, 10 episodes per tier

| Tier | min\_egg\_sep | Success | DIST\_DISTURBED | DROPPED | wrong\_contact eps |
|------|:------------:|:-------:|:---------------:|:-------:|:-----------------:|
| easy | 80 mm | **10 / 10** | 0 | 0 | 0 |
| medium | 75 mm | **7 / 10** | 2 | 1 | 3 |
| stress | 70 mm | **4 / 10** | 4 | 2 | 6 |
| extreme | 66 mm | **3 / 10** | 5 | 2 | 7 |

The monotonic curve (10 → 7 → 4 → 3) confirms that `min_egg_sep` is the
controlling variable. At 80 mm the arm's finger sweep never reaches the
distractor; at 66 mm it does in 7 of 10 randomly seeded placements. Failures
split into two physically distinct modes:

- **Precision failure** (DISTRACTOR\_DISTURBED): egg1 is placed correctly but
  the gripper contacts the distractor during approach or transport, knocking it
  outside the stability threshold. The task objective is met; the scene
  constraint is not.
- **Task failure** (DROPPED): the distractor obstructs the approach corridor,
  deflecting egg1 before kinematic attachment fires, causing a failed grasp or
  a missed bowl. Both the task objective and scene constraint fail.

All failures emerge from geometry; the controller is never modified.

### Two-egg CSV schema

```
episode_id, tier, result,
target_success, wrong_object_contact, wrong_object_grasp,
distractor_displacement_mm, distractor_stable,
target_pick_success, target_place_success, target_final_dist_to_bowl_mm,
steps, peak_grip, contact_count,
target_egg_id, distractor_egg_id
```

3-egg extension is possible but left for future work to avoid conflating density
scaling with the current benchmark design.

### Dynamic Disturbance Extension (`--dynamic-dist`)

This is an **extension beyond the static-placement benchmark**, not a
replacement. The static two-egg results (10→7→4→3 gradient) remain the
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

# results_stress.csv
python video/record_demo.py \
    --episodes 10 --tier stress --seed 42 \
    --out benchmark_stress.mp4 --log results_stress.csv

# results_extreme.csv
python video/record_demo.py \
    --episodes 10 --tier extreme --seed 42 \
    --out benchmark_extreme.mp4 --log results_extreme.csv

# results_two_egg_*.csv + benchmark_two_egg_medium.mp4
python video/record_demo.py \
    --two-egg --episodes 10 --tier easy    --seed 42 --log results_two_egg_easy.csv    --out /dev/null
python video/record_demo.py \
    --two-egg --episodes 10 --tier medium  --seed 42 --log results_two_egg_medium.csv  --out benchmark_two_egg_medium.mp4
python video/record_demo.py \
    --two-egg --episodes 10 --tier stress  --seed 42 --log results_two_egg_stress.csv  --out /dev/null
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

---

## Known Limitations

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
