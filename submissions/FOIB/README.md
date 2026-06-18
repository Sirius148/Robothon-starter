# FOIB-Egg: Fragile Object Integrity Benchmark

A MuJoCo 3 simulation benchmark that evaluates whether a 3-DOF robot arm
can pick and place an egg without compromising shell integrity.
**Shell integrity — monitored via continuous gripper force sensing — is the
primary scoring criterion.** Pick-and-place is the delivery mechanism;
force compliance throughout the grasp is the benchmark objective.

The benchmark runs headless, generates reproducible video from code, and
logs per-episode results to CSV. No manual annotation is required.

---

## Problem

Fragile object manipulation requires balancing two competing constraints:
grip must be firm enough to lift and carry the object, yet gentle enough
not to damage it. Standard pick-and-place benchmarks measure *placement
accuracy* but ignore *grasp force compliance*. FOIB-Egg makes force
compliance an explicit, scoreable criterion alongside placement.

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

---

## Failure Taxonomy

| Code | Trigger | Phase |
|------|---------|-------|
| `OVER-SQUEEZED` | `actuatorfrc` sensor exceeds 12 N | GRASP → TRANSPORT |
| `DROPPED` | egg Z falls below 0.79 m during lift/transport, or egg misses bowl after release | LIFT, TRANSPORT, CHECK |
| `TIMEOUT` | phase exceeds 2000 simulation steps (~4 s) without meeting exit condition | any phase |

The `SHELL` overlay indicator encodes these states in every video frame:

| Overlay label | Meaning |
|---|---|
| `--` | pre-grasp; monitoring not yet active |
| `CLOSING` | fingers moving in; force monitoring begins |
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

---

## Project Structure

```
models/
  scene.xml            # MJCF scene: table, egg (freejoint), bowl, 3-DOF arm
  arm_bodies.xml       # arm fragment: links, parallel gripper, ee site
controller/
  phase_controller.py  # phase state machine — no RL, no gym dependency
scripts/
  validate_scene.py    # headless smoke test: 500 steps, NaN check
  run_interactive.py   # passive MuJoCo viewer for manual inspection
video/
  overlay.py           # per-frame HUD: EP/TIER/PHASE/OBS/SHELL/GRIP/STATUS
  record_demo.py       # recorder: single demo or tiered multi-episode benchmark
requirements.txt
```

---

## Setup

```bash
conda create -n robothon python=3.11
conda activate robothon
pip install -r requirements.txt
```

---

## Running

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
--dynamic-dist`). Each episode exports one CSV to `--collect-dir`:

```
trajectories/
├── trajectory_ep001.csv
├── trajectory_ep002.csv
...
```

Every row is one simulation step. Columns: `step, episode_id, tier, phase,
ep_result, j1_pos, j2_pos, j3_pos, ee_x/y/z, egg_x/y/z, bowl_x/y/z,
grip_force, contact_count, grasped`. Two-egg mode appends `egg2_x/y/z,
egg2_disp_mm, distractor_rolling, wrong_object_contact`. The last row of
each file carries the true `ep_result`; all earlier rows have `ep_result=""`.

`--collect` is purely additive: video, benchmark CSV, and success rates are
unchanged.

### Interactive viewer
```bash
python scripts/run_interactive.py
```

---

## Included Videos

Three representative recordings are included in the repository root.

**`benchmark.mp4`** — single-egg, medium tier, 10 episodes, seed 42.
All 10 episodes succeed (10/10). Demonstrates baseline pick-and-place behaviour,
shell-force monitoring, and the overlay HUD. Serves as the backward-compatibility
reference: every metric and CSV column shown here is unchanged in v2.

**`benchmark_two_egg_medium.mp4`** — two-egg, 75 mm separation, 10 episodes,
seed 42. 7/10 success. The primary v2 showcase: the orange distractor egg
appears on the table throughout each episode, and 3 episodes end in
DISTRACTOR\_DISTURBED or DROPPED when the gripper sweep clips it. Represents
the intended operating point where the benchmark discriminates without saturating.

**`benchmark_two_egg_extreme.mp4`** — two-egg, 66 mm separation, 10 episodes,
seed 42. 3/10 success. Shows the full failure taxonomy in a single recording:
DISTRACTOR\_DISTURBED (egg1 placed, distractor knocked away), DROPPED (egg1
lost because the distractor obstructed the approach corridor), and clean SUCCESS
in the 3 episodes where the random placement puts egg2 outside the gripper sweep.
Most informative for evaluating the two failure modes in isolation.

---

## Benchmark Results

Results are deterministic for a fixed seed. Verified on macOS, MuJoCo 3.9.0.

### Medium tier — seed 42, 10 episodes

| Metric | Value |
|--------|-------|
| Shell INTACT | 10 / 10 (100 %) |
| OVER-SQUEEZED | 0 |
| Dropped / Timeout | 0 |
| Peak grip (max) | 0.370 N |
| Contact max (max) | 2 |
| Avg steps / ep | 709 |
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
| Avg steps / ep | 709 |
| Video duration | 74.0 s |
| Wall time | ~25 s |

### Extreme tier — seed 42, 10 episodes

| Metric | Value |
|--------|-------|
| Shell INTACT | 7 / 10 (70 %) |
| OVER-SQUEEZED | 0 |
| Dropped / Timeout | 3 (EP 7, 8, 9 — DROPPED) |
| Peak grip (max) | 0.370 N |
| Contact max (max) | 4 |
| Avg steps / ep | 709 |
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
fixed seed and episode count.

---

## Known Limitations

- **Shell integrity check is approximate.** `actuatorfrc` reads
  position-actuator reaction forces, which are near-zero under kinematic
  attachment (egg qpos is driven directly; no contact forces are generated
  on the egg). The `OVER-SQUEEZED` branch is fully wired into the state
  machine and overlay, but does not trigger under normal operation with
  this grasping mode. A friction-contact grasp would produce physically
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
