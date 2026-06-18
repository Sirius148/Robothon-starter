"""
Waypoint-based state machine controller for Egg Pick-and-Place benchmark.

Phases: IDLE → APPROACH → GRASP → LIFT → TRANSPORT → LOWER → RELEASE → CHECK → DONE/FAIL

Each phase drives joints toward a pre-computed joint-space waypoint.
Gate conditions are checked before transitioning to the next phase.
"""
import numpy as np
from enum import Enum, auto


class Phase(Enum):
    IDLE       = auto()
    RETRACT    = auto()   # arm retracts high before approaching egg
    APPROACH   = auto()
    GRASP      = auto()
    LIFT       = auto()
    TRANSPORT  = auto()
    LOWER      = auto()
    RELEASE    = auto()
    CHECK      = auto()
    DONE       = auto()
    FAIL       = auto()


class FailReason(Enum):
    NONE          = ""
    OVER_SQUEEZED = "OVER-SQUEEZED"
    DROPPED       = "DROPPED"
    TIMEOUT       = "TIMEOUT"


# ----- Thresholds --------------------------------------------------------
GRIP_OPEN        = 0.0    # ctrl value → fingers fully open  (joint=0 → ±40mm from center)
GRIP_CLOSED      = 0.020  # ctrl value → fingers grip egg    (inner gap ≈ 24mm, egg ≈ 22mm Y)
GRIP_FORCE_MIN   = 0.3    # N  — minimum grip force to count as grasped
GRIP_FORCE_MAX   = 12.0   # N  — over this = over-squeezed → FAIL
JOINT_THRESH     = 0.05   # rad — per-joint error to call waypoint "reached"
LIFT_HEIGHT_CHK  = 0.855  # m  — egg must be above this z before TRANSPORT
PLACE_THRESH     = 0.08   # m  — egg XY dist to bowl center to declare success
PHASE_TIMEOUT    = 2000   # sim steps per phase (~4 s at dt=0.002)

# Pre-computed joint waypoints [joint1, joint2, joint3] in radians
# Derived by DLS-IK for: arm mount x=0, egg at [0.26,0,0.80], bowl at [0.08,0.22,0.778]
_WP = {
    "retract":    np.array([ 0.000,  -1.200,  0.600]),  # arm high, clear of egg
    "above_egg":  np.array([ 0.000,  -0.404,  1.397]),  # ee ≈ [0.26, 0, 0.89]
    "at_egg":     np.array([ 0.000,  -0.129,  1.221]),  # ee ≈ [0.26, 0, 0.82]
    "lifted":     np.array([ 0.000,  -0.710,  1.516]),  # ee ≈ [0.26, 0, 0.96]
    "above_bowl": np.array([ 1.222,  -0.494,  1.595]),  # ee ≈ [0.08, 0.22, 0.90]
    "at_bowl":    np.array([ 1.222,  -0.122,  1.338]),  # ee ≈ [0.08, 0.22, 0.82]
}

# Maximum rate at which joint ctrl targets are allowed to change (trapezoidal profile).
# At dt=0.002 s this gives 0.1 rad/step; largest waypoint gap (j1 1.222 rad) ramps
# in ~13 steps — well within PHASE_TIMEOUT=2000.
JOINT_MAX_VEL = 50.0  # rad/s


class PhaseController:
    def __init__(self, model, data):
        import mujoco
        self.m  = model
        self.d  = data
        self.mj = mujoco

        def bid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        def sid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        def aid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        def snid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)

        self.egg_id    = bid("egg")
        self.ee_site   = sid("ee")
        self.bowl_site = sid("bowl_center")

        self.act_j1    = aid("act_j1")
        self.act_j2    = aid("act_j2")
        self.act_j3    = aid("act_j3")
        self.act_grip  = aid("act_grip")
        self.sen_grip  = snid("grip_force")

        # Kinematic grasp: track gripper_base body id and egg freejoint qpos address
        self.gb_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        self.egg_qpos_adr = model.jnt_qposadr[0]   # freejoint is joint 0
        self._weld_active     = False
        self._grasp_local_pos = np.zeros(3)   # egg center offset in gripper_base local frame

        # qpos addresses for joint1/2/3
        self._arm_jnt_qposadr = [
            model.jnt_qposadr[j] for j in range(model.njnt)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            in ("joint1", "joint2", "joint3")
        ]
        self._arm_acts = [self.act_j1, self.act_j2, self.act_j3]
        self._ctrl_cmd = np.zeros(3)   # smoothed ctrl target for arm joints

        self.phase       = Phase.IDLE
        self.fail_reason = FailReason.NONE
        self.phase_steps = 0
        self.step_count  = 0
        self._bowl_pos   = None
        self._peak_grip  = 0.0
        self._egg_geom_ids = frozenset(
            i for i in range(model.ngeom)
            if model.geom_bodyid[i] == self.egg_id
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        self.mj.mj_resetData(self.m, self.d)
        # Teleport arm to retract pose so it never sweeps through the egg
        for i, addr in enumerate(self._arm_jnt_qposadr):
            self.d.qpos[addr] = _WP["retract"][i]
        self.mj.mj_forward(self.m, self.d)

        self.phase       = Phase.IDLE
        self.fail_reason = FailReason.NONE
        self.phase_steps = 0
        self.step_count  = 0
        self._bowl_pos        = self.d.site_xpos[self.bowl_site].copy()
        self._weld_active     = False
        self._grasp_local_pos = np.zeros(3)
        self._peak_grip       = 0.0
        self._ctrl_cmd        = _WP["retract"].copy()   # avoid ramp from zeros
        self._drive_to_waypoint(_WP["retract"])
        self._set_grip(GRIP_OPEN)

    def step(self):
        """Call once per sim step. Returns (phase, fail_reason)."""
        self.step_count  += 1
        self.phase_steps += 1

        grip_force = abs(self.d.sensordata[self.sen_grip])
        self._peak_grip = max(self._peak_grip, grip_force)
        egg_pos    = self.d.xpos[self.egg_id].copy()
        bowl_pos   = self._bowl_pos

        # Kinematic attachment: move egg with gripper every step when grasped
        if self._weld_active:
            self._kinematic_attach()

        # Global over-squeeze guard
        if grip_force > GRIP_FORCE_MAX and self.phase not in (Phase.DONE, Phase.FAIL,
                                                               Phase.IDLE, Phase.RETRACT,
                                                               Phase.APPROACH):
            self._fail(FailReason.OVER_SQUEEZED)
            return self.phase, self.fail_reason

        if self.phase == Phase.IDLE:
            self._transition(Phase.RETRACT)

        elif self.phase == Phase.RETRACT:
            self._drive_to_waypoint(_WP["retract"])
            self._set_grip(GRIP_OPEN)
            if self._at_waypoint(_WP["retract"]):
                self._transition(Phase.APPROACH)
            elif self.phase_steps > PHASE_TIMEOUT:
                self._fail(FailReason.TIMEOUT)

        elif self.phase == Phase.APPROACH:
            self._drive_to_waypoint(_WP["above_egg"])
            self._set_grip(GRIP_OPEN)
            if self._at_waypoint(_WP["above_egg"]):
                self._transition(Phase.GRASP)
            elif self.phase_steps > PHASE_TIMEOUT:
                self._fail(FailReason.TIMEOUT)

        elif self.phase == Phase.GRASP:
            self._drive_to_waypoint(_WP["at_egg"])
            t = min(self.phase_steps / 400.0, 1.0)
            self._set_grip(GRIP_CLOSED * t)
            ee_pos = self.d.site_xpos[self.ee_site]
            ee_to_egg = np.linalg.norm(ee_pos - egg_pos)
            if not self._weld_active and ee_to_egg < 0.07 and t >= 0.7:
                self._activate_kinematic_grasp()
            if self._weld_active and self._at_waypoint(_WP["at_egg"]):
                self._transition(Phase.LIFT)
            elif self.phase_steps > PHASE_TIMEOUT:
                self._fail(FailReason.DROPPED)

        elif self.phase == Phase.LIFT:
            self._drive_to_waypoint(_WP["lifted"])
            self._set_grip(GRIP_CLOSED)
            if egg_pos[2] > LIFT_HEIGHT_CHK and self._at_waypoint(_WP["lifted"]):
                self._transition(Phase.TRANSPORT)
            elif egg_pos[2] < 0.79 and self.phase_steps > 200:
                self._fail(FailReason.DROPPED)
            elif self.phase_steps > PHASE_TIMEOUT:
                self._fail(FailReason.TIMEOUT)

        elif self.phase == Phase.TRANSPORT:
            self._drive_to_waypoint(_WP["above_bowl"])
            self._set_grip(GRIP_CLOSED)
            horiz = np.linalg.norm(egg_pos[:2] - bowl_pos[:2])
            if self._at_waypoint(_WP["above_bowl"]) and horiz < 0.14:
                self._transition(Phase.LOWER)
            elif egg_pos[2] < 0.79 and self.phase_steps > 150:
                self._fail(FailReason.DROPPED)
            elif self.phase_steps > PHASE_TIMEOUT:
                self._fail(FailReason.TIMEOUT)

        elif self.phase == Phase.LOWER:
            self._drive_to_waypoint(_WP["at_bowl"])
            self._set_grip(GRIP_CLOSED)
            if self._at_waypoint(_WP["at_bowl"]):
                self._transition(Phase.RELEASE)
            elif self.phase_steps > PHASE_TIMEOUT:
                self._fail(FailReason.TIMEOUT)

        elif self.phase == Phase.RELEASE:
            self._set_grip(GRIP_OPEN)
            if self.phase_steps == 1:
                self._release_kinematic_grasp()
            if self.phase_steps > 120:
                self._transition(Phase.CHECK)

        elif self.phase == Phase.CHECK:
            horiz = np.linalg.norm(egg_pos[:2] - bowl_pos[:2])
            if horiz < PLACE_THRESH and egg_pos[2] > bowl_pos[2] - 0.02:
                self._transition(Phase.DONE)
            else:
                self._fail(FailReason.DROPPED)

        return self.phase, self.fail_reason

    def overlay_info(self):
        grip_force = abs(self.d.sensordata[self.sen_grip])
        egg_pos    = self.d.xpos[self.egg_id]
        bowl_pos   = self._bowl_pos if self._bowl_pos is not None else np.zeros(3)
        horiz      = np.linalg.norm(egg_pos[:2] - bowl_pos[:2])
        grasped    = self._weld_active
        gates = {
            "GRASP": grasped,
            "LIFT":  egg_pos[2] > LIFT_HEIGHT_CHK,
            "PLACE": horiz < PLACE_THRESH,
            "SAFE":  grip_force < GRIP_FORCE_MAX,
        }
        n_contacts = sum(
            1 for i in range(self.d.ncon)
            if (self.d.contact[i].geom1 in self._egg_geom_ids or
                self.d.contact[i].geom2 in self._egg_geom_ids)
        )
        return {
            "phase":         self.phase.name,
            "step":          self.step_count,
            "phase_step":    self.phase_steps,
            "grip_force":    round(float(grip_force), 2),
            "grasped":       grasped,
            "egg_z":         round(float(egg_pos[2]), 3),
            "dist_bowl":     round(float(horiz), 3),
            "fail_reason":   self.fail_reason.value,
            "gates":         gates,
            "peak_grip":     round(float(self._peak_grip), 3),
            "contact_count": n_contacts,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition(self, new_phase):
        self.phase       = new_phase
        self.phase_steps = 0

    def _fail(self, reason):
        self.fail_reason = reason
        self._transition(Phase.FAIL)

    def _set_grip(self, val):
        self.d.ctrl[self.act_grip] = float(np.clip(val, 0.0, 0.035))

    def _activate_kinematic_grasp(self):
        """Record egg offset in gripper_base frame, then enable kinematic tracking."""
        gb_mat = self.d.xmat[self.gb_id].reshape(3, 3)
        dp     = self.d.xpos[self.egg_id] - self.d.xpos[self.gb_id]
        self._grasp_local_pos = gb_mat.T @ dp   # offset in gripper_base local frame
        self._weld_active     = True

    def _kinematic_attach(self):
        """Each step: set egg freejoint qpos so egg follows gripper_base."""
        gb_mat  = self.d.xmat[self.gb_id].reshape(3, 3)
        gb_quat = self.d.xquat[self.gb_id].copy()
        target  = self.d.xpos[self.gb_id] + gb_mat @ self._grasp_local_pos
        adr = self.egg_qpos_adr
        self.d.qpos[adr:adr+3] = target
        self.d.qpos[adr+3:adr+7] = gb_quat          # match gripper orientation
        self.d.qvel[0:6] = 0.0                        # kill freejoint velocity

    def _release_kinematic_grasp(self):
        """Give the egg a gentle downward initial velocity and release."""
        self._weld_active = False
        self.d.qvel[0:3] = [0.0, 0.0, -0.2]   # gentle downward push at release

    def _drive_to_waypoint(self, target_q):
        """Ramp joint ctrl targets toward target_q at most JOINT_MAX_VEL rad/s."""
        lims      = self.m.actuator_ctrlrange
        max_delta = JOINT_MAX_VEL * self.m.opt.timestep
        for i, act in enumerate(self._arm_acts):
            clamped = float(np.clip(target_q[i], lims[act, 0], lims[act, 1]))
            self._ctrl_cmd[i] += float(np.clip(clamped - self._ctrl_cmd[i],
                                               -max_delta, max_delta))
            self.d.ctrl[act] = self._ctrl_cmd[i]

    def _at_waypoint(self, target_q):
        """True when all arm joints are within JOINT_THRESH of target."""
        for i, addr in enumerate(self._arm_jnt_qposadr):
            if abs(self.d.qpos[addr] - target_q[i]) > JOINT_THRESH:
                return False
        return True
