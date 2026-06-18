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

# ---- Cartesian controller -----------------------------------------------
# Set USE_CARTESIAN_CTRL=False to revert to legacy joint-space control at any time.
USE_CARTESIAN_CTRL = False
CART_KP        = 0.05   # IK step gain (dimensionless) — dq_cmd = KP × J^+(e); ctrl = qpos + dq_cmd
CART_KD        = 0.0    # velocity damping coefficient (kept 0: see investigation notes)
CART_LAMBDA    = 0.05   # DLS damping factor — regularises J near singularities
CART_DQ_MAX    = 0.020  # rad  — max ctrl deviation from qpos per step (uniform-scaled)
CART_EE_TOL    = 0.015  # m    — Cartesian arrival threshold (replaces JOINT_THRESH)
CART_VEL_TOL   = 0.5    # rad/s — max arm-joint velocity before Cartesian IK may start
# Weighted DLS: boost j3 to correct J[z,j2]/J[z,j3] ratio (1.0 = disabled)
CART_J3_WEIGHT = 1.0    # joint-space weight for j3 in weighted DLS

# ---- Analytic IK controller ------------------------------------------------
# Enabled by USE_IK_CTRL=True; USE_CARTESIAN_CTRL must remain False.
# Structure: Z-yaw (j1) + planar 2R (j2/j3).  MuJoCo 3.9.x merges the static
# "base" body into arm_mount at compile time, so joint2 world-z = 0.990, not
# 0.860+0.18=1.040 as a naive MJCF read would suggest.  Value confirmed via FK.
USE_IK_CTRL   = False
_IK_H_J2      = 0.990    # joint2 world-z (constant for all j1)
_IK_L2        = 0.220    # link2 length (j2 → j3)
_IK_L3EFF     = 0.190    # link3 (0.18) + ee x-offset (0.01) along arm axis
_IK_DOFF      = 0.052    # ee perpendicular offset below arm axis
_IK_L3C       = np.sqrt(_IK_L3EFF**2 + _IK_DOFF**2)   # 0.1970 m — effective ee distance
_IK_PSI       = np.arctan2(_IK_DOFF, _IK_L3EFF)        # 0.2689 rad — ee phase angle
_IK_JLIM_TOL  = 0.001    # rad — margin when checking solved angles against joint limits


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

        # velocity-DOF addresses for joint1/2/3 (Jacobian column indices)
        self._arm_jnt_veladr = [
            model.jnt_dofadr[j] for j in range(model.njnt)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            in ("joint1", "joint2", "joint3")
        ]
        # Cartesian controller state (populated in reset)
        self._cart_wps   = {}   # WP name → ee Cartesian position from FK
        self._cart_debug = {}   # last-step: {"err": float, "cond": float}
        self._legacy_fallback_count = 0
        # Analytic IK controller state (populated in reset)
        self._ik_wps          = {}   # WP name → ee Cartesian position from FK
        self._ik_fallback_cnt = 0

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
        if USE_CARTESIAN_CTRL:
            self._cart_wps = self._compute_cart_waypoints()
        if USE_IK_CTRL:
            self._ik_wps = self._compute_cart_waypoints()
            self._ik_fallback_cnt = 0
        self._legacy_fallback_count = 0
        self._cart_debug = {}
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
        """Dispatch to analytic IK, Cartesian DLS, or legacy joint-space controller.

        Priority: USE_IK_CTRL > USE_CARTESIAN_CTRL > legacy.
        Analytic IK applies to all phases (no phase exclusion needed: it computes
        a fixed joint target, not incremental steps, so no warm-start dependency).
        """
        if USE_IK_CTRL and self._ik_wps:
            self._drive_ik(target_q)
        elif USE_CARTESIAN_CTRL and self._cart_wps and self.phase not in (Phase.RETRACT, Phase.APPROACH):
            self._drive_cartesian(target_q)
        else:
            self._drive_legacy(target_q)

    def _drive_legacy(self, target_q):
        """Rate-limited joint-space position commands (legacy fallback)."""
        lims      = self.m.actuator_ctrlrange
        max_delta = JOINT_MAX_VEL * self.m.opt.timestep
        for i, act in enumerate(self._arm_acts):
            clamped = float(np.clip(target_q[i], lims[act, 0], lims[act, 1]))
            self._ctrl_cmd[i] += float(np.clip(clamped - self._ctrl_cmd[i],
                                               -max_delta, max_delta))
            self.d.ctrl[act] = self._ctrl_cmd[i]

    # ------------------------------------------------------------------
    # Analytic IK methods
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_ik(px, py, pz):
        """Closed-form analytic IK for Z-yaw + planar-2R arm with ee offset.

        Derivation: j1 decouples by yaw symmetry.  The 2R planar sub-problem
        is solved by combining the ee perpendicular offset (_IK_DOFF) into an
        equivalent link length _IK_L3C at phase angle _IK_PSI, then applying
        the law of cosines.  Elbow-down branch (j3 > 0) is always chosen.

        Returns np.array([j1, j2, j3]) or None if:
          - target sits on/near the j1 axis (j1 undefined, r < 1 mm)
          - outside reachable workspace (|D| > 1)
        Joint-limit checks are done by the caller (_drive_ik).
        """
        r = np.hypot(px, py)
        if r < 1e-3:
            return None                   # target on j1 axis — yaw undefined
        j1 = np.arctan2(py, px)
        Zp = _IK_H_J2 - pz              # positive = ee below joint2 height
        D  = (r*r + Zp*Zp - _IK_L2*_IK_L2 - _IK_L3C*_IK_L3C) / (2.0*_IK_L2*_IK_L3C)
        if abs(D) > 1.0:
            return None                   # outside reachable workspace
        phi3 = np.arctan2(np.sqrt(max(0.0, 1.0 - D*D)), D)   # elbow-down (phi3 > 0)
        j2   = np.arctan2(Zp, r) - np.arctan2(_IK_L3C * np.sin(phi3),
                                                _IK_L2  + _IK_L3C * np.cos(phi3))
        j3   = phi3 - _IK_PSI
        return np.array([j1, j2, j3])

    def _drive_ik(self, target_q):
        """Analytic IK path: solve closed-form joint target, drive via legacy.

        Fallback conditions (each increments _ik_fallback_cnt and uses legacy):
          1. target_q not found in _ik_wps (unknown waypoint)
          2. _solve_ik returns None (unreachable or on j1 axis)
          3. Any solved joint angle exceeds its actuator ctrlrange
        When fallback fires, the original joint-space target_q is used directly
        (same behaviour as USE_IK_CTRL=False).
        """
        wp_name = next((n for n, q in _WP.items() if np.allclose(q, target_q)), None)
        if wp_name is None or wp_name not in self._ik_wps:
            self._drive_legacy(target_q)
            return

        q_ik = self._solve_ik(*self._ik_wps[wp_name])
        if q_ik is None:
            self._ik_fallback_cnt += 1
            self._drive_legacy(target_q)
            return

        lims = self.m.actuator_ctrlrange
        for i, act in enumerate(self._arm_acts):
            lo, hi = lims[act, 0], lims[act, 1]
            if q_ik[i] < lo - _IK_JLIM_TOL or q_ik[i] > hi + _IK_JLIM_TOL:
                self._ik_fallback_cnt += 1
                self._drive_legacy(target_q)
                return

        self._drive_legacy(q_ik)

    def _drive_cartesian(self, target_q):
        """Cartesian P-control: DLS J^+ × Kp × pos_error → incremental joint delta."""
        wp_name = next((n for n, q in _WP.items() if np.allclose(q, target_q)), None)
        if wp_name is None or wp_name not in self._cart_wps:
            self._legacy_fallback_count += 1
            self._drive_legacy(target_q)
            return

        cart_target = self._cart_wps[wp_name]
        x_curr      = self.d.site_xpos[self.ee_site].copy()
        e           = cart_target - x_curr
        err_norm    = float(np.linalg.norm(e))

        # At target: hold current joint positions to suppress integrator drift.
        if err_norm < 1e-4:
            lims   = self.m.actuator_ctrlrange
            q_curr = np.array([self.d.qpos[a] for a in self._arm_jnt_qposadr])
            for i, act in enumerate(self._arm_acts):
                clamped           = float(np.clip(q_curr[i], lims[act, 0], lims[act, 1]))
                self.d.ctrl[act]  = clamped
                self._ctrl_cmd[i] = clamped
            self._cart_debug = {"err": err_norm, "cond": 0.0}
            return

        # Translational Jacobian: shape (3, nv) → extract arm columns → 3×3
        jacp = np.zeros((3, self.m.nv))
        self.mj.mj_jacSite(self.m, self.d, jacp, None, self.ee_site)
        J = jacp[:, self._arm_jnt_veladr]

        # DLS pseudoinverse: J^+(e) gives the IK joint-space step (rad) that
        # would move ee by ≈ e under the linearisation.  Scale by CART_KP so
        # ctrl leads qpos by CART_KP × J^+(e) rad — arm decelerates naturally
        # as e → 0, preventing overshoot without requiring velocity integration.
        JJT = J @ J.T + (CART_LAMBDA ** 2) * np.eye(3)
        try:
            dq_ik = J.T @ np.linalg.solve(JJT, e)   # rad — pure IK step
        except np.linalg.LinAlgError:
            self._legacy_fallback_count += 1
            self._drive_legacy(target_q)
            return

        sv   = np.linalg.svd(J, compute_uv=False)
        cond = float(sv[0] / (sv[-1] + 1e-12))
        self._cart_debug = {"err": err_norm, "cond": cond}

        # Uniform-scale to respect per-joint ctrl deviation limit.
        # Uniform (not element-wise) scaling preserves Cartesian direction.
        dq_step = CART_KP * dq_ik
        max_abs = float(np.max(np.abs(dq_step)))
        if max_abs > CART_DQ_MAX:
            dq_step = dq_step * (CART_DQ_MAX / max_abs)
        q_curr = np.array([self.d.qpos[a] for a in self._arm_jnt_qposadr])
        q_new  = q_curr + dq_step

        lims = self.m.actuator_ctrlrange
        for i, act in enumerate(self._arm_acts):
            clamped           = float(np.clip(q_new[i], lims[act, 0], lims[act, 1]))
            self.d.ctrl[act]  = clamped
            self._ctrl_cmd[i] = clamped

    def _compute_cart_waypoints(self):
        """Run FK at each joint-space waypoint; cache ee Cartesian positions."""
        saved = np.array([self.d.qpos[a] for a in self._arm_jnt_qposadr])
        result = {}
        for name, q in _WP.items():
            for i, a in enumerate(self._arm_jnt_qposadr):
                self.d.qpos[a] = q[i]
            self.mj.mj_forward(self.m, self.d)
            result[name] = self.d.site_xpos[self.ee_site].copy()
        for i, a in enumerate(self._arm_jnt_qposadr):
            self.d.qpos[a] = saved[i]
        self.mj.mj_forward(self.m, self.d)
        return result

    def _at_waypoint(self, target_q):
        """Arrival check: Cartesian ee distance in Cartesian mode, joint error in legacy.

        For APPROACH: also gates on arm velocity < CART_VEL_TOL so the Cartesian IK
        controller starts from a nearly-stationary state (not mid-swing).
        """
        if USE_CARTESIAN_CTRL and self._cart_wps:
            wp_name = next((n for n, q in _WP.items() if np.allclose(q, target_q)), None)
            if wp_name and wp_name in self._cart_wps:
                x_curr = self.d.site_xpos[self.ee_site]
                if float(np.linalg.norm(x_curr - self._cart_wps[wp_name])) >= CART_EE_TOL:
                    return False
                if self.phase == Phase.APPROACH:
                    arm_vel = max(abs(self.d.qvel[a]) for a in self._arm_jnt_veladr)
                    if arm_vel > CART_VEL_TOL:
                        return False
                return True
        for i, addr in enumerate(self._arm_jnt_qposadr):
            if abs(self.d.qpos[addr] - target_q[i]) > JOINT_THRESH:
                return False
        return True
