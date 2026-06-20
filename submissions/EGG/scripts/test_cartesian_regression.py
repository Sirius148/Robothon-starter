"""Minimal headless regression: 3 episodes × single-egg medium. No video, no CSV write."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from controller.phase_controller import PhaseController, Phase, USE_CARTESIAN_CTRL

SCENE = os.path.join(os.path.dirname(__file__), "../models/scene.xml")
_EGG_DEFAULT  = np.array([0.26,  0.0,  0.800])
_BOWL_DEFAULT = np.array([0.08,  0.22, 0.760])
TIER = dict(egg_x=0.020, egg_y=0.003, egg_rot=np.radians(15), bowl_xy=0.015)

MAX_STEPS = 20_000
N_EPS     = 3
SEED      = 42


def run_episode(model, data, rng):
    # bowl randomisation
    bowl_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl")
    bdx, bdy = rng.uniform(-TIER["bowl_xy"], TIER["bowl_xy"], 2)
    model.body_pos[bowl_bid] = _BOWL_DEFAULT + [bdx, bdy, 0.0]

    ctrl = PhaseController(model, data)
    ctrl.reset()

    edx    = rng.uniform(-TIER["egg_x"], TIER["egg_x"])
    edy    = rng.uniform(-TIER["egg_y"], TIER["egg_y"])
    dtheta = rng.uniform(-TIER["egg_rot"], TIER["egg_rot"])
    adr    = ctrl.egg_qpos_adr
    data.qpos[adr:adr+3]   = _EGG_DEFAULT + [edx, edy, 0.0]
    data.qpos[adr+3:adr+7] = [np.cos(dtheta/2), 0.0, 0.0, np.sin(dtheta/2)]
    mujoco.mj_forward(model, data)

    last_phase = None
    for step in range(MAX_STEPS):
        phase, fail = ctrl.step()
        mujoco.mj_step(model, data)
        if phase != last_phase:
            print(f"    step={step:5d} → {phase.name}")
            last_phase = phase
        if phase in (Phase.DONE, Phase.FAIL):
            return phase, fail, step + 1
        if step % 500 == 499:
            dbg = getattr(ctrl, '_cart_debug', {})
            q2 = data.qpos[ctrl._arm_jnt_qposadr[1]]
            q3 = data.qpos[ctrl._arm_jnt_qposadr[2]]
            err_val = dbg.get('err', float('nan'))
            print(f"    step={step+1:5d} phase={phase.name} err={err_val:.4f} "
                  f"q2={q2:.3f} q3={q3:.3f}")
    return Phase.FAIL, None, MAX_STEPS


def main():
    print(f"USE_CARTESIAN_CTRL = {USE_CARTESIAN_CTRL}")
    model = mujoco.MjModel.from_xml_path(SCENE)
    rng   = np.random.default_rng(SEED)

    results = []
    for ep in range(1, N_EPS + 1):
        data  = mujoco.MjData(model)
        phase, fail, steps = run_episode(model, data, rng)
        ok = (phase == Phase.DONE)
        results.append(ok)
        status = "SUCCESS" if ok else f"FAIL ({fail})"
        print(f"  ep{ep}: {status}  steps={steps}")

    n_ok = sum(results)
    print(f"\nResult: {n_ok}/{N_EPS} success")
    if n_ok < N_EPS:
        print("REGRESSION: expected 3/3")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
