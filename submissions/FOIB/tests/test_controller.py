"""Smoke tests for PhaseController: reset, step, overlay, and shell-shard safety."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

SCENE = os.path.join(os.path.dirname(__file__), "../models/scene.xml")


@pytest.fixture(scope="module")
def ctrl_fixture():
    import mujoco
    from controller.phase_controller import PhaseController

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    ctrl = PhaseController(model, data)
    ctrl.reset()
    return ctrl, model, data


def test_reset_no_raise(ctrl_fixture):
    ctrl, _, _ = ctrl_fixture
    ctrl.reset()


def test_step_returns_phase_and_fail(ctrl_fixture):
    from controller.phase_controller import FailReason, Phase

    ctrl, _, _ = ctrl_fixture
    ctrl.reset()
    phase, fail = ctrl.step()
    assert isinstance(phase, Phase)
    assert isinstance(fail, FailReason)


def test_grasp_quality_range(ctrl_fixture):
    ctrl, _, _ = ctrl_fixture
    ctrl.reset()
    assert 0.0 <= ctrl.grasp_quality <= 1.0


def test_overlay_info_required_keys(ctrl_fixture):
    ctrl, _, _ = ctrl_fixture
    ctrl.reset()
    info = ctrl.overlay_info()
    required = (
        "phase", "step", "grip_force", "grasped", "egg_z",
        "dist_bowl", "gates", "grasp_quality", "finger_touch",
    )
    for key in required:
        assert key in info, f"overlay_info missing key: {key}"


def test_shards_underground_after_reset(ctrl_fixture):
    import mujoco

    ctrl, model, data = ctrl_fixture
    ctrl.reset()
    for i in range(8):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"shard_{i}")
        if bid >= 0:
            assert data.xpos[bid][2] < 0.0, f"shard_{i} not underground after reset"


def test_egg_qpos_adr_is_joint0(ctrl_fixture):
    ctrl, model, _ = ctrl_fixture
    assert ctrl.egg_qpos_adr == model.jnt_qposadr[0]
