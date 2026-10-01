def load_stage1_joint_trainer_config(*args, **kwargs):
    from breast_pretrain.train.stage1_joint.config import load_stage1_joint_trainer_config as _load

    return _load(*args, **kwargs)


def override_stage1_joint_trainer_config(*args, **kwargs):
    from breast_pretrain.train.stage1_joint.config import override_stage1_joint_trainer_config as _override

    return _override(*args, **kwargs)


def run_stage1_joint_trainer(*args, **kwargs):
    from breast_pretrain.train.stage1_joint.trainer import run_stage1_joint_trainer as _run

    return _run(*args, **kwargs)

__all__ = [
    "load_stage1_joint_trainer_config",
    "override_stage1_joint_trainer_config",
    "run_stage1_joint_trainer",
]
