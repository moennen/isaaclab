# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab_tasks.utils import PresetCfg

STATE_POLICY_CFG = RslRlMLPModelCfg(
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    activation="elu",
)


STATE_CRITIC_CFG = RslRlMLPModelCfg(
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    activation="elu",
)

CNN_POLICY_CFG = RslRlCNNModelCfg(
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
    cnn_cfg=RslRlCNNModelCfg.CNNCfg(
        output_channels=[16, 32],
        kernel_size=[3, 3],
        activation="elu",
        max_pool=[True, True],
        norm="batch",
        global_pool="avg",
    ),
    activation="elu",
)


ALGO_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class Dexsuite3dgKukaAllegroPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 250
    experiment_name = (MISSING,)  # type: ignore
    obs_groups = (MISSING,)  # type: ignore
    actor = (MISSING,)  # type: ignore
    critic = (MISSING,)  # type: ignore
    algorithm = MISSING  # type: ignore


@configclass
class Dexsuite3dgKukaAllegroPPORunnerCfg(PresetCfg):
    default = Dexsuite3dgKukaAllegroPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_3dg_kuka_allegro",
        obs_groups={"actor": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]},
        actor=STATE_POLICY_CFG,
        critic=STATE_CRITIC_CFG,
        algorithm=ALGO_CFG,
    )

    single_camera = Dexsuite3dgKukaAllegroPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_3dg_kuka_allegro_single_camera",
        obs_groups={"actor": ["policy", "proprio", "base_image"], "critic": ["policy", "proprio", "perception"]},
        actor=CNN_POLICY_CFG,
        critic=STATE_CRITIC_CFG,
        algorithm=ALGO_CFG.replace(num_mini_batches=16),
    )

    duo_camera = Dexsuite3dgKukaAllegroPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_3dg_kuka_allegro_duo_camera",
        obs_groups={
            "actor": ["policy", "proprio", "base_image", "wrist_image"],
            "critic": ["policy", "proprio", "perception"],
        },
        actor=CNN_POLICY_CFG,
        critic=STATE_CRITIC_CFG,
        algorithm=ALGO_CFG.replace(num_mini_batches=16),
    )

    finetune = Dexsuite3dgKukaAllegroPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_3dg_kuka_allegro_finetune",
        obs_groups={"actor": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]},
        actor=STATE_POLICY_CFG,
        critic=STATE_CRITIC_CFG,
        # Adjusted for low env count (≤ 8 envs, episode = 360 steps at 60 Hz).
        #
        # With 8 envs, num_steps_per_env=360 gives 2880 samples per rollout — one
        # full episode of diversity.  A single mini-batch avoids the instability of
        # 64-sample batches that arise when num_mini_batches=4 at this scale.
        # max_iterations=5000 yields ~14.4 M env steps, comparable to ~110 iterations
        # of the 4096-env default training (same policy-update frequency).
        num_steps_per_env=360,
        max_iterations=5000,
        algorithm=ALGO_CFG.replace(
            num_mini_batches=1,
            learning_rate=3e-4,
        ),
    )
    """Fine-tuning preset for environments with very few parallel envs (≤ 8).

    Use with ``--load_weights_only --resume`` to transfer a pre-trained policy to a
    new contact regime (e.g. ``env.sim.physics=simplicits_matched``)::

        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
            --task Isaac-Dexsuite-3dg-Kuka-Allegro-Reorient-v0 \\
            --num_envs 8 \\
            --resume --load_weights_only \\
            --load_run "2026-03-11_10-37-30" \\
            --checkpoint "model_14999.pt" \\
            env.sim.physics=simplicits_matched \\
            env.scene=simplicits_matched \\
            agent=finetune
    """
