from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
)


@configclass
class VanillaPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10_100 # video saves after (interval + 20) so give it some buffer
    save_interval = 1_000
    experiment_name = "so101_test"
    run_name = ""
    logger = "wandb"
    wandb_project = "SO101_Test"
    store_code_state = True
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"], # update for asymmetric actor-critic
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    # do not touch without good reason:
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class ReachPPORunnerCfg(VanillaPPORunnerCfg):
    experiment_name = "so101_reach"
    wandb_project = "SO101_Reach"

    def __post_init__(self):
        # Less exploration noise for the actuator-matched arm. With std 1.0 most samples already
        # land at or past the +-1 action clip, and the 0.01 entropy bonus kept rewarding a wider
        # spread even though it only produced random full-force flailing (std rose 1 -> 2 in the
        # first actuator-matched run). Start narrower and reward width less -- but not too little:
        # at 0.002 the std collapsed 0.5 -> 0.09 within 200 iterations, before the policy had
        # found the target, and it settled on folding the arm upright instead.
        self.policy.init_noise_std = 0.5
        self.algorithm.entropy_coef = 0.005