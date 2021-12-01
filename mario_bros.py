from pettingzoo.atari import mario_bros_v2
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import supersuit as ss
from torch import nn


class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                [8, 8],
                stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                [4, 4],
                stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(
                64,
                64,
                [3, 3],
                stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136,512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator(args):
    env = mario_bros_v2.parallel_env(auto_rom_install_path="./autorom")
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.dtype_v0(env, 'float32')
    env = ss.frame_stack_v1(env, 3)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)

    return env


def to_keras(model_path: str):
    import onnx
    from onnx2keras import onnx_to_keras

    # Load ONNX model
    onnx_model = onnx.load(model_path)

    # Call the converter (input - is the main model input name, can be different for your model)
    k_model = onnx_to_keras(onnx_model, ['obs', 'state_ins'])

    k_model.save("./onnx/keras.h5")


def find_best_checkpoint(env_name: str):
    analysis = tune.Analysis("./ray_results/" + env_name)  # can also be the result of `tune.run()`

    trial_logdir = analysis.get_best_logdir(metric="episode_reward_max",
                                            mode="max")  # Can also just specify trial dir directly

    return analysis.get_best_checkpoint(trial_logdir, metric="episode_reward_max", mode="max")


def onnx(env_name: str, config: dict[str, object]):
    agent = PPOTrainer(config=config,
                       env=env_name)
    agent.restore(checkpoint_path=find_best_checkpoint(env_name))

    agent.export_policy_model("./onnx", policy_id="policy_0", onnx=14)

    to_keras('./onnx/model.onnx')


if __name__ == "__main__":
    env_name = "mario_bros"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    test_env = ParallelPettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "CNNModelV2",
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)

    policies = {"policy_0": gen_policy(0)}

    policy_ids = list(policies.keys())

    config = {
        # Environment specific
        "env": env_name,
        # General
        "log_level": "ERROR",
        "framework": "torch",
        "num_gpus": 1,
        "num_workers": 2,
        "num_envs_per_worker": 2,
        "compress_observations": False,
        "batch_mode": 'truncate_episodes',
        "record_env": "videos",
        "render_env": True,

        # 'use_critic': True,
        # 'use_gae': True,
        # "lambda": 0.9,

        "gamma": .99,

        # "kl_coeff": 0.001,
        # "kl_target": 1000.,
        "clip_param": 0.4,
        'grad_clip': None,
        "entropy_coeff": 0.1,
        'vf_loss_coeff': 0.25,

        "sgd_minibatch_size": 64,
        "num_sgd_iter": 10,  # epoc
        'rollout_fragment_length': 512,
        "train_batch_size": 512 * 4,
        'lr': 2e-05,
        "clip_actions": True,

        # Method specific
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (
                lambda agent_id: policy_ids[0]),
        },
    }

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=5,
        local_dir="./ray_results/"+env_name,
        config=config,
    )