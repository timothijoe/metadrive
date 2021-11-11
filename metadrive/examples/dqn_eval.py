from typing import Union, Optional, List, Any, Callable, Tuple
import pickle
import torch
from functools import partial

from ding.config import compile_config, read_config
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.policy import DQNPolicy
from ding.model import DQN
from ding.envs.env_manager import AsyncSubprocessEnvManager
from meta_engine_env import dingMetaDriveEnv
from config_ppo import pgdrive_ppo_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, NaiveReplayBuffer, AdvancedReplayBuffer
def eval(
        cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
) -> float:
    r"""
    Overview:
        Pure evaluation entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
    """
    cfg = compile_config(
        cfg,
        AsyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator, 
        AdvancedReplayBuffer, 
        save_cfg=True
    )
    evaluator_cfg = dict(use_render=False)

    env = AsyncSubprocessEnvManager(
        env_fn=[lambda: dingMetaDriveEnv(cfg.env, evaluator_cfg) for _ in range(1)], cfg=cfg.env.manager
    )
    #env = env_fn(evaluator_env_cfg[0])
    env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)




    policy = create_policy(cfg.policy, model=model, enable_field=['eval']).eval_mode
    if state_dict is None:
        state_dict = torch.load(cfg.learner.load_path, map_location='cpu')
    policy.load_state_dict(state_dict)

    obs = env.reset()
    eval_reward = 0.
    while True:
        policy_output = policy.forward({0: obs})
        action = policy_output[0]['action']
        print(action)
        timestep = env.step(action)
        eval_reward += timestep.reward
        obs = timestep.obs
        if timestep.done:
            print(timestep.info)
            break

    env.save_replay(replay_dir='.', prefix=env._map_name)
    print('Eval is over! The performance of your RL policy is {}'.format(eval_reward))


if __name__ == "__main__":
    path = '/home/SENSETIME/zhoutong/hoffnung/metadrive/ckpt/iteration_30000.pth.tar'
    cfg = '../config/smac_MMM_qmix_config.py'
    state_dict = torch.load(path, map_location='cpu')
    eval(pgdrive_ppo_config, seed=0, state_dict=state_dict)