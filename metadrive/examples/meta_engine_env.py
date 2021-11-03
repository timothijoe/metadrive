from typing import Any, Union, List
import copy
import torch
import numpy as np
import gym 
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo, update_shape
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import ENV_REGISTRY
from metadrive import MetaDriveEnv
from ding.envs import ObsNormEnv, RewardNormEnv 

MetaDrive_INFO_DICT = {
    'MetaDrive-test-v0': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(3, 200, 200),
            value={
                # udpate in 8.11 night, maybe 0, 1
                # 'min': np.float64("-inf"),
                # 'max': np.float64("inf"),
                'min': 0.,
                'max': 1.,
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(5, ),
            value={
                'min': 0,
                'max': 5.0,
                'dtype': np.int
            },
        ),
        rew_space=EnvElementInfo(
            shape=1,
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf")
            },
        ),
        use_wrappers=None,
    ),
}
@ENV_REGISTRY.register('metadrive')
class dingMetaDriveEnv(BaseEnv):

    def __init__(self, cfg: dict, meta_env_cfg: dict = None) -> None:
        self._cfg = cfg  
        self._env_cfg = meta_env_cfg
        self._init_flag = False

    def _make_env(self) -> gym.Env:
        norm_obs = self._cfg.get('norm_obs', False)
        norm_reward = self._cfg.get('norm_reward', False)
        env = MetaDriveEnv(self._env_cfg)
        if norm_obs:
            env = ObsNormEnv(env)
        if norm_reward:
            env = RewardNormEnv(env)
        return env 

    def reset(self) -> torch.FloatTensor:
        if not self._init_flag:
            self._env = self._make_env()
            self._init_flag = True 
        obs = self._env.reset()
        obs = to_ndarray(obs).astype('float32')
        self._final_eval_reward = 0
        #obs = obs.transpose((2, 0, 1))
        return obs 

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False 

    def step(self, action: Union[torch.Tensor, np.ndarray, list]) -> BaseEnvTimestep:
        action = to_ndarray(action)
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs).astype('float32')
        #obs = obs.transpose((2, 0, 1))
        rew = to_ndarray([rew])
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def __repr__(self) -> str:
        return "DI-engine PGDrive Env({})".format(self._cfg.env_id)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def info(self) -> BaseEnvInfo:
        if self._cfg.env_id in MetaDrive_INFO_DICT:
            info = copy.deepcopy(MetaDrive_INFO_DICT[self._cfg.env_id])
            info.use_wrappers = ' '
            obs_shape, act_shape, rew_shape = update_shape(
                info.obs_space.shape, info.act_space.shape, info.rew_space.shape, info.use_wrappers.split('\n')
            )
            info.obs_space.shape = obs_shape
            info.act_space.shape = act_shape
            info.rew_space.shape = rew_shape
            return info
        else:
            raise NotImplementedError('{} not found in PGDrive_INFO_DICT [{}]'\
                .format(self._cfg.env_id, MetaDrive_INFO_DICT.keys()))

    

