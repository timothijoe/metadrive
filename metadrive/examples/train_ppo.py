import os
from typing import Dict
from ding import config
from ding.worker.collector import sample_serial_collector
from ding.worker.replay_buffer import naive_buffer
import gym
import metadrive
from metadrive import MetaDriveEnv
from tensorboardX import SummaryWriter
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, NaiveReplayBuffer, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed, deep_merge_dicts
from config_ppo import pgdrive_ppo_config
from meta_engine_env import dingMetaDriveEnv
# from PGDriveManager import PGDriveEnvManager
# from pgdrive_serial_evaluator import PGDriveSerialEvaluator
# from pgdrive_serial_collector import PGDriveSampleCollector
# from pgdriveEnv import dingPGDriveEnv
from ding.rl_utils import get_epsilon_greedy_fn
from ding.envs.env_manager import SyncSubprocessEnvManager
from ding.envs import AsyncSubprocessEnvManager

def wrapped_ppo_env():
   return gym.make("MetaDrive-test-v0")


def main(cfg, seed=0, max_iterations=int(1e10)):
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
    collector_env_num =3
    evaluator_env_num = 1
    collector_cfg = dict(use_render=True)
    evaluator_cfg = dict()
    collector_env = AsyncSubprocessEnvManager(
        env_fn=[lambda: dingMetaDriveEnv(cfg.env, collector_cfg) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    # evaluator_env = SyncSubprocessEnvManager(env_fn=[wrapped_ppo_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    # evaluator_env = AsyncSubprocessEnvManager(
    #     env_fn=[lambda: dingMetaDriveEnv(cfg.env, evaluator_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    # )
    collector_env.seed(cfg.env.collector_start_seed)
    #evaluator_env.seed(cfg.env.evaluator_start_seed,) # dynamic_seed=False)
    set_pkg_seed(cfg.env.pkg_seed, use_cuda=cfg.policy.cuda)

    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # bug here: manager cant set pgdrive-engine
    # collector = PGDriveSampleCollector(
    #     cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger
    # )
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger
    )
    # MUST HAVE !
    # collector.close()
    # evaluator = PGDriveSerialEvaluator(
    #     cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger
    # )
    # evaluator = InteractionSerialEvaluator(
    #     cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger
    # )

    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    # evaluator.close()
    # print(evaluator._policy)
    for iter in range(max_iterations):
        # if evaluator.should_eval(learner.train_iter):
        #     # evaluator.reset_env(evaluator_env)
        # stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
        # print(f'eva in {iter} iters, reward is {reward}')
            # if stop:
            #     break
        #     # evaluator.close()
        # # collector.reset_env(collector_env)
        # new_data = collector.collect(train_iter=learner.train_iter)
        # # print(new_data)
        # learner.train(new_data, collector.envstep)
        # collector.close()
        # Evaluating at the beginning and with specific frequency
        # if evaluator.should_eval(learner.train_iter):
        #     stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
        #     if stop:
        #         break
        # Update other modules
        eps = epsilon_greedy(collector.envstep)
        # Sampling data from environments
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Training
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                break
            learner.train(train_data, collector.envstep)


if __name__ == "__main__":
    main(pgdrive_ppo_config)
