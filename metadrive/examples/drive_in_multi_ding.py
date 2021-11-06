from ding.config import compile_config
from ding.envs import BaseEnvManager, DingEnvWrapper, AsyncSubprocessEnvManager
from ding.envs.env_manager import SyncSubprocessEnvManager
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.worker import BaseLearner, SampleCollector, BaseSerialEvaluator, AdvancedReplayBuffer, NaiveReplayBuffer
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config
import gym
import metadrive
from easydict import EasyDict
from ding.utils import set_pkg_seed
from tensorboardX import SummaryWriter
import os
from ding.rl_utils import get_epsilon_greedy_fn
cartpole_dqn_config = dict(
    exp_name='cartpole_dqn_multi',
    env=dict(
        collector_env_num=2,
        evaluator_env_num=1,
        n_evaluator_episode=5,
        stop_value=195,
        # manager=dict(
        #     reset_timeout = 600,
        # )
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=[3,200,200],
            action_shape=5,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=8),
        eval=dict(evaluator=dict(eval_freq=50, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
        ),
    ),
)
cartpole_dqn_config = EasyDict(cartpole_dqn_config)

def wrapped_cartpole_env():
    return DingEnvWrapper(gym.make('Meta-v1'))


def main(cfg, seed=0):
    cfg = compile_config(
        cfg,
        AsyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleCollector,
        BaseSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = AsyncSubprocessEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    #evaluator_env = AsyncSubprocessEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
    print('zt')

    #Set random seed for all package and instance
    collector_env.seed(seed)
    #evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # # Set up RL Policy
    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

    # Set up collection, training and evaluation utilities
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    # evaluator = BaseSerialEvaluator(
    #     cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    # )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    # Set up other modules, etc. epsilon greedy
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    # Training & Evaluation loop
    while True:
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
    main(cartpole_dqn_config)






# if __name__ == "__main__":
#     collector_env_num, evaluator_env_num = 2,1
#     collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
#     evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
#     print('zt')
