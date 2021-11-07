from os import environ
from ding.worker import replay_buffer
from easydict import EasyDict

metadrive_dqn_config = dict(
    exp_name='zt_multi_channel_try1',
    env=dict(
        env_id='MetaDrive-test-v0',
        norm_obs=False,
        norm_reward=False,
        #use_act_scale=True,
        collector_env_num=10,
        evaluator_env_num=5,
        n_evaluator_episode=20,
        stop_value=195,
        collector_start_seed=999,
        pkg_seed=1023,
    ),
    policy=dict(
        cuda=False,
        #on_policy=True,
        #continuous=True,
        model=dict(
            obs_shape=(5,200,200),
            action_shape=5,
            encoder_hidden_size_list=[128, 128, 64],
            # critic_head_hidden_size=256,
            # actor_head_hidden_size=256,
            # critic_hidden_size=256,
            # action_hidden_size=256,
            # if not, shape can be wrong !
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            #epoch_per_collect=20,
            batch_size=64,
            learning_rate=1e-3,
            update_per_collect=100,
            # hook=dict(
            #         load_ckpt_before_run='/home/SENSETIME/zhoutong/hoffnung/metadrive/ckpt/iteration_20000.pth.tar',
            #     ),
            #value_weight=0.5,
            #entropy_weight=0.01,
            #clip_ratio=0.2,
        ),
        collect=dict(
            # seems very imp 1024 can achieve 100+
            n_sample=1000,
            # unroll_len=1,
            # discount_factor=0.99,
            # gae_lambda=0.97,
            # collector=dict(
            #     transform_obs = True,
            # )
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(
                replay_buffer_size=10000,),
                #replay_buffer_start_size=0,),
        ), 
    ),
)
pgdrive_ppo_config = EasyDict(metadrive_dqn_config)
main_config = pgdrive_ppo_config

# use for env compile
pgdrive_ppo_create_config = dict(
    # env=dict(
    #     type='pgdrive',
    #     import_names=['pgdrive_env'],
    # ),
    # env_manager=dict(type='base'),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='ppo',
        import_names=['ding.policy.ppo'],
    ),
    replay_buffer=dict(type='naive', ),
)
pgdrive_ppo_create_config = EasyDict(pgdrive_ppo_create_config)
create_config = pgdrive_ppo_create_config

# test 
print(main_config)
