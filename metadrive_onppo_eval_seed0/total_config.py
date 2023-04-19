exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 2,
            'retry_type': 'reset',
            'auto_reset': True,
            'step_timeout': None,
            'reset_timeout': None,
            'retry_waiting_time': 0.1,
            'cfg_type': 'BaseEnvManagerDict',
            'shared_memory': False,
            'context': 'spawn'
        },
        'stop_value': 255,
        'metadrive': {
            'use_render': False,
            'traffic_density': 0.1,
            'map': 'XSOS',
            'horizon': 4000,
            'driving_reward': 1.0,
            'speed_reward': 0.1,
            'use_lateral_reward': False,
            'out_of_road_penalty': 40.0,
            'crash_vehicle_penalty': 40.0,
            'decision_repeat': 20,
            'out_of_route_done': True,
            'show_bird_view': False
        },
        'n_evaluator_episode': 16,
        'collector_env_num': 1,
        'evaluator_env_num': 1
    },
    'policy': {
        'model': {
            'obs_shape': [5, 84, 84],
            'action_shape': 2,
            'action_space': 'continuous',
            'bound_type': 'tanh',
            'encoder_hidden_size_list': [128, 128, 64]
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'multi_gpu': False,
            'epoch_per_collect': 10,
            'batch_size': 64,
            'learning_rate': 0.0003,
            'value_weight': 0.5,
            'entropy_weight': 0.001,
            'clip_ratio': 0.02,
            'adv_norm': False,
            'value_norm': True,
            'ppo_param_init': True,
            'grad_clip_type': 'clip_norm',
            'grad_clip_value': 10,
            'ignore_done': False
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict'
            },
            'unroll_len': 1,
            'discount_factor': 0.99,
            'gae_lambda': 0.95,
            'n_sample': 1000
        },
        'eval': {
            'evaluator': {
                'eval_freq': 1000,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'n_episode': 16,
                'stop_value': 255
            }
        },
        'other': {
            'replay_buffer': {}
        },
        'type': 'ppo',
        'cuda': True,
        'on_policy': True,
        'priority': False,
        'priority_IS_weight': False,
        'recompute_adv': True,
        'action_space': 'continuous',
        'nstep_return': False,
        'multi_agent': False,
        'transition_with_policy_data': True,
        'cfg_type': 'PPOPolicyDict'
    },
    'exp_name': 'metadrive_onppo_eval_seed0',
    'seed': 0
}
