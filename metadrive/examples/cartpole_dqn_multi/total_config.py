exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 5,
            'step_timeout': 60,
            'auto_reset': True,
            'reset_timeout': 600,
            'retry_waiting_time': 0.1,
            'shared_memory': False,
            'context': 'fork',
            'wait_num': 2,
            'step_wait_timeout': 0.01,
            'connect_timeout': 60,
            'cfg_type': 'AsyncSubprocessEnvManagerDict'
        },
        'collector_env_num': 1,
        'evaluator_env_num': 1,
        'n_evaluator_episode': 5,
        'stop_value': 195
    },
    'policy': {
        'model': {
            'obs_shape': [3, 200, 200],
            'action_shape': 5,
            'encoder_hidden_size_list': [128, 128, 64],
            'dueling': True
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'multi_gpu': False,
            'update_per_collect': 3,
            'batch_size': 64,
            'learning_rate': 0.001,
            'target_update_freq': 100,
            'ignore_done': False
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleCollectorDict'
            },
            'unroll_len': 1,
            'n_sample': 8
        },
        'eval': {
            'evaluator': {
                'eval_freq': 50,
                'cfg_type': 'BaseSerialEvaluatorDict',
                'stop_value': 195,
                'n_episode': 5
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 20000,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict'
            },
            'eps': {
                'type': 'exp',
                'start': 0.95,
                'end': 0.1,
                'decay': 10000
            }
        },
        'type': 'dqn',
        'cuda': False,
        'on_policy': False,
        'priority': False,
        'priority_IS_weight': False,
        'discount_factor': 0.97,
        'nstep': 1,
        'cfg_type': 'DQNPolicyDict'
    },
    'exp_name': 'cartpole_dqn_multi',
    'seed': 0
}
