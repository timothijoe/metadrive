"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import argparse
import random

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE

if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        manual_control=True,
        traffic_density=0.1,
        environment_num=10,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        map=20,  # seven block
        start_seed=random.randint(0, 1000)
    )
    parser = argparse.ArgumentParser()
    #parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
    parser.add_argument("--observation", type=str, default="birdview", choices=["lidar", "rgb_camera", "birdview"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(dict(offscreen_render=True))
    env = MetaDriveEnv(config)
    try:
        o = env.reset()
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("The observation is an numpy array with shape: ", o.shape)
            i = 0
        for j in range(1, 1000000000):
            i += 1
            # print(env.action_type.actions_indexes["LANE_LEFT"])

            
            # if i < 10:
            #     action_zt = env.action_type.actions_indexes["Holdon"]
            if i %2 == 0:
                action_zt = env.action_type.actions_indexes["IDLE"]
            elif (i+1) % 4 == 0:
                action_zt = env.action_type.actions_indexes["LANE_LEFT"]
            else:
                action_zt = env.action_type.actions_indexes["LANE_RIGHT"]

            #action_zt = env.action_type.actions_indexes["LANE_LEFT"] if i % 2 ==0 else env.action_type.actions_indexes["LANE_RIGHT"]
            o, r, d, info = env.step(action_zt)
            print(' i = {}'.format(i))
            # print('o.shape: {}'.format(o.shape))
            #print(o)
            #o, r, d, info = env.zt_step(env.action_type.actions_indexes["LANE_RIGHT"])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
            if d or info["arrive_dest"]:
                env.reset()
                i = 0
                env.current_track_vehicle.expert_takeover = True
    except:
        pass
    finally:
        env.close()
