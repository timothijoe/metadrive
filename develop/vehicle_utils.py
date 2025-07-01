from panda3d.core import LineSegs, NodePath
from panda3d.core import Material, Vec3, LVecBase4
from metadrive.component.pg_space import VehicleParameterSpace, ParameterSpace
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import Semantics
from metadrive.engine.asset_loader import AssetLoader

from metadrive.component.vehicle.vehicle_type import DefaultVehicle
import numpy as np
from metadrive.utils import Config, safe_clip_for_small_array
from typing import Callable, Optional, Union, List, Dict, AnyStr
import copy
from collections import deque
from develop.navigation_utils import HRLNodeNavigation


class MacroDefaultVehicle(DefaultVehicle):

    def __init__(self, vehicle_config: Union[dict, Config] = None, name: str = None, random_seed=None):
        super(MacroDefaultVehicle, self).__init__(vehicle_config, name, random_seed)
        self.macro_succ = False
        self.macro_crash = False
        self.last_spd = 0
        self.last_macro_position = self.last_position
        self.v_wps = [[0,0], [1,1]]
        self.v_indx = 1
        self.physics_world_step_size = self.engine.global_config["physics_world_step_size"]
        self.penultimate_state = {}
        self.penultimate_state['position'] = np.array([0,0]) #self.last_position
        self.penultimate_state['yaw'] = 0 
        self.penultimate_state['speed'] = 0
        self.traj_wp_list = [] 
        self.traj_wp_list.append(copy.deepcopy(self.penultimate_state))
        self.traj_wp_list.append(copy.deepcopy(self.penultimate_state))
        self.taecrl_max_spd = 10.0
        self.taecrl_max_acc = 5.0 #2.5
        self.taecrl_max_steer = 0.5
        self.vis_state = np.zeros(6)

    def before_macro_step(self, macro_action):
        if macro_action ==0:
            self.last_macro_position = self.position
        else:
            pass
        return

    def reset(
        self,
        vehicle_config=None,
        name=None,
        random_seed=None,
        position: np.ndarray = None,
        heading: float = 0.0,
        *args,
        **kwargs
    ):
        """
        pos is a 2-d array, and heading is a float (unit degree)
        if pos is not None, vehicle will be reset to the position
        else, vehicle will be reset to spawn place
        """
        if name is not None:
            self.rename(name)

        # reset fully
        self.update_config(self.engine.global_config["vehicle_config"])
        if random_seed is not None:
            assert isinstance(random_seed, int)
            self.seed(random_seed)
            self.sample_parameters()

        if vehicle_config is not None:
            self.update_config(vehicle_config)
        from metadrive.component.vehicle.vehicle_type import vehicle_class_to_type
        #self.config["vehicle_model"] = vehicle_class_to_type[self.__class__]
        self.config["vehicle_model"] = 'default'

        # Update some modules that might not be initialized before
        self.add_navigation()

        self.set_pitch(0)
        self.set_roll(0)
        if position is not None:
            # Highest priority
            pass
        elif self.config["spawn_position_heading"] is None:
            # spawn_lane_index has second priority
            map = self.engine.current_map
            if map is None:
                logger.warning("No map is provided. Set vehicle to position (0, 0) with heading 0")
                position = [0, 0]
                heading = 0
            else:
                lane = map.road_network.get_lane(self.config["spawn_lane_index"])
                position = lane.position(self.config["spawn_longitude"], self.config["spawn_lateral"])
                heading = lane.heading_theta_at(self.config["spawn_longitude"])
        else:
            assert self.config["spawn_position_heading"] is not None, "At least setting one initialization method"
            position = self.config["spawn_position_heading"][0]
            heading = self.config["spawn_position_heading"][1]

        self.spawn_place = position
        # print("position:", position)
        self.set_heading_theta(heading)
        self.set_static(False)
        # self.set_wheel_friction(self.config["wheel_friction"])

        if len(position) == 2:
            self.set_position(position, height=self.HEIGHT / 2)
        elif len(position) == 3:
            self.set_position(position[:2], height=position[-1])
        else:
            raise ValueError()

        self.reset_navigation()
        self.body.clearForces()
        self.body.setLinearVelocity(Vec3(0, 0, 0))
        self.body.setAngularVelocity(Vec3(0, 0, 0))
        self.system.resetSuspension()
        self._apply_throttle_brake(0.0)
        # np.testing.assert_almost_equal(self.position, pos, decimal=4)

        # done info
        self._init_step_info()

        # other info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = self.spawn_place
        self.last_heading_dir = self.heading
        self.last_velocity = self.velocity  # 2D vector
        self.last_speed = self.speed  # Scalar

        self.update_dist_to_left_right()
        self.takeover = False
        self.energy_consumption = 0

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()
        self.expert_takeover = False
        if self.config["navigation_module"] and self.engine.current_map is not None:
            assert self.navigation

        if self.config["spawn_velocity"] is not None:
            self.set_velocity(self.config["spawn_velocity"], in_local_frame=self.config["spawn_velocity_car_frame"])

        # clean lights
        if self.config["light"]:
            self.add_light()
        else:
            self.remove_light()

        # self.add_light()
        
    def add_navigation(self):
        if self.navigation is not None or self.config["navigation_module"] is None or self.engine.current_map is None:
            return
        navi = self.config["navigation_module"]
        navi = HRLNodeNavigation
        self.config["seq_traj_len"] = 10
        self.config["show_seq_traj"] = True 
        self.config["enable_u_turn"] = False
        self.navigation = navi(
            # self.engine,
            show_navi_mark=self.config["show_navi_mark"],
            show_dest_mark=self.config["show_dest_mark"],
            show_line_to_dest=self.config["show_line_to_dest"],
            seq_traj_len = self.config["seq_traj_len"],
            show_seq_traj = self.config["show_seq_traj"],
            enable_u_turn = self.config["enable_u_turn"],
            panda_color=self.panda_color,
            name=self.name,
            vehicle_config=self.config
        )