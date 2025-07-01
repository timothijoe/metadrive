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
    
