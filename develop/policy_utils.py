import numpy as np
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.component.lane.point_lane import PointLane
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math import not_zero, wrap_to_pi, norm
import logging

class MacroTrajPolicy(IDMPolicy):
    """
    We implement this policy based on the HighwayEnv code base.
    """
    def __init__(self, control_object, random_seed):
        super(MacroTrajPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.base_pos = self.control_object.position
        self.base_heading = self.control_object.heading_theta
        self.last_heading = self.base_heading

    def convert_wp_to_world_coord(self, rbt_pos, rbt_heading, wp, visual=False):
        compose_visual = 0
        if visual:
            compose_visual += 0 # 4.51 / 2
        theta = np.arctan2(wp[1], wp[0] + compose_visual)
        rbt_heading = rbt_heading #np.arctan2(rbt_heading[1], rbt_heading[0])
        theta = wrap_to_pi(rbt_heading) + wrap_to_pi(theta)
        norm_len = norm(wp[0] + compose_visual, wp[1])
        position = rbt_pos
        heading = np.sin(theta) * norm_len
        side = np.cos(theta) * norm_len
        return position[0] + side, position[1] + heading

    def convert_waypoint_list_coord(self, rbt_pos, rbt_heading, wp_list, visual = False):
        wp_w_list = []
        LENGTH = 4.51
        for wp in wp_list:
            wp_w = self.convert_wp_to_world_coord(rbt_pos, rbt_heading, wp, visual)
            wp_w_list.append(wp_w)
        return wp_w_list

    def act(self, *args, **kwargs):
        if (self.control_object.arrive_destination and hasattr(self.control_object, 'macro_succ')):
            self.control_object.macro_succ = True
        if (self.control_object.crash_vehicle and hasattr(self.control_object, 'crash_vehicle')):
            self.control_object.macro_crash = True
        frame = args[1]
        wp_list = args[2]
        ego_vehicle = self.control_object
        if frame ==0:
            self.base_pos = ego_vehicle.position
            self.base_heading = ego_vehicle.heading_theta
            self.control_object.v_wps = self.convert_waypoint_list_coord(self.base_pos, self.base_heading,wp_list, True)
            self.control_object.penultimate_state = self.control_object.traj_wp_list[-2] # if len(wp_list)>2 else self.control_object.traj_wp_list[-1]
            new_state = {}        
            new_state['position'] = ego_vehicle.position
            new_state['yaw'] = ego_vehicle.heading_theta
            new_state['speed'] = ego_vehicle.last_spd
            self.control_object.traj_wp_list = []
            self.control_object.traj_wp_list.append(new_state)
        self.control_object.v_indx = frame 
        wp_list = self.convert_waypoint_list_coord(self.base_pos, self.base_heading, wp_list)
        current_pos = np.array(wp_list[frame])
        target_pos = np.array(wp_list[frame+1])
        diff = target_pos - current_pos 
        norm = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
        if abs(norm) < 0.001:
            heading_theta_at = self.last_heading
        else:
            direction = diff / norm 
            heading_theta_at = np.arctan2(direction[1], direction[0])
        self.last_heading = heading_theta_at 
        steering = 0#self.steering_conrol_traj(lateral, heading_theta_at)
        throtle_brake = 0 #self.speed_control(target_vel)
        ttarget_pos = self.base_pos + target_pos
        hheading_theata_at = heading_theta_at + self.base_heading
        ego_vehicle.set_position(target_pos)
        ego_vehicle.set_heading_theta(heading_theta_at)
        ego_vehicle.last_spd = norm / ego_vehicle.physics_world_step_size
        new_state = {}
        new_state['position'] = target_pos
        new_state['yaw'] = heading_theta_at
        new_state['speed'] = ego_vehicle.last_spd
        self.control_object.traj_wp_list.append(new_state)
        if hasattr(self.control_object, 'taecrl_max_spd'):
            if hasattr(self.control_object, 'taecrl_max_steer') and hasattr(self.control_object, 'taecrl_max_acc') :
                if hasattr(self.control_object, 'vis_state'):
                    steering = self.control_object.vis_state[5] / self.control_object.taecrl_max_steer
                    if steering > 1.0:
                        steering = 1.0
                    elif steering < -1.0:
                        steering = -1.0
                    throtle_brake = self.control_object.vis_state[4] / self.control_object.taecrl_max_acc
                    if throtle_brake > 1.0:
                        throtle_brake = 1.0
                    elif throtle_brake < -1.0:
                        throtle_brake = -1.0
        return [steering, throtle_brake]