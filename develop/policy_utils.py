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
        self.heading_pid = PIDController(1.2, 0.1, 3.5)
        self.lateral_pid = PIDController(0.3, .0, 0.0)
        self.ACC_FACTOR = 1.0
        self.DELTA = 4.0

    def convert_wp_to_world_coord_old(self, rbt_pos, rbt_heading, wp, visual=False):
        theta = np.arctan2(wp[1], wp[0])
        rbt_heading = rbt_heading #np.arctan2(rbt_heading[1], rbt_heading[0])
        theta = wrap_to_pi(rbt_heading) + wrap_to_pi(theta)
        norm_len = norm(wp[0], wp[1])
        position = rbt_pos
        heading = np.sin(theta) * norm_len
        side = np.cos(theta) * norm_len
        return position[0] + side, position[1] + heading
    
    
    def convert_wp_to_world_coord(self, rbt_pos, rbt_heading, wp, visual=False):
        """
        将目标点从机器人坐标系转换到世界坐标系，包括位置和朝向。

        :param rbt_pos: 机器人在世界坐标系中的位置 (x, y)。
        :param rbt_heading: 机器人在世界坐标系中的朝向（弧度）。
        :param wp: 目标点在机器人坐标系中的坐标和朝向 (x', y', theta')。
        :param visual: 是否考虑视觉偏移量。
        :return: 目标点在世界坐标系中的坐标和朝向 (X, Y, Theta)。
        """
        # 提取目标点在机器人坐标系中的坐标和朝向
        wp_x, wp_y, wp_theta, wp_vel = wp

        # 机器人在世界坐标系中的位置
        rbt_x, rbt_y = rbt_pos

        # 计算目标点在世界坐标系中的位置，使用标准旋转公式
        cos_heading = np.cos(rbt_heading)
        sin_heading = np.sin(rbt_heading)

        world_x = rbt_x + wp_x * cos_heading - wp_y * sin_heading
        world_y = rbt_y + wp_x * sin_heading + wp_y * cos_heading

        # 计算目标点在世界坐标系中的朝向
        world_theta = wrap_to_pi(rbt_heading + wp_theta)

        return world_x, world_y, world_theta, wp_vel

    def convert_waypoint_list_coord(self, rbt_pos, rbt_heading, wp_list, visual = False):
        wp_w_list = []
        LENGTH = 4.51
        for wp in wp_list:
            wp_w = self.convert_wp_to_world_coord(rbt_pos, rbt_heading, wp, visual)
            wp_w_list.append(wp_w)
        return wp_w_list

    def local_coordinates(self, x_target, y_target, theta_target, x_ego, y_ego):
        """
        计算车辆在目标点局部坐标系下的纵向和横向距离。

        :param ego_position: 车辆的全局位置 (x, y)。
        :return: (longitude, lateral) - 纵向距离和横向距离。
        """
        # # 目标点的位置和方向
        # x_target, y_target, theta = self.x, self.y, self.theta

        # # 车辆位置
        # x_ego, y_ego = ego_position
        theta = theta_target
        # 相对位置向量
        delta_x = x_ego - x_target
        delta_y = y_ego - y_target
        # 计算纵向和横向距离
        longitude = delta_x * np.cos(theta) + delta_y * np.sin(theta)
        lateral = -delta_x * np.sin(theta) + delta_y * np.cos(theta)
        return longitude, lateral

    def steering_control(self, target_point) -> float:
        ego_vehicle = self.control_object
        rbt_pose = ego_vehicle.position
        v_heading = ego_vehicle.heading_theta
        lane_heading = target_point[2]
        long, lat = self.local_coordinates(target_point[0], target_point[1], target_point[2], rbt_pose[0], rbt_pose[1])
        steering = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)
    
    def acceleration(self, target_point):
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(target_point[3], 0)
        ego_cur_speed= ego_vehicle.speed_km_h / 3.6
        acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_cur_speed, 0) / ego_target_speed, self.DELTA))
        return acceleration 

    def act(self, *args, **kwargs):
        # if (self.control_object.arrive_destination and hasattr(self.control_object, 'macro_succ')):
        #     self.control_object.macro_succ = True
        # if (self.control_object.crash_vehicle and hasattr(self.control_object, 'crash_vehicle')):
        #     self.control_object.macro_crash = True
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
        current_pos = np.array(wp_list[frame][:2])
        target_pos = np.array(wp_list[frame+1][:2])
        target_point = wp_list[frame+1]
        rbt_point = ego_vehicle.position
        lon, lat = self.local_coordinates(target_point[0], target_point[1], target_point[2], rbt_point[0], rbt_point[1])
        diff = target_pos - current_pos 
        norm = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
        if abs(norm) < 0.001:
            heading_theta_at = self.last_heading
        else:
            direction = diff / norm 
            heading_theta_at = np.arctan2(direction[1], direction[0])
        heading_theta_at = wp_list[frame+1][2]
        self.last_heading = heading_theta_at 
        steering = 0#self.steering_conrol_traj(lateral, heading_theta_at)
        throtle_brake = 0 #self.speed_control(target_vel)
        ttarget_pos = self.base_pos + target_pos
        hheading_theata_at = heading_theta_at + self.base_heading
        # ego_vehicle.set_position(target_pos)
        # ego_vehicle.set_heading_theta(heading_theta_at)
        ego_vehicle.last_spd = norm / ego_vehicle.physics_world_step_size
        new_state = {}
        new_state['position'] = target_pos
        new_state['yaw'] = heading_theta_at
        new_state['speed'] = ego_vehicle.last_spd
        self.control_object.traj_wp_list.append(new_state)
        steering = self.steering_control(target_point)
        acceleration = self.acceleration(target_point)
        throtle_brake = acceleration
        
        # if hasattr(self.control_object, 'taecrl_max_spd'):
        #     if hasattr(self.control_object, 'taecrl_max_steer') and hasattr(self.control_object, 'taecrl_max_acc') :
        #         if hasattr(self.control_object, 'vis_state'):
        #             steering = self.control_object.vis_state[5] / self.control_object.taecrl_max_steer
        #             if steering > 1.0:
        #                 steering = 1.0
        #             elif steering < -1.0:
        #                 steering = -1.0
        #             throtle_brake = self.control_object.vis_state[4] / self.control_object.taecrl_max_acc
        #             if throtle_brake > 1.0:
        #                 throtle_brake = 1.0
        #             elif throtle_brake < -1.0:
        #                 throtle_brake = -1.0
        return [steering, throtle_brake]