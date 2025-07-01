import numpy as np
from panda3d.core import TransparencyAttrib, LineSegs, NodePath, Vec3
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pg_space import Parameter, BlockParameterSpace
from metadrive.component.pgblock.bottleneck import Merge, Split
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.navigation_module.base_navigation import BaseNavigation
from metadrive.utils import clip, norm, get_np_random
from metadrive.utils.math import panda_vector, wrap_to_pi
from metadrive.utils.pg.utils import ray_localization
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation


class HRLNodeNavigation(NodeNetworkNavigation):
    """
    A helper class for localizing vehicles and retrieving navigation information. 
    Supports trajectory visualization and waypoint conversion.
    """

    def __init__(
        self,
        show_navi_mark: bool = False,
        show_dest_mark: bool = False,
        show_line_to_dest: bool = False,
        seq_traj_len: int = 30,
        show_seq_traj: bool = False,
        enable_u_turn: bool = False,
        panda_color=None,
        name=None,
        vehicle_config=None
    ):
        super().__init__(
            show_navi_mark=show_navi_mark,
            show_dest_mark=show_dest_mark,
            show_line_to_dest=show_line_to_dest,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )
        self._show_traj = show_seq_traj
        self.seq_traj_len = seq_traj_len
        self.enable_u_turn = enable_u_turn
        self.u_turn_case = False
        self.should_redraw = False
        self.activate_car_pos_marker = False
        self.LINE_TO_DEST_HEIGHT += 4

        if self._show_traj:
            self._initialize_trajectories()

    def panda_position(self, position, z = 0.0):
        if len(position) == 3:
            z = position[2]
        return Vec3(position[0], -position[1], z)

    def _initialize_trajectories(self):
        """
        Initialize trajectory line segments for visualization.
        """
        for i in range(self.seq_traj_len):
            line = self._create_line_segment(self.navi_mark_color, alpha=0.7)
            self.__dict__[f'traj_{i}'] = NodePath(line.create())
            self.__dict__[f'traj_{i}'].reparentTo(self.origin)

        # Initialize current position marker
        line = self._create_line_segment((0.5, 0.5, 0.5), alpha=0.7)
        self.current_pos_marker = NodePath(line.create())
        self.current_pos_marker.reparentTo(self.origin)

    def _create_line_segment(self, color, alpha=1.0, thickness=2):
        """
        Helper function to create a LineSegs object with specified color and thickness.

        :param color: RGB tuple for the line color.
        :param alpha: Alpha value for transparency.
        :param thickness: Line thickness.
        :return: A LineSegs object.
        """
        line = LineSegs()
        line.setColor(*color, alpha)
        line.setThickness(thickness)
        return line

    def _draw_trajectories(self, wp_list):
        """
        Draw trajectories based on a list of waypoints.

        :param wp_list: List of waypoints to visualize.
        """
        for i in range(self.seq_traj_len):
            self._update_trajectory_line(i, wp_list[i], wp_list[i + 1])

    def _update_trajectory_line(self, index, start_wp, end_wp):
        """
        Update a single trajectory line segment.

        :param index: Trajectory index.
        :param start_wp: Starting waypoint (x, y).
        :param end_wp: Ending waypoint (x, y).
        """
        line = self._create_line_segment(self.navi_mark_color, alpha=0.7)
        line.moveTo(self.panda_position(start_wp, self.LINE_TO_DEST_HEIGHT))
        line.drawTo(self.panda_position(end_wp, self.LINE_TO_DEST_HEIGHT))

        # Update the trajectory node
        self.__dict__[f'traj_{index}'].removeNode()
        self.__dict__[f'traj_{index}'] = NodePath(line.create(False))
        #self.__dict__[f'traj_{index}'].hide(CamMask.Shadow | CamMask.RgbCam)
        self.__dict__[f'traj_{index}'].reparentTo(self.origin)

    def convert_wp_to_world_coord(self, rbt_pos, rbt_heading, wp):
        """
        Convert a waypoint from local to world coordinates.

        :param rbt_pos: Robot's position.
        :param rbt_heading: Robot's heading.
        :param wp: Waypoint in local coordinates.
        :return: Waypoint in world coordinates.
        """
        theta = wrap_to_pi(np.arctan2(wp[1], wp[0]) + np.arctan2(rbt_heading[1], rbt_heading[0]))
        norm_len = norm(wp[0], wp[1])
        x = rbt_pos[0] + np.cos(theta) * norm_len
        y = rbt_pos[1] + np.sin(theta) * norm_len
        return x, y

    def convert_waypoint_list_coord(self, rbt_pos, rbt_heading, wp_list):
        """
        Convert a list of waypoints from local to world coordinates.

        :param rbt_pos: Robot's position.
        :param rbt_heading: Robot's heading.
        :param wp_list: List of waypoints in local coordinates.
        :return: List of waypoints in world coordinates.
        """
        return [self.convert_wp_to_world_coord(rbt_pos, rbt_heading, wp) for wp in wp_list]

    # def show_car_pos(self, wp_list, current_time_step):
    #     """
    #     Visualize the current car position and trajectory.

    #     :param wp_list: List of waypoints.
    #     :param current_time_step: Current time step index.
    #     """
    #     cx, cy = wp_list[current_time_step]
    #     ncx, ncy = wp_list[current_time_step + 1]
    #     theta = np.arctan2(ncy - cy, ncx - cx)

    #     # Draw current position marker
    #     line = self._create_line_segment((0.9, 0.9, 0.9), alpha=1.0, thickness=10)
    #     line.moveTo(self.panda_position((cx - 0.1 * np.sin(theta), cy + 0.1 * np.cos(theta)), self.LINE_TO_DEST_HEIGHT))
    #     line.drawTo(self.panda_position((cx + 0.1 * np.sin(theta), cy - 0.1 * np.cos(theta)), self.LINE_TO_DEST_HEIGHT))
    #     self.current_pos_marker.removeNode()
    #     self.current_pos_marker = NodePath(line.create(False))
    #     #self.current_pos_marker.hide(CamMask.Shadow | CamMask.RgbCam)
    #     self.current_pos_marker.reparentTo(self.origin)

    #     # Draw trajectory ahead of the car
    #     for i in range(self.seq_traj_len):
    #         color = (1.0, 0.4, 0.0) if current_time_step > i else (0.0, 0.7, 1.0)
    #         self._update_trajectory_line(i, wp_list[i], wp_list[i + 1], color=color)

    def show_car_pos(self, wp_list, current_time_step):
        #print(current_time_step)
        cx = wp_list[current_time_step][0]
        cy = wp_list[current_time_step][1]
        ncx = wp_list[current_time_step+1][0]
        ncy = wp_list[current_time_step+1][1]
        theta = np.arctan2(ncy-cy, ncx-cx)
        theta = 0.0
        lines = LineSegs()
        lines.setColor(0.9, 0.9, 0.9, 1.0)
        #lines.moveTo(panda_position(wp_list[i][0], self.LINE_TO_DEST_HEIGHT+4))
        lines.moveTo(self.panda_position((cx-0.1*np.sin(theta) , cy + 0.1*np.cos(theta)), self.LINE_TO_DEST_HEIGHT))
        lines.drawTo(self.panda_position((cx + 0.1*np.sin(theta), cy - 0.1*np.cos(theta)), self.LINE_TO_DEST_HEIGHT))
        lines.setThickness(10)
        self.current_pos_marker.removeNode()
        self.current_pos_marker = NodePath(lines.create(False))
        #self.current_pos_marker.hide(CamMask.Shadow | CamMask.RgbCam)
        self.current_pos_marker.reparentTo(self.origin)
        for i in range(self.seq_traj_len):
            lines = LineSegs()
            if current_time_step > i:
                lines.setColor(1.0, 0.4, 0.0, 0.7)
            else:
                lines.setColor(0.0, 0.7, 1.0, 0.7)
            #lines.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7)
            #lines.moveTo(panda_position(wp_list[i][0], self.LINE_TO_DEST_HEIGHT+4))
            lines.moveTo(self.panda_position((wp_list[i][0], wp_list[i][1]), self.LINE_TO_DEST_HEIGHT))
            lines.drawTo(self.panda_position((wp_list[i+1][0], wp_list[i+1][1]), self.LINE_TO_DEST_HEIGHT))
            lines.setThickness(2)
            self.__dict__['traj_{}'.format(i)].removeNode()
            self.__dict__['traj_{}'.format(i)] = NodePath(lines.create(False))
            #self.__dict__['traj_{}'.format(i)].hide(CamMask.Shadow | CamMask.RgbCam)
            self.__dict__['traj_{}'.format(i)].reparentTo(self.origin)


    def get_waypoint_list(self):
        """
        Generate a list of waypoints for demonstration purposes.

        :return: List of waypoints.
        """
        x = np.arange(0, 50, 0.1)
        y = np.cos(x) - 1
        x += 4.51 / 2  # Offset by half the vehicle length
        return [[x[i], y[i]] for i in range(len(x))]
    
    def update_localization(self, ego_vehicle):
        """
        Update current position, route completion and checkpoints according to current position.

        Args:
            ego_vehicle: a vehicle object

        Returns:
            None
        """
        position = ego_vehicle.position
        lane, lane_index = self._update_current_lane(ego_vehicle)
        long, _ = lane.local_coordinates(position)
        need_update = self._update_target_checkpoints(lane_index, long)
        assert len(self.checkpoints) >= 2

        # Update travelled_length for route completion
        long_in_ref_lane, _ = self.current_ref_lanes[0].local_coordinates(position)
        travelled = long_in_ref_lane - self._last_long_in_ref_lane
        self.travelled_length += travelled
        self._last_long_in_ref_lane = long_in_ref_lane
        # print(f"{self.travelled_length=}, {travelled=}, {long_in_ref_lane=}, "
        #       f"{self.route_completion=}, {self._last_long_in_ref_lane=}")

        # target_road_1 is the road segment the vehicle is driving on.
        if need_update:
            target_road_1_start = self.checkpoints[self._target_checkpoints_index[0]]
            target_road_1_end = self.checkpoints[self._target_checkpoints_index[0] + 1]
            target_lanes_1 = self.map.road_network.graph[target_road_1_start][target_road_1_end]
            self.current_ref_lanes = target_lanes_1
            self.current_road = Road(target_road_1_start, target_road_1_end)

            self._last_long_in_ref_lane = self.current_ref_lanes[0].local_coordinates(position)[0]

            # target_road_2 is next road segment the vehicle should drive on.
            target_road_2_start = self.checkpoints[self._target_checkpoints_index[1]]
            target_road_2_end = self.checkpoints[self._target_checkpoints_index[1] + 1]
            target_lanes_2 = self.map.road_network.graph[target_road_2_start][target_road_2_end]

            if target_road_1_start == target_road_2_start:
                # When we are in the final road segment that there is no further road to drive on
                self.next_road = None
                self.next_ref_lanes = None
            else:
                self.next_road = Road(target_road_2_start, target_road_2_end)
                self.next_ref_lanes = target_lanes_2

        self._navi_info.fill(0.0)
        half = self.CHECK_POINT_INFO_DIM
        # Put the next checkpoint's information into the first half of the navi_info
        self._navi_info[:half], lanes_heading1, next_checkpoint = self._get_info_for_checkpoint(
            lanes_id=0, ref_lane=self.current_ref_lanes[0], ego_vehicle=ego_vehicle
        )

        # Put the next of the next checkpoint's information into the first half of the navi_info
        self._navi_info[half:], lanes_heading2, next_next_checkpoint = self._get_info_for_checkpoint(
            lanes_id=1,
            ref_lane=self.next_ref_lanes[0] if self.next_ref_lanes is not None else self.current_ref_lanes[0],
            ego_vehicle=ego_vehicle
        )

        if hasattr(ego_vehicle, 'v_indx') and self._show_traj:
            #print(ego_vehicle.v_indx)
            if ego_vehicle.v_indx == 0:
                #self.draw_car_path(ego_vehicle.v_wps)
                self.activate_car_pos_marker = True
            if self.activate_car_pos_marker:
                self.show_car_pos(ego_vehicle.v_wps, ego_vehicle.v_indx)

        self.navi_arrow_dir = [lanes_heading1, lanes_heading2]
        if self._show_navi_info:
            # Whether to visualize little boxes in the scene denoting the checkpoints
            pos_of_goal = next_checkpoint
            self._goal_node_path.setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
            self._goal_node_path.setH(self._goal_node_path.getH() + 3)

            pos_of_goal = next_next_checkpoint
            self._goal_node_path2.setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
            self._goal_node_path2.setH(self._goal_node_path2.getH() + 3)

            dest_pos = self._dest_node_path.getPos()
            self._draw_line_to_dest(start_position=ego_vehicle.position, end_position=(dest_pos[0], dest_pos[1]))
            navi_pos = self._goal_node_path.getPos()
            next_navi_pos = self._goal_node_path2.getPos()
            self._draw_line_to_navi(
                start_position=ego_vehicle.position,
                end_position=(navi_pos[0], navi_pos[1]),
                next_checkpoint=(next_navi_pos[0], next_navi_pos[1])
            )