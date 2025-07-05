class StanleyIDMPolicy(IDMPolicy):
    """Use IDM policy for speed. Use Stanley policy."""

    def __init__(self, *args, **kwargs):
        super(StanleyIDMPolicy, self).__init__(*args, **kwargs)
        self.k = 0.5  # steering control gain
        self.last_target_idx = 0

    def act(self, agent_id):
        # concat lane
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        try:
            if success and self.enable_lane_change:
                # perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.MAX_LONG_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except:
            # error fallback
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane
            # logging.warning("IDM bug! fall back")
            # print("IDM bug! fall back")

        # control by Stanley and IDM

        target_lane = self.get_target_lane(steering_target_lane)  # [x, y, heading, self.target_speed, 0]

        steering, _ = self.stanley_steering_control(target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        action = [steering, acc]
        self.action_info["action"] = action
        return action

    def get_target_lane(self, steering_target_lane):
        """
        only for test
        """
        target_lane = []
        start_point = steering_target_lane.start
        end_point = steering_target_lane.end
        x, y = start_point[0], start_point[1]
        delta_x = end_point[0] - start_point[0]
        delta_y = end_point[1] - start_point[1]
        for i in range(20):
            x += delta_x * i / 20
            y += delta_y * i / 20
            long, lat = steering_target_lane.local_coordinates([x, y])
            heading = steering_target_lane.heading_theta_at(long + 1)
            target_lane.append([x, y, heading, self.target_speed, 0])

        return target_lane

    def reset(self):
        super().reset()
        self.k = 0.5  # steering control gain
        self.last_target_idx = 0

    def stanley_steering_control(self, target_lane):
        """
        Stanley steering control.
        """
        ego_vehicle = self.control_object
        cx = [point[0] for point in target_lane]
        cy = [point[1] for point in target_lane]
        cyaw = [point[2] for point in target_lane]

        current_target_idx, error_front_axle = self.calc_target_index(self.control_object, cx, cy)

        if self.last_target_idx >= current_target_idx:
            current_target_idx = self.last_target_idx
        else:
            self.last_target_idx = current_target_idx

        # theta_e corrects the heading error
        theta_e = self.normalize_angle(cyaw[current_target_idx] - ego_vehicle.heading_theta)
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.k * error_front_axle, ego_vehicle.speed)
        # Steering control
        delta = theta_e + theta_d

        return delta, current_target_idx

    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].
        """
        if isinstance(angle, float):
            is_float = True
        else:
            is_float = False
        angle = np.asarray(angle).flatten()
        mod_angle = (angle + np.pi) % (2 * np.pi) - np.pi

        if is_float:
            return mod_angle.item()
        else:
            return mod_angle

    def calc_target_index(self, ego_vehicle, cx, cy):
        """
        Compute index in the trajectory list of the target.
        """
        # Calc front axle position
        fx = ego_vehicle.position[0] + ego_vehicle.LENGTH * np.cos(ego_vehicle.heading_theta)
        fy = ego_vehicle.position[1] + ego_vehicle.LENGTH * np.sin(ego_vehicle.heading_theta)

        # Search nearest point index
        dx = [fx - icx for icx in cx]
        dy = [fy - icy for icy in cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(ego_vehicle.heading_theta + np.pi / 2),
                        -np.sin(ego_vehicle.heading_theta + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle