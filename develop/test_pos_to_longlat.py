import numpy as np

class TargetWaypoint:
    def __init__(self, x, y, theta, v):
        """
        初始化目标点。

        :param x: 目标点的 x 坐标。
        :param y: 目标点的 y 坐标。
        :param theta: 目标点的方向（弧度）。
        :param v: 目标点的目标速度。
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v

    def local_coordinates(self, ego_position):
        """
        计算车辆在目标点局部坐标系下的纵向和横向距离。

        :param ego_position: 车辆的全局位置 (x, y)。
        :return: (longitude, lateral) - 纵向距离和横向距离。
        """
        # 目标点的位置和方向
        x_target, y_target, theta = self.x, self.y, self.theta

        # 车辆位置
        x_ego, y_ego = ego_position

        # 相对位置向量
        delta_x = x_ego - x_target
        delta_y = y_ego - y_target

        # 计算纵向和横向距离
        longitude = delta_x * np.cos(theta) + delta_y * np.sin(theta)
        lateral = -delta_x * np.sin(theta) + delta_y * np.cos(theta)

        return longitude, lateral
    
    
def local_coordinates(x_target, y_target, theta_target, x_ego, y_ego):
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
    
    
# 定义目标点
target_waypoint = TargetWaypoint(x=10, y=5, theta=np.pi/4, v=15)  # x=10, y=5, 朝向 45 度，目标速度 15 m/s

# 车辆位置
ego_position = (12, 7)  # x=12, y=7

# 计算局部坐标下的纵向和横向误差
longitude, lateral = target_waypoint.local_coordinates(ego_position)

print(f"Longitude error: {longitude:.2f} m, Lateral error: {lateral:.2f} m")