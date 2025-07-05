import numpy as np
import torch

def calculate_and_combine_tensor_with_zeros(coordinates, delta_t=0.1):
    """
    计算一系列 (x, y) 坐标对应的 theta 和 speed，并将第一行补零，后续拼接计算值。
    
    参数:
        coordinates (list of tuples): 一系列 (x, y) 坐标点，例如 [(x1, y1), (x2, y2), ...]。
        delta_t (float): 时间间隔，默认 0.1 秒。
    
    返回:
        torch.Tensor: 拼接后的张量，形状为 (N, 4)，第一行为 [x, y, 0, 0]。
    """
    # 确保输入是 NumPy 数组，方便计算
    coordinates = np.array(coordinates)
    x, y = coordinates[:, 0], coordinates[:, 1]

    # 计算 dx 和 dy
    dx = np.diff(x)
    dy = np.diff(y)

    # 计算 theta = dy/dx
    theta = np.arctan2(dy, dx)  # 使用 arctan2 确保结果在 [-pi, pi] 范围内

    # 计算距离（欧几里得距离）
    distances = np.sqrt(dx**2 + dy**2)

    # 计算速度 speed = distance / delta_t
    speed = distances / delta_t

    # 为第一行补零
    zero_row = np.array([[x[0], y[0], 0.0, 10.0]])

    # 将计算的 (x[:-1], y[:-1], theta, speed) 拼接到第一行之后
    combined = np.column_stack((x[1:], y[1:], theta, speed))

    # 拼接第一行和后续计算结果
    combined = np.vstack((zero_row, combined))

    # 转换为 PyTorch 张量
    combined_tensor = torch.tensor(combined, dtype=torch.float32)

    return combined_tensor


# 示例输入
coordinates = [
    (0, 0),
    (1, 1),
    (2, 3),
    (3, 6),
    (5, 10)
]

# 调用函数
result_tensor = calculate_and_combine_tensor_with_zeros(coordinates)

# 输出结果
print("Result Tensor:")
print(result_tensor)