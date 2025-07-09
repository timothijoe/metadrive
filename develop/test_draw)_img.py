import pygame
import math

# 初始化 Pygame
pygame.init()

# 定义窗口大小
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Car Animation with Transparent Background")

# 加载背景图片
background_image = pygame.image.load("/home/zhoutong/Downloads/image_v1.jpg").convert_alpha()
background_image = pygame.transform.scale(background_image, (WINDOW_WIDTH, WINDOW_HEIGHT))

# 加载带透明背景的汽车图片
car_image = pygame.image.load("/home/zhoutong/Downloads/car_img_v2-removebg-preview.png").convert_alpha()  # 确保加载透明背景
original_width, original_height = car_image.get_size()

# 缩放汽车图片（保持比例）
car_width = 50
aspect_ratio = original_width / original_height
car_height = int(car_width / aspect_ratio)
car_image = pygame.transform.scale(car_image, (car_width, car_height))

# 圆形轨迹参数
circle_center = (400, 300)  # 圆心位置
circle_radius = 150         # 圆半径
angle = 0                   # 初始角度（弧度制）

# 动画参数
fps = 60                     # 帧率
speed = 2 * math.pi / 360     # 每帧旋转的角度（弧度）

# 初始化时钟
clock = pygame.time.Clock()

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # 退出事件
            running = False

    # 计算当前汽车的位置
    car_x = circle_center[0] + circle_radius * math.cos(angle)  # 圆上 x 坐标
    car_y = circle_center[1] + circle_radius * math.sin(angle)  # 圆上 y 坐标

    # 计算汽车的朝向角度
    tangent_angle = math.atan2(
        circle_radius * math.sin(angle + 0.01) - circle_radius * math.sin(angle),
        circle_radius * math.cos(angle + 0.01) - circle_radius * math.cos(angle)
    )
    car_angle = -math.degrees(tangent_angle)  # 转换为角度制

    # 更新角度
    angle += speed
    if angle > 2 * math.pi:  # 超过一圈后重置角度
        angle -= 2 * math.pi

    # 绘制背景
    screen.blit(background_image, (0, 0))

    # 绘制轨迹圆
    pygame.draw.circle(screen, (255, 255, 255), circle_center, circle_radius, 1)  # 白色圆边

    # 绘制汽车
    rotated_car = pygame.transform.rotozoom(car_image, car_angle, 1.0)  # 使用rotozoom旋转汽车图像
    rotated_rect = rotated_car.get_rect(center=(car_x, car_y))  # 保持旋转后图像的中心点一致
    screen.blit(rotated_car, rotated_rect.topleft)  # 绘制旋转后的汽车图像

    # 更新屏幕
    pygame.display.flip()

    # 控制帧率
    clock.tick(fps)

# 退出 Pygame
pygame.quit()
