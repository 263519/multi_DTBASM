import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
import random
import numpy as np
import math

NUM_ROBOTS = 10
MAX_LINEAR_SPEED = 0.5
MAX_ANGULAR_SPEED = 1.0
OCCUPANCY_THRESHOLD = 50

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.declare_parameter('separation_strength', 0.7)
        self.declare_parameter('alignment_strength', 0.7)
        self.declare_parameter('cohesion_strength', 0.7)
        self.declare_parameter('neighborhood_radius', 1.5)
        
        self.declare_parameter('obstacle_strength', 1.5)
        self.declare_parameter('obstacle_detection_radius', 2.0)
        
        self.declare_parameter('goal_strength', 1.0)
        self.declare_parameter('use_goal', False)
        self.declare_parameter('goal_x', -2.0)
        self.declare_parameter('goal_y', -3.0)

        self.publishers_ = []
        self.odom_subscribers_ = []
        self.robot_poses = [None] * NUM_ROBOTS
        self.robot_velocities = [None] * NUM_ROBOTS
        self.robot_yaws = [None] * NUM_ROBOTS

        self.map_data = None
        self.map_info = None

        for i in range(NUM_ROBOTS):
            topic_name = f'/robot_{i}/cmd_vel'
            pub = self.create_publisher(Twist, topic_name, 10)
            self.publishers_.append(pub)

            odom_topic = f'/robot_{i}/odom'
            self.odom_subscribers_.append(self.create_subscription(
                Odometry,
                odom_topic,
                lambda msg, index=i: self.odom_callback(msg, index),
                10))

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_subscriber_ = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )
        self.get_logger().info('Subscribing to /map topic...')

        self.timer_ = self.create_timer(0.1, self.apply_flocking_rules)
        self.get_logger().info(f'Robot Controller Node has started for {NUM_ROBOTS} robots.')

    def map_callback(self, msg):
        self.map_data = msg.data
        self.map_info = msg.info
        self.get_logger().info(f'Received map: {self.map_info.width}x{self.map_info.height} @ {self.map_info.resolution} m/cell')

    def world_to_map(self, world_pos):
        if self.map_info is None:
            return None
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        res = self.map_info.resolution
        map_x = int((world_pos[0] - origin_x) / res)
        map_y = int((world_pos[1] - origin_y) / res)
        return (map_x, map_y)

    def map_to_world(self, map_pos):
        if self.map_info is None:
            return None
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        res = self.map_info.resolution
        world_x = (map_pos[0] + 0.5) * res + origin_x
        world_y = (map_pos[1] + 0.5) * res + origin_y
        return np.array([world_x, world_y])

    def is_occupied(self, map_x, map_y):
        if self.map_data is None or self.map_info is None:
            return False
        width = self.map_info.width
        height = self.map_info.height
        if 0 <= map_x < width and 0 <= map_y < height:
            index = map_y * width + map_x
            if self.map_data[index] >= OCCUPANCY_THRESHOLD:
                return True
        return False

    def odom_callback(self, msg, robot_index):
        self.robot_poses[robot_index] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = self.euler_from_quaternion(orientation_list)
        self.robot_yaws[robot_index] = yaw
        linear_velocity = msg.twist.twist.linear.x
        self.robot_velocities[robot_index] = np.array([linear_velocity * math.cos(yaw), linear_velocity * math.sin(yaw)])

    def euler_from_quaternion(self, quaternion):
        x, y, z, w = quaternion
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z

    def get_neighbors(self, robot_index):
        neighbors = []
        current_pos = self.robot_poses[robot_index]
        if current_pos is None:
            return neighbors
        neighborhood_radius = self.get_parameter('neighborhood_radius').get_parameter_value().double_value
        for i in range(NUM_ROBOTS):
            if i == robot_index:
                continue
            neighbor_pos = self.robot_poses[i]
            if neighbor_pos is not None:
                distance = np.linalg.norm(current_pos - neighbor_pos)
                if distance < neighborhood_radius:
                    neighbors.append(i)
        return neighbors

    def separation(self, robot_index, neighbors):
        s = self.get_parameter('separation_strength').get_parameter_value().double_value
        v_sep = np.zeros(2)
        current_pos = self.robot_poses[robot_index]
        min_dist = 0.4
        if len(neighbors) == 0:
            return v_sep
        for neighbor_index in neighbors:
            neighbor_pos = self.robot_poses[neighbor_index]
            diff = current_pos - neighbor_pos
            distance = np.linalg.norm(diff)
            if distance > 0 and distance < min_dist:
                v_sep += diff / (distance ** 3)
            elif distance > 0:
                v_sep += diff / (distance * distance)
        return v_sep * s

    def alignment(self, robot_index, neighbors):
        a = self.get_parameter('alignment_strength').get_parameter_value().double_value
        v_align = np.zeros(2)
        if len(neighbors) == 0:
            return v_align
        for neighbor_index in neighbors:
            if self.robot_velocities[neighbor_index] is not None:
                v_align += self.robot_velocities[neighbor_index]
        v_align /= len(neighbors)
        current_velocity = self.robot_velocities[robot_index]
        if current_velocity is not None:
             v_align = v_align - current_velocity
        return v_align * a

    def cohesion(self, robot_index, neighbors):
        c = self.get_parameter('cohesion_strength').get_parameter_value().double_value
        v_coh = np.zeros(2)
        current_pos = self.robot_poses[robot_index]
        if len(neighbors) == 0:
            return v_coh
        center_of_mass = np.zeros(2)
        for neighbor_index in neighbors:
            center_of_mass += self.robot_poses[neighbor_index]
        center_of_mass /= len(neighbors)
        v_coh = center_of_mass - current_pos
        return v_coh * c

    def obstacle_avoidance(self, robot_index):
        v_obs = np.zeros(2)
        if self.map_info is None or self.robot_poses[robot_index] is None:
            return v_obs
        strength = self.get_parameter('obstacle_strength').get_parameter_value().double_value
        detection_radius = self.get_parameter('obstacle_detection_radius').get_parameter_value().double_value
        current_pos_world = self.robot_poses[robot_index]
        current_pos_map = self.world_to_map(current_pos_world)
        if current_pos_map is None:
            return v_obs
        radius_cells = int(detection_radius / self.map_info.resolution)
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx == 0 and dy == 0:
                    continue
                check_map_x = current_pos_map[0] + dx
                check_map_y = current_pos_map[1] + dy
                if self.is_occupied(check_map_x, check_map_y):
                    obstacle_pos_world = self.map_to_world((check_map_x, check_map_y))
                    diff = current_pos_world - obstacle_pos_world
                    distance = np.linalg.norm(diff)
                    if distance > 0 and distance < detection_radius:
                        v_obs += diff / (distance * distance)
        return v_obs * strength

    def goal_seeking(self, robot_index):
        v_goal = np.zeros(2)
        
        use_goal = self.get_parameter('use_goal').get_parameter_value().bool_value
        
        if not use_goal:
            return v_goal
            
        current_pos = self.robot_poses[robot_index]
        if current_pos is None:
            return v_goal

        strength = self.get_parameter('goal_strength').get_parameter_value().double_value
        goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        
        goal_pos = np.array([goal_x, goal_y])
        
        v_goal = goal_pos - current_pos
        
        return v_goal * strength

    def apply_flocking_rules(self):
        if any(p is None for p in self.robot_poses):
            self.get_logger().info('Waiting for odometry data from all robots...', throttle_duration_sec=5)
            for i in range(NUM_ROBOTS):
                msg = Twist()
                self.publishers_[i].publish(msg)
            return

        if self.map_info is None:
            self.get_logger().info('Waiting for data from /map topic...', throttle_duration_sec=5)
            for i in range(NUM_ROBOTS):
                msg = Twist()
                self.publishers_[i].publish(msg)
            return

        for i in range(NUM_ROBOTS):
            neighbors = self.get_neighbors(i)

            v_sep = self.separation(i, neighbors)
            v_align = self.alignment(i, neighbors)
            v_coh = self.cohesion(i, neighbors)
            v_obs = self.obstacle_avoidance(i)
            v_goal = self.goal_seeking(i)

            total_v = v_sep + v_align + v_coh + v_obs + v_goal
            
            total_v += np.random.rand(2) * 0.1 - 0.05

            msg = Twist()
            
            if self.robot_yaws[i] is None:
                continue

            current_angle = self.robot_yaws[i]
            target_angle = math.atan2(total_v[1], total_v[0])
            
            angle_diff = target_angle - current_angle
            while angle_diff > math.pi: angle_diff -= 2 * math.pi
            while angle_diff < -math.pi: angle_diff += 2 * math.pi

            angular_speed_gain = 2.0 
            angular_speed = angular_speed_gain * angle_diff

            if abs(angle_diff) > math.pi / 2:
                msg.linear.x = MAX_LINEAR_SPEED * 0.5
            else:
                msg.linear.x = MAX_LINEAR_SPEED

            msg.angular.z = max(min(angular_speed, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)

            self.publishers_[i].publish(msg)

        use_goal = self.get_parameter('use_goal').get_parameter_value().bool_value
        log_msg = 'Publishing cmd_vel (flocking + avoidance'
        if use_goal:
            log_msg += ' + goal'
        log_msg += ') to all robots...'
        self.get_logger().info(log_msg, throttle_duration_sec=2)


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy # NEW: Importy QoS
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
# from nav_msgs.msg import OccupancyGrid # NEW: Import dla mapy
# import random
# import numpy as np
# import math

# NUM_ROBOTS = 20
# MAX_LINEAR_SPEED = 2.0
# MAX_ANGULAR_SPEED = 2.0
# OCCUPANCY_THRESHOLD = 50 # NEW: Próg, powyżej którego komórka jest uznawana za zajętą

# class RobotController(Node):
#     def __init__(self):
#         super().__init__('robot_controller')

#         # UPDATED: Dodano parametry dla unikania przeszkód
#         self.declare_parameter('separation_strength', 0.7)
#         self.declare_parameter('alignment_strength', 0.8)
#         self.declare_parameter('cohesion_strength', 0.8)
#         self.declare_parameter('neighborhood_radius', 5.0)
#         self.declare_parameter('obstacle_strength', 1.5) # NEW
#         self.declare_parameter('obstacle_detection_radius', 2.0) # NEW

#         self.publishers_ = []
#         self.odom_subscribers_ = []
#         self.robot_poses = [None] * NUM_ROBOTS
#         self.robot_velocities = [None] * NUM_ROBOTS
#         self.robot_yaws = [None] * NUM_ROBOTS

#         # NEW: Zmienne do przechowywania mapy
#         self.map_data = None
#         self.map_info = None

#         for i in range(NUM_ROBOTS):
#             topic_name = f'/robot_{i}/cmd_vel'
#             pub = self.create_publisher(Twist, topic_name, 10)
#             self.publishers_.append(pub)

#             odom_topic = f'/robot_{i}/odom'
#             self.odom_subscribers_.append(self.create_subscription(
#                 Odometry,
#                 odom_topic,
#                 lambda msg, index=i: self.odom_callback(msg, index),
#                 10))

#         # NEW: Subskrybent mapy
#         # Używamy QoS TRANSIENT_LOCAL, aby otrzymać ostatnio opublikowaną mapę
#         map_qos = QoSProfile(
#             depth=1,
#             reliability=ReliabilityPolicy.RELIABLE,
#             durability=DurabilityPolicy.TRANSIENT_LOCAL
#         )
#         self.map_subscriber_ = self.create_subscription(
#             OccupancyGrid,
#             '/map',
#             self.map_callback,
#             map_qos
#         )
#         self.get_logger().info('Subskrybowanie tematu /map...')

#         self.timer_ = self.create_timer(0.1, self.apply_flocking_rules)
#         self.get_logger().info(f'Robot Controller Node has started for {NUM_ROBOTS} robots.')

#     # NEW: Callback dla mapy
#     def map_callback(self, msg):
#         self.map_data = msg.data
#         self.map_info = msg.info
#         self.get_logger().info(f'Otrzymano mapę: {self.map_info.width}x{self.map_info.height} @ {self.map_info.resolution} m/cell')

#     # NEW: Funkcja pomocnicza do konwersji współrzędnych świata na mapę
#     def world_to_map(self, world_pos):
#         if self.map_info is None:
#             return None
        
#         origin_x = self.map_info.origin.position.x
#         origin_y = self.map_info.origin.position.y
#         res = self.map_info.resolution
        
#         map_x = int((world_pos[0] - origin_x) / res)
#         map_y = int((world_pos[1] - origin_y) / res)
        
#         return (map_x, map_y)

#     # NEW: Funkcja pomocnicza do konwersji współrzędnych mapy na świat (środek komórki)
#     def map_to_world(self, map_pos):
#         if self.map_info is None:
#             return None
            
#         origin_x = self.map_info.origin.position.x
#         origin_y = self.map_info.origin.position.y
#         res = self.map_info.resolution
        
#         world_x = (map_pos[0] + 0.5) * res + origin_x
#         world_y = (map_pos[1] + 0.5) * res + origin_y
        
#         return np.array([world_x, world_y])

#     # NEW: Funkcja pomocnicza do sprawdzania, czy komórka jest zajęta
#     def is_occupied(self, map_x, map_y):
#         if self.map_data is None or self.map_info is None:
#             return False # Zakładamy, że wolne, jeśli nie ma mapy
            
#         width = self.map_info.width
#         height = self.map_info.height
        
#         # Sprawdzenie, czy komórka jest w granicach mapy
#         if 0 <= map_x < width and 0 <= map_y < height:
#             index = map_y * width + map_x
#             if self.map_data[index] >= OCCUPANCY_THRESHOLD:
#                 return True
        
#         # Poza granicami lub wolne
#         return False

#     def odom_callback(self, msg, robot_index):
#         self.robot_poses[robot_index] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        
#         orientation_q = msg.pose.pose.orientation
#         orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
#         _, _, yaw = self.euler_from_quaternion(orientation_list)
#         self.robot_yaws[robot_index] = yaw
        
#         linear_velocity = msg.twist.twist.linear.x
#         self.robot_velocities[robot_index] = np.array([linear_velocity * math.cos(yaw), linear_velocity * math.sin(yaw)])

#     def euler_from_quaternion(self, quaternion):
#         x, y, z, w = quaternion
#         t0 = +2.0 * (w * x + y * z)
#         t1 = +1.0 - 2.0 * (x * x + y * y)
#         roll_x = math.atan2(t0, t1)
     
#         t2 = +2.0 * (w * y - z * x)
#         t2 = +1.0 if t2 > +1.0 else t2
#         t2 = -1.0 if t2 < -1.0 else t2
#         pitch_y = math.asin(t2)
     
#         t3 = +2.0 * (w * z + x * y)
#         t4 = +1.0 - 2.0 * (y * y + z * z)
#         yaw_z = math.atan2(t3, t4)
     
#         return roll_x, pitch_y, yaw_z

#     def get_neighbors(self, robot_index):
#         neighbors = []
#         current_pos = self.robot_poses[robot_index]
#         if current_pos is None:
#             return neighbors

#         neighborhood_radius = self.get_parameter('neighborhood_radius').get_parameter_value().double_value

#         for i in range(NUM_ROBOTS):
#             if i == robot_index:
#                 continue
            
#             neighbor_pos = self.robot_poses[i]
#             if neighbor_pos is not None:
#                 distance = np.linalg.norm(current_pos - neighbor_pos)
#                 if distance < neighborhood_radius:
#                     neighbors.append(i)
#         return neighbors

#     def separation(self, robot_index, neighbors):
#         s = self.get_parameter('separation_strength').get_parameter_value().double_value
#         v_sep = np.zeros(2)
#         current_pos = self.robot_poses[robot_index]

#         if len(neighbors) == 0:
#             return v_sep

#         for neighbor_index in neighbors:
#             neighbor_pos = self.robot_poses[neighbor_index]
#             diff = current_pos - neighbor_pos
#             distance = np.linalg.norm(diff)
#             if distance > 0:
#                 v_sep += diff / (distance * distance)
        
#         return v_sep * s

#     def alignment(self, robot_index, neighbors):
#         a = self.get_parameter('alignment_strength').get_parameter_value().double_value
#         v_align = np.zeros(2)

#         if len(neighbors) == 0:
#             return v_align

#         for neighbor_index in neighbors:
#             if self.robot_velocities[neighbor_index] is not None:
#                 v_align += self.robot_velocities[neighbor_index]
        
#         v_align /= len(neighbors)
        
#         current_velocity = self.robot_velocities[robot_index]
#         if current_velocity is not None:
#              v_align = v_align - current_velocity

#         return v_align * a

#     def cohesion(self, robot_index, neighbors):
#         c = self.get_parameter('cohesion_strength').get_parameter_value().double_value
#         v_coh = np.zeros(2)
#         current_pos = self.robot_poses[robot_index]

#         if len(neighbors) == 0:
#             return v_coh

#         center_of_mass = np.zeros(2)
#         for neighbor_index in neighbors:
#             center_of_mass += self.robot_poses[neighbor_index]
        
#         center_of_mass /= len(neighbors)
        
#         v_coh = center_of_mass - current_pos
        
#         return v_coh * c

#     # NEW: Funkcja obliczająca wektor unikania przeszkód
#     def obstacle_avoidance(self, robot_index):
#         v_obs = np.zeros(2)
#         if self.map_info is None or self.robot_poses[robot_index] is None:
#             return v_obs

#         strength = self.get_parameter('obstacle_strength').get_parameter_value().double_value
#         detection_radius = self.get_parameter('obstacle_detection_radius').get_parameter_value().double_value
        
#         current_pos_world = self.robot_poses[robot_index]
#         current_pos_map = self.world_to_map(current_pos_world)
        
#         if current_pos_map is None:
#             return v_obs

#         # Przelicz promień wykrywania ze świata na komórki mapy
#         radius_cells = int(detection_radius / self.map_info.resolution)
        
#         # Sprawdź komórki w kwadratowym obszarze wokół robota
#         for dx in range(-radius_cells, radius_cells + 1):
#             for dy in range(-radius_cells, radius_cells + 1):
#                 if dx == 0 and dy == 0:
#                     continue # Pomiń komórkę, w której jest robot
                    
#                 check_map_x = current_pos_map[0] + dx
#                 check_map_y = current_pos_map[1] + dy

#                 if self.is_occupied(check_map_x, check_map_y):
#                     # Znaleziono przeszkodę, oblicz wektor odpychający
#                     obstacle_pos_world = self.map_to_world((check_map_x, check_map_y))
#                     diff = current_pos_world - obstacle_pos_world
#                     distance = np.linalg.norm(diff)
                    
#                     # Sprawdź, czy przeszkoda jest w okrągłym promieniu
#                     if distance > 0 and distance < detection_radius:
#                         # Siła odpychająca odwrotnie proporcjonalna do kwadratu dystansu
#                         v_obs += diff / (distance * distance) 
                        
#         return v_obs * strength


#     def apply_flocking_rules(self):
#         # Czekaj na dane odom ze wszystkich robotów
#         if any(p is None for p in self.robot_poses):
#             self.get_logger().info('Oczekiwanie na dane odometrii ze wszystkich robotów...', throttle_duration_sec=5)
#             for i in range(NUM_ROBOTS):
#                 msg = Twist()
#                 self.publishers_[i].publish(msg)
#             return

#         # NEW: Czekaj na dane mapy
#         if self.map_info is None:
#             self.get_logger().info('Oczekiwanie na dane z tematu /map...', throttle_duration_sec=5)
#             # Zatrzymujemy roboty, dopóki nie ma mapy
#             for i in range(NUM_ROBOTS):
#                 msg = Twist()
#                 self.publishers_[i].publish(msg)
#             return

#         for i in range(NUM_ROBOTS):
#             neighbors = self.get_neighbors(i)

#             v_sep = self.separation(i, neighbors)
#             v_align = self.alignment(i, neighbors)
#             v_coh = self.cohesion(i, neighbors)
#             v_obs = self.obstacle_avoidance(i) # NEW: Pobierz wektor unikania przeszkód

#             # UPDATED: Dodaj wektor unikania przeszkód do sumy
#             total_v = v_sep + v_align + v_coh + v_obs
            
#             total_v += np.random.rand(2) * 0.1 - 0.05

#             msg = Twist()
            
#             if self.robot_yaws[i] is None:
#                 continue

#             current_angle = self.robot_yaws[i]
#             target_angle = math.atan2(total_v[1], total_v[0])
            
#             angle_diff = target_angle - current_angle
#             while angle_diff > math.pi: angle_diff -= 2 * math.pi
#             while angle_diff < -math.pi: angle_diff += 2 * math.pi

#             angular_speed_gain = 2.0 
#             angular_speed = angular_speed_gain * angle_diff

#             if abs(angle_diff) > math.pi / 2:
#                 msg.linear.x = MAX_LINEAR_SPEED * 0.5
#             else:
#                 msg.linear.x = MAX_LINEAR_SPEED

#             msg.angular.z = max(min(angular_speed, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)

#             self.publishers_[i].publish(msg)

#         self.get_logger().info('Publishing cmd_vel (flocking + obstacle avoidance) to robots...', throttle_duration_sec=2)

# def main(args=None):
#     rclpy.init(args=args)
#     node = RobotController()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
# import random
# import numpy as np
# import math

# NUM_ROBOTS = 20
# MAX_LINEAR_SPEED = 2.0
# MAX_ANGULAR_SPEED = 2.0

# class RobotController(Node):
#     def __init__(self):
#         super().__init__('robot_controller')

#         self.declare_parameter('separation_strength', 0.7)
#         self.declare_parameter('alignment_strength', 0.8)
#         self.declare_parameter('cohesion_strength', 0.8)
#         self.declare_parameter('neighborhood_radius', 5.0)

#         self.publishers_ = []
#         self.odom_subscribers_ = []
#         self.robot_poses = [None] * NUM_ROBOTS
#         self.robot_velocities = [None] * NUM_ROBOTS
#         self.robot_yaws = [None] * NUM_ROBOTS

#         for i in range(NUM_ROBOTS):
#             topic_name = f'/robot_{i}/cmd_vel'
#             pub = self.create_publisher(Twist, topic_name, 10)
#             self.publishers_.append(pub)

#             odom_topic = f'/robot_{i}/odom'
#             self.odom_subscribers_.append(self.create_subscription(
#                 Odometry,
#                 odom_topic,
#                 lambda msg, index=i: self.odom_callback(msg, index),
#                 10))

#         self.timer_ = self.create_timer(0.1, self.apply_flocking_rules)
#         self.get_logger().info(f'Robot Controller Node has started for {NUM_ROBOTS} robots.')

#     def odom_callback(self, msg, robot_index):
#         self.robot_poses[robot_index] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        
#         orientation_q = msg.pose.pose.orientation
#         orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
#         _, _, yaw = self.euler_from_quaternion(orientation_list)
#         self.robot_yaws[robot_index] = yaw
        
#         linear_velocity = msg.twist.twist.linear.x
#         self.robot_velocities[robot_index] = np.array([linear_velocity * math.cos(yaw), linear_velocity * math.sin(yaw)])

#     def euler_from_quaternion(self, quaternion):
#         x, y, z, w = quaternion
#         t0 = +2.0 * (w * x + y * z)
#         t1 = +1.0 - 2.0 * (x * x + y * y)
#         roll_x = math.atan2(t0, t1)
     
#         t2 = +2.0 * (w * y - z * x)
#         t2 = +1.0 if t2 > +1.0 else t2
#         t2 = -1.0 if t2 < -1.0 else t2
#         pitch_y = math.asin(t2)
     
#         t3 = +2.0 * (w * z + x * y)
#         t4 = +1.0 - 2.0 * (y * y + z * z)
#         yaw_z = math.atan2(t3, t4)
     
#         return roll_x, pitch_y, yaw_z

#     def get_neighbors(self, robot_index):
#         neighbors = []
#         current_pos = self.robot_poses[robot_index]
#         if current_pos is None:
#             return neighbors

#         neighborhood_radius = self.get_parameter('neighborhood_radius').get_parameter_value().double_value

#         for i in range(NUM_ROBOTS):
#             if i == robot_index:
#                 continue
            
#             neighbor_pos = self.robot_poses[i]
#             if neighbor_pos is not None:
#                 distance = np.linalg.norm(current_pos - neighbor_pos)
#                 if distance < neighborhood_radius:
#                     neighbors.append(i)
#         return neighbors

#     def separation(self, robot_index, neighbors):
#         s = self.get_parameter('separation_strength').get_parameter_value().double_value
#         v_sep = np.zeros(2)
#         current_pos = self.robot_poses[robot_index]

#         if len(neighbors) == 0:
#             return v_sep

#         for neighbor_index in neighbors:
#             neighbor_pos = self.robot_poses[neighbor_index]
#             diff = current_pos - neighbor_pos
#             distance = np.linalg.norm(diff)
#             if distance > 0:
#                 v_sep += diff / (distance * distance)
        
#         return v_sep * s

#     def alignment(self, robot_index, neighbors):
#         a = self.get_parameter('alignment_strength').get_parameter_value().double_value
#         v_align = np.zeros(2)

#         if len(neighbors) == 0:
#             return v_align

#         for neighbor_index in neighbors:
#             if self.robot_velocities[neighbor_index] is not None:
#                 v_align += self.robot_velocities[neighbor_index]
        
#         v_align /= len(neighbors)
        
#         current_velocity = self.robot_velocities[robot_index]
#         if current_velocity is not None:
#              v_align = v_align - current_velocity

#         return v_align * a

#     def cohesion(self, robot_index, neighbors):
#         c = self.get_parameter('cohesion_strength').get_parameter_value().double_value
#         v_coh = np.zeros(2)
#         current_pos = self.robot_poses[robot_index]

#         if len(neighbors) == 0:
#             return v_coh

#         center_of_mass = np.zeros(2)
#         for neighbor_index in neighbors:
#             center_of_mass += self.robot_poses[neighbor_index]
        
#         center_of_mass /= len(neighbors)
        
#         v_coh = center_of_mass - current_pos
        
#         return v_coh * c

#     def apply_flocking_rules(self):
#         if any(p is None for p in self.robot_poses):
#             self.get_logger().info('Waiting for odometry data from all robots...', throttle_duration_sec=5)
#             for i in range(NUM_ROBOTS):
#                 msg = Twist()
#                 self.publishers_[i].publish(msg)
#             return

#         for i in range(NUM_ROBOTS):
#             neighbors = self.get_neighbors(i)

#             v_sep = self.separation(i, neighbors)
#             v_align = self.alignment(i, neighbors)
#             v_coh = self.cohesion(i, neighbors)

#             total_v = v_sep + v_align + v_coh
            
#             total_v += np.random.rand(2) * 0.1 - 0.05

#             msg = Twist()
            
#             if self.robot_yaws[i] is None:
#                 continue

#             current_angle = self.robot_yaws[i]
#             target_angle = math.atan2(total_v[1], total_v[0])
            
#             angle_diff = target_angle - current_angle
#             while angle_diff > math.pi: angle_diff -= 2 * math.pi
#             while angle_diff < -math.pi: angle_diff += 2 * math.pi

#             angular_speed_gain = 2.0 
#             angular_speed = angular_speed_gain * angle_diff

#             if abs(angle_diff) > math.pi / 2:
#                 msg.linear.x = MAX_LINEAR_SPEED * 0.5
#             else:
#                 msg.linear.x = MAX_LINEAR_SPEED

#             msg.angular.z = max(min(angular_speed, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)

#             self.publishers_[i].publish(msg)

#         self.get_logger().info('Publishing flocking cmd_vel to all robots...', throttle_duration_sec=2)

# def main(args=None):
#     rclpy.init(args=args)
#     node = RobotController()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
