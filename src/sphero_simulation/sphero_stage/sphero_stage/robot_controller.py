import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import random
import numpy as np
import math

NUM_ROBOTS = 20
MAX_LINEAR_SPEED = 3.0
MAX_ANGULAR_SPEED = 3.0

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.declare_parameter('separation_strength', 0.5)
        self.declare_parameter('alignment_strength', 0.8)
        self.declare_parameter('cohesion_strength', 1.0)
        self.declare_parameter('neighborhood_radius', 5.0)

        self.publishers_ = []
        self.odom_subscribers_ = []
        self.robot_poses = [None] * NUM_ROBOTS
        self.robot_velocities = [None] * NUM_ROBOTS
        self.robot_yaws = [None] * NUM_ROBOTS

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

        self.timer_ = self.create_timer(0.1, self.apply_flocking_rules)
        self.get_logger().info(f'Robot Controller Node has started for {NUM_ROBOTS} robots.')

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

        if len(neighbors) == 0:
            return v_sep

        for neighbor_index in neighbors:
            neighbor_pos = self.robot_poses[neighbor_index]
            diff = current_pos - neighbor_pos
            distance = np.linalg.norm(diff)
            if distance > 0:
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

    def apply_flocking_rules(self):
        if any(p is None for p in self.robot_poses):
            self.get_logger().info('Waiting for odometry data from all robots...', throttle_duration_sec=5)
            for i in range(NUM_ROBOTS):
                msg = Twist()
                self.publishers_[i].publish(msg)
            return

        for i in range(NUM_ROBOTS):
            neighbors = self.get_neighbors(i)

            v_sep = self.separation(i, neighbors)
            v_align = self.alignment(i, neighbors)
            v_coh = self.cohesion(i, neighbors)

            total_v = v_sep + v_align + v_coh
            
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

        self.get_logger().info('Publishing flocking cmd_vel to all robots...', throttle_duration_sec=2)

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
