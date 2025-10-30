#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import math
import random

NUM_ROBOTS = 10

NEIGHBOR_RADIUS = 3.5
SEPARATION_DISTANCE = 2.0

MAX_LINEAR_SPEED = 0.5
MAX_ANGULAR_SPEED = 1.0

WEIGHT_SEPARATION = 2.5
WEIGHT_ALIGNMENT = 1.5
WEIGHT_COHESION = 1.0

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        self.robot_states = [None] * NUM_ROBOTS
        
        self.publishers_ = []
        self.subscribers_ = []

        for i in range(NUM_ROBOTS):
            pub_topic = f'/robot_{i}/cmd_vel'
            pub = self.create_publisher(Twist, pub_topic, 10)
            self.publishers_.append(pub)
            
            sub_topic = f'/robot_{i}/odom'
            sub = self.create_subscription(
                Odometry,
                sub_topic,
                lambda msg, robot_id=i: self.odom_callback(msg, robot_id),
                10)
            self.subscribers_.append(sub)

        self.timer_ = self.create_timer(0.1, self.compute_flocking)
        
        self.get_logger().info(f'Robot Controller Node has started for {NUM_ROBOTS} robots.')
        self.get_logger().info('Waiting for initial odometry data from all robots...')

    def odom_callback(self, msg, robot_id):
        self.robot_states[robot_id] = msg

    def get_heading_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def compute_flocking(self):
        
        if any(state is None for state in self.robot_states):
            return

        for i in range(NUM_ROBOTS):
            
            current_state = self.robot_states[i]
            current_pos = current_state.pose.pose.position
            current_q = current_state.pose.pose.orientation
            current_heading = self.get_heading_from_quaternion(current_q)
            current_vel = current_state.twist.twist.linear 
            current_vel_vec = np.array([current_vel.x, current_vel.y])

            separation_vec = np.array([0.0, 0.0])
            alignment_vec = np.array([0.0, 0.0])
            cohesion_vec = np.array([0.0, 0.0])
            neighbor_count = 0

            for j in range(NUM_ROBOTS):
                if i == j:
                    continue
                
                neighbor_state = self.robot_states[j]
                neighbor_pos = neighbor_state.pose.pose.position
                
                dist = math.sqrt((current_pos.x - neighbor_pos.x)**2 + 
                                 (current_pos.y - neighbor_pos.y)**2)
                
                if 0 < dist < NEIGHBOR_RADIUS:
                    neighbor_count += 1
                    
                    if dist < SEPARATION_DISTANCE:
                        diff = np.array([current_pos.x - neighbor_pos.x, 
                                         current_pos.y - neighbor_pos.y])
                        diff /= (dist * dist) 
                        separation_vec += diff

                    neighbor_vel = neighbor_state.twist.twist.linear
                    alignment_vec += np.array([neighbor_vel.x, neighbor_vel.y])
                    
                    cohesion_vec += np.array([neighbor_pos.x, neighbor_pos.y])

            msg = Twist()
            
            if neighbor_count > 0:
                
                separation_vec /= neighbor_count
                alignment_vec /= neighbor_count
                cohesion_vec /= neighbor_count
                
                steer_sep = separation_vec * WEIGHT_SEPARATION
                
                steer_align = (alignment_vec - current_vel_vec) * WEIGHT_ALIGNMENT
                
                center_of_mass = cohesion_vec 
                vec_to_center = (center_of_mass - np.array([current_pos.x, current_pos.y]))
                steer_coh = vec_to_center * WEIGHT_COHESION

                final_linear_vec = steer_sep + steer_align + steer_coh
                
                desired_heading = math.atan2(final_linear_vec[1], final_linear_vec[0])
                desired_speed = np.linalg.norm(final_linear_vec)
                
                angle_to_turn = desired_heading - current_heading
                
                while angle_to_turn > math.pi: angle_to_turn -= 2 * math.pi
                while angle_to_turn < -math.pi: angle_to_turn += 2 * math.pi
                
                angular_vel = angle_to_turn * 2.0
                
                linear_vel = desired_speed * (1.0 - abs(angle_to_turn) / math.pi)

                msg.angular.z = max(min(angular_vel, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)
                msg.linear.x = max(min(linear_vel, MAX_LINEAR_SPEED), 0.0)
            
            else:
                msg.linear.x = 0.2
                msg.angular.z = random.uniform(-0.5, 0.5) 

            self.publishers_[i].publish(msg)

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