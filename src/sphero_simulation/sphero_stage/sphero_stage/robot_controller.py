#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
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
WEIGHT_NAVIGATION = 0.7
WEIGHT_OBSTACLE = 3.0
WEIGHT_WANDER = 0.3

GOAL_POSITION = np.array([-1.0, 1.0])
ARRIVAL_RADIUS = 0.5
OBSTACLE_SENSE_RADIUS = 1.0


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        self.robot_states = [None] * NUM_ROBOTS
        self.map_data = None
        self.map_info = None

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

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
        self.get_logger().info('Waiting for initial odometry and map data...')

    def odom_callback(self, msg, robot_id):
        self.robot_states[robot_id] = msg

    def map_callback(self, msg):
        self.map_data = msg.data
        self.map_info = msg.info
        self.get_logger().info('Map data received!')

    def get_heading_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def world_to_map(self, world_x, world_y):
        if self.map_info is None:
            return None, None
        
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        res = self.map_info.resolution
        
        map_x = int((world_x - origin_x) / res)
        map_y = int((world_y - origin_y) / res)
        return map_x, map_y

    def map_to_world(self, map_x, map_y):
        if self.map_info is None:
            return None, None
            
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        res = self.map_info.resolution
        
        world_x = (map_x + 0.5) * res + origin_x
        world_y = (map_y + 0.5) * res + origin_y
        return world_x, world_y

    def is_obstacle(self, map_x, map_y):
        if self.map_data is None:
            return False
        
        if 0 <= map_x < self.map_info.width and 0 <= map_y < self.map_info.height:
            idx = map_y * self.map_info.width + map_x
            return self.map_data[idx] > 50
        return False

    def compute_flocking(self):
        
        if self.map_data is None or any(state is None for state in self.robot_states):
            return

        for i in range(NUM_ROBOTS):
            current_state = self.robot_states[i]
            current_pos = current_state.pose.pose.position
            current_pos_vec = np.array([current_pos.x, current_pos.y])
            current_q = current_state.pose.pose.orientation
            current_heading = self.get_heading_from_quaternion(current_q)
            current_vel = current_state.twist.twist.linear 
            current_vel_vec = np.array([current_vel.x, current_vel.y])

            separation_vec = np.array([0.0, 0.0])
            alignment_vec = np.array([0.0, 0.0])
            cohesion_vec = np.array([0.0, 0.0])
            steer_wander = np.array([0.0, 0.0])
            
            neighbor_count = 0

            for j in range(NUM_ROBOTS):
                if i == j: continue
                
                neighbor_state = self.robot_states[j]
                neighbor_pos = neighbor_state.pose.pose.position
                
                dist = math.sqrt((current_pos.x - neighbor_pos.x)**2 + 
                                 (current_pos.y - neighbor_pos.y)**2)
                
                if 0 < dist < NEIGHBOR_RADIUS:
                    neighbor_count += 1
                    
                    if dist < SEPARATION_DISTANCE:
                        diff = current_pos_vec - np.array([neighbor_pos.x, neighbor_pos.y])
                        diff /= (dist * dist)
                        separation_vec += diff

                    neighbor_vel = neighbor_state.twist.twist.linear
                    alignment_vec += np.array([neighbor_vel.x, neighbor_vel.y])
                    
                    cohesion_vec += np.array([neighbor_pos.x, neighbor_pos.y])

            if neighbor_count > 0:
                separation_vec = (separation_vec / neighbor_count) * WEIGHT_SEPARATION
                
                avg_vel = alignment_vec / neighbor_count
                alignment_vec = (avg_vel - current_vel_vec) * WEIGHT_ALIGNMENT
                
                center_of_mass = cohesion_vec / neighbor_count
                vec_to_center = (center_of_mass - current_pos_vec)
                cohesion_vec = vec_to_center * WEIGHT_COHESION
            else:
                wander_angle = random.uniform(-0.8, 0.8)
                new_heading = current_heading + wander_angle
                steer_wander = np.array([math.cos(new_heading), math.sin(new_heading)])
                steer_wander = steer_wander * WEIGHT_WANDER

            vec_to_goal = GOAL_POSITION - current_pos_vec
            dist_to_goal = np.linalg.norm(vec_to_goal)
            
            if dist_to_goal > ARRIVAL_RADIUS:
                steer_nav = (vec_to_goal / dist_to_goal) * MAX_LINEAR_SPEED
            else:
                steer_nav = vec_to_goal * (MAX_LINEAR_SPEED / ARRIVAL_RADIUS)
            
            steer_nav = steer_nav * WEIGHT_NAVIGATION
            
            steer_obs = np.array([0.0, 0.0])
            obs_count = 0
            
            check_radius_cells = int(OBSTACLE_SENSE_RADIUS / self.map_info.resolution)
            current_map_x, current_map_y = self.world_to_map(current_pos.x, current_pos.y)

            if current_map_x is not None:
                for u in range(-check_radius_cells, check_radius_cells + 1):
                    for v in range(-check_radius_cells, check_radius_cells + 1):
                        if u == 0 and v == 0: continue
                        
                        cell_x, cell_y = current_map_x + u, current_map_y + v
                        
                        if self.is_obstacle(cell_x, cell_y):
                            obs_world_x, obs_world_y = self.map_to_world(cell_x, cell_y)
                            obs_pos_vec = np.array([obs_world_x, obs_world_y])
                            dist_to_obs = np.linalg.norm(current_pos_vec - obs_pos_vec)
                            
                            if dist_to_obs < OBSTACLE_SENSE_RADIUS:
                                diff = current_pos_vec - obs_pos_vec
                                diff /= (dist_to_obs * dist_to_obs)
                                steer_obs += diff
                                obs_count += 1
                                
            if obs_count > 0:
                steer_obs = (steer_obs / obs_count) * WEIGHT_OBSTACLE

            final_linear_vec = (separation_vec + 
                                alignment_vec + 
                                cohesion_vec + 
                                steer_nav + 
                                steer_obs + 
                                steer_wander)
            
            msg = Twist()
            
            desired_heading = math.atan2(final_linear_vec[1], final_linear_vec[0])
            desired_speed = np.linalg.norm(final_linear_vec)
            
            angle_to_turn = desired_heading - current_heading
            while angle_to_turn > math.pi: angle_to_turn -= 2 * math.pi
            while angle_to_turn < -math.pi: angle_to_turn += 2 * math.pi
            
            angular_vel = angle_to_turn * 2.0
            linear_vel = desired_speed * (1.0 - abs(angle_to_turn) / math.pi)

            msg.angular.z = max(min(angular_vel, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)
            msg.linear.x = max(min(linear_vel, MAX_LINEAR_SPEED), 0.0)
            
            if dist_to_goal < 0.1:
                msg.linear.x = 0.0
                msg.angular.z = 0.0

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