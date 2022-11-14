#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import random

class Move_robot(Node):

    def __init__(self):
        
        super().__init__("Move_robot")
        
        # topic that the turtlebot3 movement subscribes to
        self.topic_port = "/cmd_vel"
        
        # publishing a twist message on topic object
        self.message_publisher = self.create_publisher(Twist, self.topic_port, 10) #(msg type, topic, queuesize(?))
        
        # how often the callback function is called
        timer_period = 1  
        self.timer = self.create_timer(timer_period, self.move_callback)
        
        # twist message object
        self.vel_msg = Twist()
        self.vel_msg.linear.x = 0.2
        self.vel_msg.angular.z = 0.1
        
        self.r = self.create_rate(30)
        
    def move_callback(self):
        #called every 0.1 seconds by self.timer
        self.message_publisher.publish(self.vel_msg)    
        #displayed after every called
        self.get_logger().info('linear speed {}, angular speed {}'.format(self.vel_msg.linear.x, self.vel_msg.angular.z))
        #self.vel_msg.linear.x += 0.02
        
        
    def shutdown_turtlebot(self):
        self.message_publisher.publish(Twist())
        

def main(args=None):
    
    # To initialize ROS2 node
    rclpy.init(args=args)
    
    # call node
    move = Move_robot()
    
    # not sure
    rclpy.spin(move)

    # note sure
    move.destroy_node()
    
    # note sure
    rclpy.shutdown()

if __name__ == "__main__":
    
    main()
    
