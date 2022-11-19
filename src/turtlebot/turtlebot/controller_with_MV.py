#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import .controller_publisher_utils as utils
from collections import Counter



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
        self.vel_msg.angular.z = 0.0
        
        #self.r = self.create_rate(30)
        
        self.IMAGE_WIDTH = 640
        self.IMAGE_HEIGHT = 480
        self.framecounter = 0
        self.cam = cv2.VideoCapture(0)
        self.trackbarvals = [30, 300, 0, 480]
        self.inittrackbars = utils.initializeTrackbars(self.trackbarvals, self.IMAGE_WIDTH, self.IMAGE_HEIGHTMAGE_HEIGHT)

        
    def move_callback(self):
        
        self.framecounter += 1
        if self.cam.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        _, img = self.cam.read()  # GET THE IMAGE
        img = cv2.resize(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))  # RESIZE
        
        hist, turn, warp = utils.pipeline(img)
        self.vel_msg.angular.z = turn
        
        cv2.imshow("video", img)
        both = cv2.addWeighted(warp, 0.5, hist, 0.5, 0.0)
        cv2.imshow("both", both)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q To exit
            self.finish
        
        #called every 0.1 seconds by self.timer
        self.message_publisher.publish(self.vel_msg)    
        #displayed after every called
        self.get_logger().info('linear speed {}, angular speed {}'.format(self.vel_msg.linear.x, self.vel_msg.angular.z))
        #self.vel_msg.linear.x += 0.02
    
    
    def finish(self):
        self.cam.release()
        cv2.destroyAllWindows()
        
        
    
        
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
    
