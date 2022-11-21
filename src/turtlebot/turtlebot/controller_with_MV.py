#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import numpy as np
#import controller_publisher_utils as utils
from .controller_publisher_utils import *
import time

TEMPLATE_SIZE = 150

class Move_robot(Node):

    def __init__(self):
        
        super().__init__("Move_robot")
        
        # topic that the turtlebot3 movement subscribes to
        self.topic_port = "/cmd_vel"
        
        # publishing a twist message on topic object
        self.message_publisher = self.create_publisher(Twist, self.topic_port, 10) #(msg type, topic, queuesize(?))
        
        # how often the callback function is called
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.move_callback)
        
        # twist message object
        self.vel_msg = Twist()
        self.vel_msg.linear.x = 0.1
        self.vel_msg.angular.z = 0.0
        self.turn = self.vel_msg.angular.z
        
        #self.r = self.create_rate(30)
        
        self.IMAGE_WIDTH = 640
        self.IMAGE_HEIGHT = 480
        self.framecounter = 0
        self.cam = cv2.VideoCapture(0)
        self.trackbarvals = [[160., 350.], [490., 350.], [0.,480.], [650., 480.]]
        #self.inittrackbas = initializeTrackbars(self.trackbarvals, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)

        self.count = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('media/testing.avi', self.fourcc, 1//timer_period, (1152, 576))

    
        self.load_ants_template = cv2.imread('/home/ubuntu/ROS_car_project/src/turtlebot/turtlebot/load_ants.jpg')
        self.load_ants_template = cv2.resize(self.load_ants_template, (TEMPLATE_SIZE, TEMPLATE_SIZE))

        self.deploy_ants_template = cv2.imread('/home/ubuntu/ROS_car_project/src/turtlebot/turtlebot/deploy_ants.jpg')
        self.deploy_ants_template = cv2.resize(self.deploy_ants_template, (TEMPLATE_SIZE, TEMPLATE_SIZE))

        self.can_pause = True

    def move_callback(self):
        
        self.framecounter += 1
        if self.cam.get(cv2.CAP_PROP_FRAME_COUNT) == self.framecounter:
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.framecounter = 0

        _, img = self.cam.read()  # GET THE IMAGE
        img = cv2.resize(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))  # RESIZE
        
        self.turn, img_stack, speed, stop, pause = pipeline(img, self.trackbarvals, self.turn, self.load_ants_template, self.deploy_ants_template)
        
        #self.vel_msg.angular.z = self.turn
        #self.vel_msg.linear.x = speed
        
        self.vel_msg.angular.z = 0.0
        self.vel_msg.linear.x = 0.0
        

        #cv2.imshow("video", img)
        #both = cv2.addWeighted(warp, 0.5, hist, 0.5, 0.0)
        #cv2.imshow("both", both)
                
        #if cv2.waitKey(1) & 0xFF == ord('q'):  # press q To exit
        #    self.finish
        
        #called every 0.1 seconds by self.timer
        self.message_publisher.publish(self.vel_msg)    
        #displayed after every called
        self.get_logger().info('linear speed {}, angular speed {}'.format(self.vel_msg.linear.x, self.vel_msg.angular.z))
        #self.vel_msg.linear.x += 0.02

        
        self.out.write(img_stack)
        if stop:
            self.finish()
        if pause and self.can_pause:
            self.vel_msg.linear.x = 0.0
            self.vel_msg.angular.z = 0.0
            self.message_publisher.publish(self.vel_msg)
            for i in range(0, 100):
                self.out.write(img_stack)
                time.sleep(0.1)
            self.vel_msg.linear.x = speed
            self.vel_msg.angular.z = self.turn
            self.message_publisher.publish(self.vel_msg)
            self.can_pause = False

             
    
    def finish(self):
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = 0.0
        self.message_publisher.publish(self.vel_msg)
        print("stopped")
        self.out.release()
        print("released video")
        self.cam.release()
        print("released camera")
        cv2.destroyAllWindows()
        self.shutdown_turtlebot()
        print("shutdown turtlebot")
        exit() 
        
    
        
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
    
