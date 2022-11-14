import cv2
from rclpy.node import Node 
import rclpy

'''def main():
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)
            count+=1
            if cv2.waitKey(1) & count > 300:
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
'''

class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')
        timerPeriod = 0.1
        self.timer = self.create_timer(timerPeriod, self.timer_callback)
        self.count = 0
        

    def timer_callback(self):
        self.cap = cv2.VideoCapture(0)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output.mp4', self.fourcc, 20.0, (640, 480))
        ret, frame = self.cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)
            # write the flipped frame
            self.out.write(frame)
            self.count+=1
            if cv2.waitKey(1) & self.count > 3000:
                exit()

def main():
    rclpy.init()

    video_recorder = VideoRecorder()

    rclpy.spin(video_recorder)

    video_recorder.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()