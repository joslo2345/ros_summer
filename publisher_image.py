#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
def pub_image():
    VideoRaw = rospy.Publisher('Image', Image, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)
    cam = cv2.VideoCapture(0)
    if (not cam.isOpened()):
        return
    while not rospy.is_shutdown():
        meta, frame = cam.read()
        msg_frame = CvBridge().cv2_to_imgmsg(frame, "bgr8")
        #rospy.loginfo(msg_frame)
        VideoRaw.publish(msg_frame)
        rate.sleep()

if __name__ == '__main__':
    try:
        pub_image()
    except rospy.ROSInterruptException:
        pass
