#!/usr/bin/env python

# panorama to cubemap
# why mimikkyu ? because it's a 'cu'be image maker


import rospy
from sensor_msgs.msg import Image as ImageMsg
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
from omnicv import fisheyeImgConv

class pano2cube():
    '''make a cubemap from a panorama image'''

    def __init__(self):
        #subscriber and publisher
        self.image_sub = rospy.Subscriber('/kodak/metamon', ImageMsg, self.callback)
        self.image_pub = rospy.Publisher('/kodak/mimikkyu', ImageMsg, queue_size=1)
        self.bridge = CvBridge()
        self.data = ImageMsg()

        # image parameters
        self.input_width = 0
        self.input_height = 0

    def callback(self, data):
        '''callback function for panorama image'''
        rospy.loginfo_once("mimikkyu : received image")

        # convert ROS image to CV image
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # print(type(cv_image))
        # print(cv_image.shape)
        # set params
        self.input_width = cv_image.shape[1]
        self.input_height = cv_image.shape[0]
        sqr = int(self.input_width / 4);
        output_width = int(sqr * 4)
        output_height = int(sqr * 3)
        # set a cubemap
        # cube_image = np.zeros((output_height, output_width, 3), dtype=np.uint8) #black image

        # convert CV image to numpy array
        # cv_image = np.array(cv_image)
        # cube_image = np.array(cube_image)
        # print(type(cv_image))
        # print(cv_image.shape)

        # make a cubemap
        mapper = fisheyeImgConv()
        cube_image = mapper.equirect2cubemap(cv_image,side=sqr,dice=1) #dice : 1 = dice shape / 0 = line shape
        # print("mimikkyu!")

        # convert numpy array to CV image
        # cube_image = cv2.cvtColor(cube_image, cv2.COLOR_RGB2BGR)
        # convert CV image to ROS image and publish to topic
        ros_image = self.bridge.cv2_to_imgmsg(cube_image, "rgb8")
        self.image_pub.publish(ros_image)
        # cv2.imshow("cube", ros_image)



def main():
    '''metamon to mimikkyu'''
    rospy.init_node("copikkyu_streamer")

    mimikkyu = pano2cube()

    print("Let's go mimikkyu!")
    rospy.spin()

if __name__ == "__main__":
    main()
