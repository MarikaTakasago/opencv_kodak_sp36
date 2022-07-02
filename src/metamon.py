#!/usr/bin/env python

# fisheye to panorama
# for kodak PIXPRO sp360

import rospy
from sensor_msgs.msg import Image as ImageMsg
import cv2
import numpy as np
from cv_bridge import CvBridge
import math
from PIL import Image


class fish2pano():
    '''make a panorama image from a hemisphere fisheye image'''

    def __init__(self):

        # subscriver and publisher
        self.image_sub = rospy.Subscriber('/kodak/image', ImageMsg, self.callback)
        self.image_pub = rospy.Publisher('/metamon/image', ImageMsg, queue_size=1)
        self.bridge = CvBridge()
        self.data = ImageMsg()


        # image parameters
        self.input_width = 1024
        self.input_height = 1024
        self.output_width = 1024
        self.output_height = 1024

    def callback(self, data):
        '''callback function for subscriber'''
        rospy.loginfo_once("get image")
        # convert ROS image to CV image
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # print(cv_image.shape)

        # make a panorama image
        # pano_image = self.make_pano(cv_image)
        pano_image = self.make_pano_by_scratch(cv_image)

        # convert CV image to ROS image and publish to topic
        ros_image = self.bridge.cv2_to_imgmsg(pano_image, "rgb8")
        self.image_pub.publish(ros_image)
        # cv2.imshow("pano", ros_image)


    #make panorama image in 2 ways (tests)
    #first method (not use opencv function)
    def make_pano_by_scratch(self,cv_image):
        '''make a panorama image from a fisheye image'''

        # set params
        h = int(self.input_height / 2)
        w = int(3 * h)

        # set a panorama image
        pano_image = np.zeros((h, w, 3), np.uint8)
        # print(pano_image.shape)

        # convert CV image to numpy array
        cv_image = np.array(cv_image)
        # print(cv_image.shape)
        pano_image = np.array(pano_image)


        # make a panorama image
        for i in range(0,w):
            for j in range(0,h):
                r = j
                p = 2 * math.pi * i / w
                ix = h - r * math.cos(p)
                iy = h + r * math.sin(p)
                # if(int(ix)-1 == 0 ) or (int(iy)-1 == 0) : print(ix, iy)
                pano_image[j][i] = cv_image[int(iy-1)][int(ix-1)]

        # convert numpy array to CV image
        pano_image = cv2.cvtColor(pano_image, cv2.COLOR_BGR2RGB)
        # pano_image = cv2.resize(pano_image, (h*2,h))

        return pano_image

    #second method (use opencv function)
    def make_pano(self, cv_image):
        '''make a panorama image from a fisheye image'''

        # set a panorama image
        pano_image = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        # pano_image = cv2.resize(cv_image, (self.output_width, self.output_height))

        # convert CV image to numpy array
        # cv_image = np.array(cv_image)
        # pano_image = np.array(pano_image)

        K = self.K()
        D = self.D()
        cv2.fisheye.undistortImage(cv_image, K, D, pano_image) # undistort the image

        # convert numpy array to CV image
        pano_image = cv2.cvtColor(pano_image, cv2.COLOR_BGR2RGB)

        return pano_image

    def K(self):
        '''camera matrix'''
        return np.array([[0.805, 0.0, 20.0],
                         [0.0, 0.805, 20.0],
                         [0.0, 0.0, 1.0]])
    def D(self):
        '''distortion coefficients'''
        return np.array([-0.049, 0.152, 0.000, 0.000 ])


# main
def main():
    '''kodak to metamon'''

    rospy.init_node("metamon_streamer")

    metamon = fish2pano()

    print ("start")
    rospy.spin()


if __name__ == "__main__":
    main()
