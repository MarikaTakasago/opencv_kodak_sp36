#!/usr/bin/env python

# fisheye to panorama
# for kodak PIXPRO sp360

import rospy
from sensor_msgs.msg import Image as ImageMsg
import cv2
import numpy as np
from cv_bridge import CvBridge
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
        self.input_width = 1440
        self.input_height = 1440
        self.output_width = 1440
        self.output_height = 720

    def callback(self, data):
        '''callback function for subscriber'''
        print("subsubsub")
        # convert ROS image to CV image
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # make a panorama image
        pano_image = self.make_pano(cv_image)

        # convert CV image to ROS image and publish to topic
        ros_image = self.bridge.cv2_to_imgmsg(pano_image, "bgr8")
        self.image_pub.publish(ros_image)
        cv2.imshow("pano", ros_image)

    def calibrate():
        '''calibrate the camera'''
        # set

    def make_pano(self, cv_image):
        '''make a panorama image from a fisheye image'''

        # set a panorama image
        pano_image = cv2.resize(cv_image, (self.output_width, self.output_height))

        # convert CV image to numpy array
        # cv_image = np.array(cv_image)
        # pano_image = np.array(pano_image)

        K = self.K()
        D = self.D()
        cv2.fisheye.undistortImage(cv_image, K, D,pano_image) # undistort the image

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

        # # convert CV image to numpy array
        # cv_image = np.array(cv_image)
        #
        # # crop the image
        # cv_image = cv_image[0:self.input_height, 0:self.input_width]
        #
        # # resize the image
        # cv_image = cv2.resize(cv_image, (self.output_width, self.output_height))
        #
        # # make a panorama image
        # stitcher = cv2.Stitcher.create(0)
        # pano_image = stitcher.stitch(cv_image)
        #
        # # convert numpy array to CV image
        # pokan = cv2.cvtColor(pano_image, cv2.COLOR_BGR2RGB)
        #
        # # return the panorama image
        # # return pano_image
        return pokan


def main():
    '''kodak to metamon'''

    rospy.init_node("metamon_streamer")

    metamon = fish2pano()

    print ("start")
    rospy.spin()


if __name__ == "__main__":
    main()
