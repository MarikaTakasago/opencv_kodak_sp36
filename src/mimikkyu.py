#!/usr/bin/env python

# panorama to cubemap
# why mimikkyu ? because it's a 'cu'be map

import rospy
from sensor_msgs.msg import Image as ImageMsg
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
from numba import jit

class pano2cube():
    '''make a cubemap from a panorama image'''

    def __init__(self):
        #subscriber and publisher
        self.image_sub = rospy.Subscriber('/metamon/image', ImageMsg, self.callback)
        self.image_pub = rospy.Publisher('/mimikkyu/image', ImageMsg, queue_size=1)
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
        # cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        # cv_image = cv2.resize(cv_image, (1280,640))
        # print(cv_image.shape)

        # set params
        self.input_width = cv_image.shape[1]
        self.input_height = cv_image.shape[0]
        sqr = self.input_width / 4;
        output_width = int(sqr * 3)
        output_height = int(sqr * 2)
        # set a cubemap
        cube_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        # convert CV image to numpy array
        cv_image = np.array(cv_image)
        cube_image = np.array(cube_image)

        # make a cubemap
        cube_image = self.make_cube(cv_image, cube_image, self.input_width, self.input_height , output_width, output_height ,sqr)

        # convert CV image to ROS image and publish to topic
        ros_image = self.bridge.cv2_to_imgmsg(cube_image, "rgb8")
        self.image_pub.publish(ros_image)
        # cv2.imshow("cube", ros_image)

    @jit #for faster processing
    def make_cube(self, cv_image, cube_image, input_width, input_height, output_width, output_height, sqr):
        ''' make a cubemap from a panorama image'''

        # make a cubemap
        for i in range(1, output_height+1):
            for j in range(1, output_width+1):
                tx = 0.0
                ty = 0.0
                x = 0.0
                y = 0.0
                z = 0.0
                pix = 0

                if i < sqr + 1: # top half of the cubemap
                    if j < sqr + 1: # top left = left
                        # if j == 0: print("top left")
                        tx = j
                        ty = i
                        x = tx - 0.5 * sqr
                        y = 0.5 * sqr
                        z = (ty - 0.5 * sqr)

                    elif j < sqr * 2 + 1: # top middle = front
                        # if j == sqr+1: print("top middle")
                        tx = j - sqr
                        ty = i
                        x = 0.5 * sqr
                        y = (tx - 0.5 * sqr) * -1
                        z = (ty - 0.5 * sqr)
                        pix = 1

                    else: # top right = right
                        # if j == sqr * 2 + 1: print("top right")
                        tx = j - sqr * 2
                        ty = i
                        x = (tx - 0.5 * sqr) * -1
                        y = -0.5 * sqr
                        z = (ty - 0.5 * sqr)

                else: # bottom half of the cubemap
                    if j < sqr + 1: # bottom left = back
                        tx = j
                        ty = i - sqr
                        x = int(-0.5 * sqr)
                        y = int(tx - 0.5 * sqr)
                        z = int(ty - 0.5 * sqr)

                    elif j < sqr * 2 + 1: # bottom middle = bottom
                        tx = j - sqr
                        ty = i - sqr
                        x = (ty - 0.5 * sqr) * -1
                        y = (tx - 0.5 * sqr) * -1
                        z = 0.5 * sqr

                    else: # bottom right = top
                        tx = j - sqr * 2
                        ty = i - sqr
                        x = ty - 0.5 * sqr
                        y = (tx - 0.5 * sqr) * -1
                        z = (-0.5 * sqr)

                # find polor coordinate
                r = math.sqrt(x**2 + y**2 + z**2)
                # calc normalize theta
                if y < 0:
                    theta = math.atan2(y, x) * -1 / (2 * math.pi)
                else:
                    theta = (math.pi + (math.pi - math.atan2(y, x))) / (2 * math.pi)
                # normalize phi
                phi = (math.pi - math.acos(z / r)) / math.pi

                # find the pixel coordinate
                x_pixel = int(theta * output_width)
                y_pixel = int(phi * output_height)

                # catch overflow
                if x_pixel >= input_width:
                    x_pixel -= input_width
                if y_pixel >= input_height:
                    y_pixel -= input_height
                if pix ==1:
                    print(x_pixel, y_pixel)

                # copy the pixel
                cube_image[i-1][j-1] = cv_image[y_pixel][x_pixel]

        # convert numpy array to CV image
        cube_image = cv2.cvtColor(cube_image, cv2.COLOR_BGR2RGB)

        return cube_image




def main():
    '''metamon to mimikkyu'''
    rospy.init_node("mimikkyu_streamer")

    mimikkyu = pano2cube()

    print("Let's go mimikkyu!")
    rospy.spin()

if __name__ == "__main__":
    main()
