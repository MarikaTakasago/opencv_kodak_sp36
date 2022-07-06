#!/usr/bin/env python

# panorama to cubemap
# why copikkyu ? because it's a copy of mimikkyu.

import rospy
from sensor_msgs.msg import Image as ImageMsg
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
from numba import jit
from PIL import Image

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
        # print(type(cv_image))
        # print(cv_image.shape)
        # CV image to PIL image
        pil_image = Image.fromarray(cv_image)
        # set params
        self.input_width = cv_image.shape[1]
        self.input_height = cv_image.shape[0]
        sqr = int(self.input_width / 4);
        output_width = int(sqr * 4)
        output_height = int(sqr * 3)
        # set a cubemap
        cube_image = np.zeros((output_height, output_width, 3), dtype=np.uint8) #black image
        cube_image = Image.fromarray(cube_image)

        # convert CV image to numpy array
        # cv_image = np.array(cv_image)
        # cube_image = np.array(cube_image)
        # print(type(cv_image))
        # print(cv_image.shape)

        # make a cubemap
        self.make_cube(pil_image, cube_image, self.input_width, self.input_height , output_width, output_height ,sqr)

        #convert PIL image to CV image
        cube_image = np.array(cube_image, dtype=np.uint8)
        cube_image = cv2.cvtColor(cube_image, cv2.COLOR_RGB2BGR)
        # convert CV image to ROS image and publish to topic
        ros_image = self.bridge.cv2_to_imgmsg(cube_image, "rgb8")
        self.image_pub.publish(ros_image)
        # cv2.imshow("cube", ros_image)

    # @jit #for faster processing
    def make_cube(self, pil_image, cube_image, input_width, input_height, output_width, output_height, sqr):
        ''' make a cubemap from a panorama image'''

        # make a cubemap
        # i,j : cubemap index(tx,ty)
        # x,y,z : panorama vector
        # u,v : panorama index

        # make array ?
        input_pixel = pil_image.load()
        output_pixel = cube_image.load()

def projection(theta,phi):
        if theta<0.615:
            return projectTop(theta,phi)
        Elif theta>2.527:
            return projectBottom(theta,phi)
        Elif phi <= pi/4 or phi > 7*pi/4:
            return projectLeft(theta,phi)
        Elif phi > pi/4 and phi <= 3*pi/4:
            return projectFront(theta,phi)
        Elif phi > 3*pi/4 and phi <= 5*pi/4:
            return projectRight(theta,phi)
        Elif phi > 5*pi/4 and phi <= 7*pi/4:
            return projectBack(theta,phi)

def projectLeft(theta,phi):
        x = 1
        y = tan(phi)
        z = cot(theta) / cos(phi)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Left",x,y,z)

def projectFront(theta,phi):
        x = tan(phi-pi/2)
        y = 1
        z = cot(theta) / cos(phi-pi/2)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Front",x,y,z)

def projectRight(theta,phi):
        x = -1
        y = tan(phi)
        z = -cot(theta) / cos(phi)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Right",x,-y,z)

def projectBack(theta,phi):
        x = tan(phi-3*pi/2)
        y = -1
        z = cot(theta) / cos(phi-3*pi/2)
        if z < -1:
            return projectBottom(theta,phi)
        if z > 1:
            return projectTop(theta,phi)
        return ("Back",-x,y,z)

def projectTop(theta,phi):
        # (a sin θ cos ø, a sin θ sin ø, a cos θ) = (x,y,1)
        a = 1 / cos(theta)
        x = tan(theta) * cos(phi)
        y = tan(theta) * sin(phi)
        z = 1
        return ("Top",x,y,z)

def projectBottom(theta,phi):
        # (a sin θ cos ø, a sin θ sin ø, a cos θ) = (x,y,-1)
        a = -1 / cos(theta)
        x = -tan(theta) * cos(phi)
        y = -tan(theta) * sin(phi)
        z = -1
        return ("Bottom",x,y,z)

# Convert coords in cube to image coords
# coords is a Tuple with the side and x,y,z coords
# Edge is the length of an Edge of the cube in pixels
def cubeToImg(coords,Edge):
    if coords[0]=="Left":
        (x,y) = (int(Edge*(coords[2]+1)/2), int(Edge*(3-coords[3])/2) )
    Elif coords[0]=="Front":
        (x,y) = (int(Edge*(coords[1]+3)/2), int(Edge*(3-coords[3])/2) )
    Elif coords[0]=="Right":
        (x,y) = (int(Edge*(5-coords[2])/2), int(Edge*(3-coords[3])/2) )
    Elif coords[0]=="Back":
        (x,y) = (int(Edge*(7-coords[1])/2), int(Edge*(3-coords[3])/2) )
    Elif coords[0]=="Top":
        (x,y) = (int(Edge*(3-coords[1])/2), int(Edge*(1+coords[2])/2) )
    Elif coords[0]=="Bottom":
        (x,y) = (int(Edge*(3-coords[1])/2), int(Edge*(5-coords[2])/2) )
    return (x,y)



        for i in xrange(output_height):
            for j in xrange(output_width):
                pixel = input_pixel[i,j]
                phi = i * 2 * pi / output_height
                theta = j * pi / output_width
                res = projection(theta,phi)

                (x,y) = cubeToImg(res,Edge)
                #if i % 100 == 0 and j % 100 == 0:
                #   print i,j,phi,theta,res,x,y
                if x >= output_height:
                    #print "x out of range ",x,res
                    x = output_height - 1
                if y >= output_width:
                    #print "y out of range ",y,res
                    y = output_width - 1
                output_pixel[x,y] = pixel
        # for i in range(output_height):
        #     face = int (i / sqr) # 0:back , 1:left, 2:front ,3:right
        #     if face == 2:
        #         rangej = range(0,sqr*3)
        #     else:
        #         rangej = range(0,sqr*2)
        #
        #     for j in rangej:
        #         if j<sqr:
        #             face = 4 #top
        #         elif j>=sqr*2:
        #             face = 5 #bottom
        #         else:
        #             face = face
        #
        #         # output image vector
        #         a = 2.0*i/sqr
        #         b = 2.0*j/sqr
        #         if face==0: # back
        #             (x,y,z) = (-1.0, 1.0-a, 3.0 - b)
        #         elif face==1: # left
        #             (x,y,z) = (a-3.0, -1.0, 3.0 - b)
        #         elif face==2: # front
        #             (x,y,z) = (1.0, a - 5.0, 3.0 - b)
        #         elif face==3: # right
        #             (x,y,z) = (7.0-a, 1.0, 3.0 - b)
        #         elif face==4: # top
        #             (x,y,z) = (b-1.0, a -5.0, 1.0)
        #         elif face==5: # bottom
        #             (x,y,z) = (5.0-b, a-5.0, -1.0)
        #
        #         # find polor coordinate
        #         r = math.sqrt(x**2 + y**2)
        #         theta = (math.atan2(y,x) + math.pi) / math.pi # calc normalize theta
        #         phi = (math.pi / 2 - math.atan2(z,r)) / math.pi # calc normalize phi
        #
        #         # find the pixel coordinate
        #         u = int(2.0 * sqr * theta)
        #         v = int(2.0 * sqr * phi)
        #
        #         # use bilinear interpolation between the four surrounding pixels
        #         # get the four surrounding pixels
        #         u2 = u + 1 # pixel to top right
        #         v2 = v + 1
        #         mu = u - u2 # fraction of pixel to top right
        #         mv = v - v2
        #         # four surrounding pixels
        #         p1 = input_pixel[u % input_width , np.clip(v,0,(input_height-1))]
        #         p2 = input_pixel[u2 % input_width , np.clip(v,0,(input_height-1))]
        #         p3 = input_pixel[u % input_width , np.clip(v2,0,(input_height-1))]
        #         p4 = input_pixel[u2 % input_width , np.clip(v2,0,(input_height-1))]
        #
        #         # bilinear interpolation
        #         (r,g,b) = (p1[0]*(1-mu)*(1-mv) + p2[0]*mu*(1-mv) + p3[0]*(1-mu)*mv + p4[0]*mu*mv,
        #                     p1[1]*(1-mu)*(1-mv) + p2[1]*mu*(1-mv) + p3[1]*(1-mu)*mv + p4[1]*mu*mv,
        #                     p1[2]*(1-mu)*(1-mv) + p2[2]*mu*(1-mv) + p3[2]*(1-mu)*mv + p4[2]*mu*mv)
        #         output_pixel[i,j] = (int(round(r)),int(round(g)),int(round(b)))


def main():
    '''metamon to mimikkyu'''
    rospy.init_node("copikkyu_streamer")

    mimikkyu = pano2cube()

    print("Let's go mimikkyu!")
    rospy.spin()

if __name__ == "__main__":
    main()
