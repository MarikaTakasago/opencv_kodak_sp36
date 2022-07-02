# opencv_kodak_sp_360

ROS package for streaming Kodak sp_360 (not 4k)
use Wifi to capture images

## Befor running
Import Numba for high speed forloop : `pip3 install numba`
Wifi setting : Move`script`and run `./connetct_wifi.sh kodak`

## Description
`kodak_sp360.py` : get fisheye image from camera
`metamon.py` : fisheye image to panorama image
