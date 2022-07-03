# opencv_kodak_sp_360

ROS package for streaming Kodak sp_360 (not 4k)

use Wifi to capture images

## Befor running
Import Numba for high speed forloop : `pip3 install numba`

(use Numba0.55.2 or older)

Wifi setting : go `script` and run `./connetct_wifi.sh kodak`

## Description
`kodak_sp360.py` : get fisheye image from camera

`metamon.py` : fisheye image to panorama image

`mimikkyu.py` : panorama image to cube image
