# opencv_kodak_sp_360

ROS package for streaming Kodak sp_360 (not 4k)

use Wifi to capture images

## Dependency

Numba : `pip3 install numba` for high speed forloop (use Numba0.55.2 or older)

[omnicv](https://github.com/kaustubh-sadekar/OmniCV-Lib) : convert to cubemap


## Befor running

Wifi setting : go `script` and run `./connetct_wifi.sh kodak`

## Description
`kodak_sp360.py` : get fisheye image from camera

`metamon.py` : fisheye image to panorama image

`mimikkyu.py` : panorama image to cube image
