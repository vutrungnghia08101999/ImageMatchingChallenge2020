NOTE
===========
- Go throught list of 3d points and get exactly 2 matched patches
- The 2d keypoint is at the center of the patch.
- The size of each patch could be 64 x 64 or 32 x 32 x 3
- Make sure the scale of the image is [0, 255] NOT [0, 1]
- Make sure the images are rgb not bgr

How to use
===========
1. create environment
2. change configs.yml and run.sh file (do not change batch_size and test_size)
3. ./run.sh
