import cv2
import numpy as np

def get_SIFT_keypoints(sift, img, max_kp=10000):
    # convert to gray-scale and compute SIFT keypoints
    keypoints = sift.detect(img, None)
    # print(len(keypoints))
    response = np.array([kp.response for kp in keypoints])
    respSort = np.argsort(response)[::-1]

    pt = np.array([kp.pt for kp in keypoints])[respSort]
    size = np.array([kp.size for kp in keypoints])[respSort]
    
    # get unique kp
    _, unique = np.unique(pt, axis=0, return_index=True)
    unique = np.sort(unique)[:max_kp]
    pt = pt[unique]
    size = size[unique]
    response = response[unique]
    return pt, size, response

