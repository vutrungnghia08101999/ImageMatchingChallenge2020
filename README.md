HIEU
===
Input:
 - images.bin: keypoinys, 3dpoints
 - points.bin: 3dpoints, image_ids, 1 keypoint for each image_id

Output:
```bash
{
    75: {
        'xys': # N x 2,
        'points3D_ids': # N,
        'name': # 92753909_16247242.jpg,
        'confident_scores': # N,
        'descriptors': # N x 128,
        'height': # 606,
        'width': # 817
    },
    ...
    ...
    ...
}
```

NGHIA
===========
Input: Hieu's output

Output:
```bash
{
    (14, 24): {
        'matches': [(idx1, idx2), .... ] # list
        'keypoints': (N x 2, M x 2) # tuple
        'descriptors': (N x 128, M x 128) # tuple 
        'scores': (N, M) # tuple 
        '3dpoints': (N, M) # tuple
        'shape': ({'width': 1056, 'height': 780}, {'width': 1032, 'height': 637}) # tuple
        'name': ('06857713_2570944910.jpg', '34537245_34002183.jpg') # tuple
    }
    ...
    ...
    ...
}
```
