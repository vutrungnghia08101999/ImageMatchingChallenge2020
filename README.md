NOTE
=====
- Make sure a scene has the following files:
  - points.csv {DataFrame}:
    - df['point_id'] = [2, 4, 7, ....]
  - gray.npy, rgb.npy {dictionary}:
    - len(dic[2]) = 2
    - dic[2][0].shape = 64 x 64 || 32 x 32 x 3
    - dic[2][1].shape = 64 x 64 || 32 x 32 x 3
  - 50000_50000_gray.npy, 50000_50000_rgb.npy {np.array}:
    - s.shape = (100000, 3)
    - (a, b, c) = s[0]:
      - a.shape = (64, 64) || (32, 32, 3)
      - b.shape = (64, 64) || (32, 32, 3)
      - c = 0 || 1

HOW TO USE
===
- run main.ipynb
