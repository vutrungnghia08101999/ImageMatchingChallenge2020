Generate exactly 2 patches per 3D point
===

Step 1: generate_rgb_data.py

Step 2: rgb_to_gray.py

Step 3: compress_gray_data.py

Step 4: evaluate_gray_data.py

For evaluation step
===
Here are FPRR95 on 100.000 pairs and average neg_distance/pos_distance on 5000 triplets of Brown dataset.

* liberty: 0.00230000 - 3.4153578028304805

* notredame: 0.00646000 - 2.9300811551070987

* yosemite: 0.02378000 - 3.3890420449252354

NOTE
===
This code uses CPU only
