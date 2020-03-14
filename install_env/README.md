Install extract_patches environment
```
conda env create -f extract_patches.yml
```
Install hardnet environment
```
conda env create -f hardnet.yml
```
Install structure-from-motion environment (benchmark)
```
conda env create -f conda_env_yuhe.yml

git clone https://github.com/colmap/colmap
cd colmap
mkdir build
cp ../__download__ build/__download__
conda activate sfm
python scripts/python/build.py --build_path ./build --colmap_path ./
```