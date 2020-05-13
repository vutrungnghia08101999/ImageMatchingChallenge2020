DATA_DIR="/mnt/DATA/Phototourism-dataset/train/"
OUT_DIR="/mnt/DATA/cvpr_2020/"
EXPORT="/mnt/DATA/patch_datasets/sift_scale/"

# training scenes

# SCENE="brandenburg_gate"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="buckingham_palace"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

SCENE="colosseum_exterior"
python run.py \
    --scene_path $DATA_DIR$SCENE"/dense" \
    --out_path $OUT_DIR$SCENE \
    --patch_per_point 2 \
    --export_path $EXPORT$SCENE \

# SCENE="grand_place_brussels"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="hagia_sophia_interior"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="notre_dame_front_facade"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="palace_of_westminster"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="pantheon_exterior"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="prague_old_town_square"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="taj_mahal"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="temple_nara_japan"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="trevi_fountain"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="westminster_abbey"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# validation scenes
# DATA_DIR="/mnt/DATA/Phototourism-dataset/val/"

# SCENE="reichstag"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="sacre_coeur"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \

# SCENE="st_peters_square"
# python run.py \
#     --scene_path $DATA_DIR$SCENE"/dense" \
#     --out_path $OUT_DIR$SCENE \
#     --patch_per_point 2 \
#     --export_path $EXPORT$SCENE \
