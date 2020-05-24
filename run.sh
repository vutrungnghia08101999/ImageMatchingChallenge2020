jupyter nbconvert --to script main.ipynb
CUDA_VISIBLE_DEVICES=0 python main.py --train_scenes brandenburg_gate,grand_place_brussels,hagia_sophia_interior,palace_of_westminster,pantheon_exterior,taj_mahal,temple_nara_japan \
                                      --valid_gt valid/yfcc_test_pairs_with_gt.txt
# CUDA_VISIBLE_DEVICES=0 python main.py --train_scenes reichstag \
#                                       --valid_gt valid/yfcc_test_pairs_with_gt.txt

