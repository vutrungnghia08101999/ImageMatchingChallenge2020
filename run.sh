jupyter nbconvert --to script main.ipynb
# CUDA_VISIBLE_DEVICES=0 python main.py --train_scenes brandenburg_gate,colosseum_exterior,grand_place_brussels,hagia_sophia_interior,palace_of_westminster,pantheon_exterior,prague_old_town_square,st_peters_square,taj_mahal,trevi_fountain,temple_nara_japan \
#                                       --valid_gt valid/yfcc_test_pairs_with_gt.txt
CUDA_VISIBLE_DEVICES=0 python main.py --train_scenes buckingham_palace \
                                      --valid_gt valid/yfcc_buckingham_palace.txt \
				      --sinkhorn_iterations  5 \
				      --weight /home/hieu123/nghia/models/checkpoint_20-05-27_06:23:06_0.pth
# CUDA_VISIBLE_DEVICES=0 python main.py --train_scenes brandenburg_gate,grand_place_brussels,hagia_sophia_interior,palace_of_westminster,pantheon_exterior,taj_mahal,temple_nara_japan \
#                                      --valid_gt valid/yfcc_test_pairs_with_gt.txt \
#                                      --weights /home/hieu123/nghia/models/checkpoint_20-05-24_22:53:59_1.pth
