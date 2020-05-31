# jupyter nbconvert --to script main.ipynb
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name iou0_1_9layers_without_weightGT_max_2048kpts_has_3dp_has_descriptors_fix_dataset \
                                      --train_scenes brandenburg_gate,colosseum_exterior,grand_place_brussels,hagia_sophia_interior,palace_of_westminster,pantheon_exterior,prague_old_town_square,taj_mahal,trevi_fountain,temple_nara_japan \
                                      --valid_gt valid/yfcc_test_pairs_with_gt.txt
#                                      --weights /home/hieu123/nghia/models/checkpoint_20-05-29_03:40:12_2.pth \
#                                      --lr 0.000001
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_name iou0_1_9layers_without_weightGT_max_2048kpts_has_3dp_has_descriptors \
#                                       --train_scenes reichstag \
#                                       --valid_gt valid/yfcc_reichstag.txt
#                                       --sinkhorn_iterations  20
# CUDA_VISIBLE_DEVICES=0 python main.py --train_scenes brandenburg_gate,grand_place_brussels,hagia_sophia_interior,palace_of_westminster,pantheon_exterior,taj_mahal,temple_nara_japan \
#                                      --valid_gt valid/yfcc_test_pairs_with_gt.txt \
#                                      --weights /home/hieu123/nghia/models/checkpoint_20-05-24_22:53:59_1.pth
