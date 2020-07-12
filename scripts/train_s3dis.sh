DATA_ROOT = /home/yc/chen/
TRAIN_SET = $DATA_ROOT/data/point_cloud/s3dis
python train.py $TRAIN_SET \
--dataset_name 's3dis'  --model_name 'KPConv_Net'  --task 'segmentation'  --seed 2048 \
--workers 8  --epochs 100  --batch_size 6  --epoch_steps 500 \
--optimizer 'Adam'  --lr 0.01  --momentum 0.98 --beta 0.999 --weight_decay 0 \
--decay_style 'LambdaLR' --decay_basenum 0.98  --decay_step 50 --decay_rate 0.7 \
--offsets_loss 'fitting'  --offset_decay 0.1 \
--KP_influence 'linear' \
--KP_extent 1.2 \
--deform_fitting_mode 'point2point' \
--deform_fitting_power 1.0 \
--deform_lr_factor 0.1 \
--repulse_extent 1.2 \
--use_batch_norm True \
--batch_norm_momentum 0.02 \
--conv_radius = 2.5 \
--deform_radius = 6.0 \
--density_parameter 5.0 \
--aggregation_mode 'sum' \
--fixed_kernel_points 'center' \
--in_features_dim 5 \
--in_points_dim 3  \
--modulated False \
--grad_clip_norm 100.0 \
--batch_average_loss False \
--in_radius 1.5 \
--segmentation_ratio 1.0 \
--first_features_dim 128 \
--density_parameter 3.0 \
--num_pts 40960 \
--num_cls 13 \
--num_kpts 15 \
--sub_grid_size 0.03 \
--val_split 5 \
--segloss_balance = 'none' \
--augment_scale_anisotropic True \
--augment_symmetries [True,False,False] \
--augment_rotation 'vertical' \
--augment_scale_min 0.8 \
--augment_scale_max 1.2 \
--augment_noise 0.001 \
--augment_occulusion 'none' \
--augment_occulusion_ratio 0.2 \
--augment_occulusion_num 1 \
--augment_color 0.8 \
--is_debug True

# ['simple',
# # 'resnetb',
# # 'resnetb_strided',
# # 'resnetb',
# # 'resnetb_strided',
# # 'resnetb',
# # 'resnetb_strided',
# # 'resnetb',
# # 'resnetb_strided',
# # 'resnetb',
# # 'nearest_upsample',
# # 'unary',
# # 'nearest_upsample',
# # 'unary',
# # 'nearest_upsample',
# # 'unary',
# # 'nearest_upsample',
# # 'unary']
# S3DIS  Semantic3d

#architecture = ['simple',
#                    'resnetb',
#                    'resnetb_strided',
#                    'resnetb',
#                    'resnetb',
#                    'resnetb_strided',
#                    'resnetb_deformable',
#                    'resnetb_deformable',
#                    'resnetb_deformable_strided',
#                    'resnetb_deformable',
#                    'resnetb_deformable',
#                    'resnetb_deformable_strided',
#                    'resnetb_deformable',
#                    'resnetb_deformable',
#                    'nearest_upsample',
#                    'unary',
#                    'nearest_upsample',
#                    'unary',
#                    'nearest_upsample',
#                    'unary',
#                    'nearest_upsample',
#                    'unary']