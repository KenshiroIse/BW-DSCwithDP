
# SCIS2025
# EfficientNet-B0 w8a8
# echo EfficientNet-B0 w8a8
# echo clean cifar-10
# claen cifar-10
#  python training_random_injection.py --model_name quanteffnet_w8a8_with_DP \
#  -b 50 -w 16 -i 224 --lr 0.1 --ep_clean 50 --ep_noise 0 -n 0 \
#  --save_dir efficient_b0_w8a8/clean_ep50_cifar10 --save_name clean_ep50_cifar10

#  python evaluation_with_noise.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch ./privacy_aware_dsc/efficient_b0_w8a8/clean_ep50_cifar10/clean_ep50_cifar10.pth.tar \
#  -b 50 -w 16 -i 224  --save_dir efficient_b0_w8a8/clean_ep50_cifar10 

# random block noise inejection training
# echo random block noise inejection training
# echo noise scale:5
# python training_random_injection.py --model_name quanteffnet_w8a8_with_DP \
#  -b 50 -w 16 -i 224 --lr 0.1 --ep_clean 25 --ep_noise 25 -n 5 \
#  --save_dir efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3 --save_name r_clean_25ep_noise_5_ep25_cifar10

# python evaluation_with_noise.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch ./privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v2/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 50 -w 16 -i 224  --save_dir efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v2

# python privacy_dsc.py --model_name quanteffnet_w8a8_with_DP --model_series b0\
#  --quant_arch ./privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v2/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 50 -w 16 -i 224 -n 5 --save_dir efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v2

# # EfficientNet-B3 w8a8
# echo EfficientNet-B3 w8a8
# # random block noise inejection training
# echo random block noise inejection training
# echo noise scale:5
# python training_random_injection.py --model_name quanteffnet_w8a8_b3_with_DP \
#  -b 40 -w 16 -i 300 --lr 0.01 --ep_clean 25 --ep_noise 25 -n 5 \
#  --save_dir efficient_b3_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3 --save_name r_clean_25ep_noise_5_ep25_cifar10

# python evaluation_with_noise.py --model_name quanteffnet_w8a8_b3_with_DP \
#  --quant_arch ./privacy_aware_dsc/efficient_b3_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v2/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 40 -w 16 -i 300  --save_dir efficient_b3_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v2

# python privacy_dsc.py --model_name quanteffnet_w8a8_b3_with_DP --model_series b3 \
#  --quant_arch ./privacy_aware_dsc/efficient_b3_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v2/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 40 -w 16 -i 300 -n 5 --save_dir efficient_b3_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v2

 # clean cifar-10
# echo clean cifar-10
#  python training_random_injection.py --model_name quanteffnet_w8a8_b3_with_DP \
#  -b 40 -w 16 -i 300 --lr 0.1 --ep_clean 50 --ep_noise 0 -n 0 \
#  --save_dir efficient_b3_w8a8/clean_ep50_cifar10_v1 --save_name clean_ep50_cifar10

#  python evaluation_with_noise.py --model_name quanteffnet_w8a8_b3_with_DP \
#  --quant_arch ./privacy_aware_dsc/efficient_b3_w8a8/clean_ep50_cifar10_v1/clean_ep50_cifar10.pth.tar \
#  -b 40 -w 16 -i 300  --save_dir efficient_b3_w8a8/clean_ep50_cifar10_v1

# 卒論
# imagenet-100
# EfficientNet-B0 w8a8
#  python training_random_injection.py --model_name quanteffnet_w8a8_with_DP \
#  -b 50 -w 16 -i 224 --lr 0.1 --ep_clean 100 --ep_noise 0 -n 0 \
#  --save_dir efficient_b0_w8a8/clean_ep100_imagenet100 --save_name clean_ep100_imagenet100

#  python evaluation_with_noise.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch ./privacy_aware_dsc/efficient_b0_w8a8/clean_ep100_imagenet100/clean_ep100_imagenet100.pth.tar \
#  -b 50 -w 16 -i 224  --save_dir efficient_b0_w8a8/clean_ep100_imagenet100

# python training_random_injection.py --model_name quanteffnet_w8a8_with_DP \
#  -b 50 -w 16 -i 224 --lr 0.1 --ep_clean 50 --ep_noise 50 -n 5 \
#  --save_dir efficient_b0_w8a8/clean_50ep_noise_5_50ep_imagenet100 --save_name clean_50ep_noise_5_50ep_imagenet100

# python evaluation_with_noise.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch ./privacy_aware_dsc/efficient_b0_w8a8/clean_50ep_noise_5_50ep_imagenet100/clean_50ep_noise_5_50ep_imagenet100.pth.tar \
#  -b 50 -w 16 -i 224  --save_dir efficient_b0_w8a8/clean_50ep_noise_5_50ep_imagenet100

# python privacy_dsc.py --model_name quanteffnet_w8a8_with_DP --model_series b0\
#  --quant_arch ./privacy_aware_dsc/efficient_b0_w8a8/clean_50ep_noise_5_50ep_imagenet100/clean_50ep_noise_5_50ep_imagenet100.pth.tar \
#  -b 50 -w 16 -i 224 -n 5 --save_dir efficient_b0_w8a8/clean_50ep_noise_5_50ep_imagenet100

# EfficientNet-B3 w8a8/
# python training_random_injection.py --model_name quanteffnet_w8a8_b3_with_DP \
#  -b 40 -w 16 -i 300 --lr 0.1 --ep_clean 50 --ep_noise 50 -n 5 \
#  --save_dir efficient_b3_w8a8/clean_50ep_noise_5_50ep_imagenet100 --save_name clean_50ep_noise_5_50ep_imagenet100

# python evaluation_with_noise.py --model_name quanteffnet_w8a8_b3_with_DP \
#  --quant_arch ./privacy_aware_dsc/efficient_b3_w8a8/clean_50ep_noise_5_50ep_imagenet100/clean_50ep_noise_5_50ep_imagenet100.pth.tar \
#  -b 40 -w 16 -i 300  --save_dir efficient_b3_w8a8/clean_50ep_noise_5_50ep_imagenet100

# python privacy_dsc.py --model_name quanteffnet_w8a8_b3_with_DP --model_series b3 \
#  --quant_arch ./privacy_aware_dsc/efficient_b3_w8a8/clean_50ep_noise_5_50ep_imagenet100/clean_50ep_noise_5_50ep_imagenet100.pth.tar \
#  -b 40 -w 16 -i 300 -n 5 --save_dir efficient_b3_w8a8/clean_50ep_noise_5_50ep_imagenet100


# python training_random_injection.py --model_name quanteffnet_w8a8_b3_with_DP \
#  -b 40 -w 16 -i 300 --lr 0.1 --ep_clean 100 --ep_noise 0 -n 0 \
#  --save_dir efficient_b3_w8a8/clean_100ep_imagenet100 --save_name clean_100ep_imagenet100

# python evaluation_with_noise.py --model_name quanteffnet_w8a8_b3_with_DP \
#  --quant_arch ./privacy_aware_dsc/efficient_b3_w8a8/clean_100ep_imagenet100/clean_100ep_imagenet100.pth.tar \
#  -b 40 -w 16 -i 300  --save_dir efficient_b3_w8a8/clean_100ep_imagenet100


# echo whitebox
# echo split_point:2
# # noiseなし学習
# python model_inversion_train.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 50 -i 224 --split_point 2 -e 30 -l 1.5 --env whitebox --save_dir block_2

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_2/final_inversion.pth \
#  -b 50 -i 224 --split_point 2  --env whitebox --save_dir block_2

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_2/final_inversion.pth \
#  -b 50 -i 224 --split_point 2 --noise_flag True --noise_scale 5 --env whitebox --save_dir block_2

# #  noiseあり学習
# python model_inversion_train.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 50 -i 224 --split_point 2 --noise_flag True --noise_scale 5 -e 30 -l 1.5 --env whitebox --save_dir block_2_noise

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_2_noise/final_inversion.pth \
#  -b 50 -i 224 --split_point 2  --env whitebox --save_dir block_2_noise

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_2_noise/final_inversion.pth \
#  -b 50 -i 224 --split_point 2 --noise_flag True --noise_scale 5 --env whitebox --save_dir block_2_noise

# echo split_point:4
# # noiseなし学習
# python model_inversion_train.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 50 -i 224 --split_point 4 -e 30 -l 2.0 --env whitebox --save_dir block_4

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_4/final_inversion.pth \
#  -b 50 -i 224 --split_point 4  --env whitebox --save_dir block_4

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_4/final_inversion.pth \
#  -b 50 -i 224 --split_point 4 --noise_flag True --noise_scale 5 --env whitebox --save_dir block_4

# #  noiseあり学習
# python model_inversion_train.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 50 -i 224 --split_point 4 --noise_flag True --noise_scale 5 -e 30 -l 2.0 --env whitebox --save_dir block_4_noise

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_4_noise/final_inversion.pth \
#  -b 50 -i 224 --split_point 4  --env whitebox --save_dir block_4_noise

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_4_noise/final_inversion.pth \
#  -b 50 -i 224 --split_point 4 --noise_flag True --noise_scale 5 --env whitebox --save_dir block_4_noise

# echo split_point:6
# # noiseなし学習
# python model_inversion_train.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 50 -i 224 --split_point 6 -e 30 -l 2.5 --env whitebox --save_dir block_6

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_6/final_inversion.pth \
#  -b 50 -i 224 --split_point 6  --env whitebox --save_dir block_6

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_6/final_inversion.pth \
#  -b 50 -i 224 --split_point 6 --noise_flag True --noise_scale 5 --env whitebox --save_dir block_6

# #  noiseあり学習
# python model_inversion_train.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 50 -i 224 --split_point 6 --noise_flag True --noise_scale 5 -e 30 -l 2.5 --env whitebox --save_dir block_6_noise

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_6_noise/final_inversion.pth \
#  -b 50 -i 224 --split_point 6  --env whitebox --save_dir block_6_noise

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/whitebox/block_6_noise/final_inversion.pth \
#  -b 50 -i 224 --split_point 6 --noise_flag True --noise_scale 5 --env whitebox --save_dir block_6_noise

# test
echo blackbox
# echo block_4
# # ノイズなし学習
# python model_inversion_train.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  -b 30 -i 224 --split_point 4 -e 15 --env blackbox --save_dir block_4

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/blackbox/block_4/final_inversion.pth \
#  -b 30 -i 224 --split_point 4 --env blackbox --save_dir block_4

# python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
#  --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
#  --inv_arch /home/naoki/work/Privacy-DSC/MIA/blackbox/block_4/final_inversion.pth \
#  -b 30 -i 224 --split_point 4 --noise_flag True --noise_scale 5 --env blackbox --save_dir block_4

echo block_2
python model_inversion_train.py --model_name quanteffnet_w8a8_with_DP \
 --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
 -b 30 -i 224 --split_point 2 -e 15 -l 2 --env blackbox --save_dir block_2

python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
 --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
 --inv_arch /home/naoki/work/Privacy-DSC/MIA/blackbox/block_2/final_inversion.pth \
 -b 30 -i 224 --split_point 2 --env blackbox --save_dir block_2

python model_inversion_attack.py --model_name quanteffnet_w8a8_with_DP \
 --quant_arch /home/naoki/work/Privacy-DSC/privacy_aware_dsc/efficient_b0_w8a8/r_clean_25ep_noise_5_ep25_cifar10_v3/r_clean_25ep_noise_5_ep25_cifar10.pth.tar \
 --inv_arch /home/naoki/work/Privacy-DSC/MIA/blackbox/block_2/final_inversion.pth \
 -b 30 -i 224 --split_point 2 --noise_flag True --noise_scale 5 --env blackbox --save_dir block_2

