#!/bin/bash
# baseline
python3 train.py --model Proto --dataset meta_iNat --opt adam --lr 1e-3 --gamma .5 --epoch 20 --stage 5 --val_epoch 2 --weight_decay 5e-4 --nesterov --train_way 30 --train_shot 1 --train_transform_type 1 --test_transform_type 2 --test_shot 1 --no_val --gpu_num 1
python3 train.py --model Proto --dataset meta_iNat --opt adam --lr 1e-3 --gamma 5e-1 --epoch 20 --stage 5 --val_epoch 2 --weight_decay 5e-4 --nesterov --train_way 20 --train_shot 5 --train_transform_type 1 --test_transform_type 2 --test_shot 5 --no_val --gpu_num 1
python3 train.py --model FRN --dataset meta_iNat --opt adam --lr 1e-3 --gamma .5 --epoch 20 --stage 5 --val_epoch 2 --weight_decay 5e-4 --train_way 20 --train_shot 5 --train_transform_type 1 --test_transform_type 2 --test_shot 1 5 --no_val --gpu_num 1
# TDM
python3 train.py --TDM --noise --model Proto --dataset meta_iNat --opt adam --lr 1e-3 --gamma .5 --epoch 20 --stage 5 --val_epoch 2 --weight_decay 5e-4 --nesterov --train_way 30 --train_shot 1 --train_transform_type 1 --test_transform_type 2 --test_shot 1 --no_val --gpu_num 1
python3 train.py --TDM --noise --model Proto --dataset meta_iNat --opt adam --lr 1e-3 --gamma 5e-1 --epoch 20 --stage 5 --val_epoch 2 --weight_decay 5e-4 --nesterov --train_way 20 --train_shot 5 --train_transform_type 1 --test_transform_type 2 --test_shot 5 --no_val --gpu_num 1
python3 train.py --TDM --noise --model FRN --dataset meta_iNat --opt adam --lr 1e-3 --gamma .5 --epoch 20 --stage 5 --val_epoch 2 --weight_decay 5e-4 --train_way 20 --train_shot 5 --train_transform_type 1 --test_transform_type 2 --test_shot 1 5 --no_val --gpu_num 1
