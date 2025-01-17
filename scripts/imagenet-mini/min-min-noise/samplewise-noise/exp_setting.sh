#!/bin/bash



# Exp Setting
export config_path=configs/imagenet-mini
export config_train_path=configs/imagenet-mini/train
export dataset_path=~/workspace/imagenet
export dataset_type=ImageNetMini
export poison_dataset_type=PoisonImageNetMini
export attack_type=min-min
export perturb_type=samplewise
export base_version=resnet18
export gen_epoch=10
export epsilon=16
export step_size=1.6
export num_steps=20
export train_step=300
export universal_stop_error=0.1
export universal_train_target='train_dataset'
export exp_args=${dataset_type}-eps=${epsilon}-se=${universal_stop_error}-base_version=${base_version}
export exp_path=experiments/imagenet-mini/${attack_type}_${perturb_type}/${exp_args}
export scripts_path=scripts/imagenet-mini/${attack_type}-noise/${perturb_type}-noise
