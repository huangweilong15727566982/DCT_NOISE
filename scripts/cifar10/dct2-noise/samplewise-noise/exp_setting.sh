#!/bin/bash
# Exp Setting
export config_path=configs/cifar10
export dataset_type=CIFAR10
export poison_dataset_type=PoisonCIFAR10
export attack_type=dct2
export perturb_type=samplewise
export base_version=resnet18
export sm=95
export channel=10
export epsilon=0.07178979876918531
export step_size=0.007178979876918531
export num_steps=20
export universal_stop_error=0.01
export universal_train_target='train_dataset'
export exp_args=${dataset_type}-sm=${sm}_${channel}-base_version=${base_version}
export exp_path=experiments/cifar10/${attack_type}_${perturb_type}/${exp_args}
export scripts_path=scripts/cifar10/${attack_type}-noise/${perturb_type}-noise/sm${sm}_${channel}
