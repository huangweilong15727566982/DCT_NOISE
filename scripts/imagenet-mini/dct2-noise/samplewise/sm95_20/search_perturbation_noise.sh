#!/bin/bash

# Load Exp Settings
source exp_setting.sh


# Remove previous files
echo $exp_path


# Search Universal Perturbation and build datasets
cd ../../../../../
pwd
rm -rf $exp_name
python3 perturbation.py --config_path             $config_path       \
                        --exp_name                $exp_path          \
                        --version                 $base_version      \
                        --train_data_path         $dataset_path      \
                        --train_data_type         $dataset_type      \
                        --test_data_path          $dataset_path      \
                        --test_data_type          $dataset_type      \
                        --noise_shape             5200 3 224 224    \
                        --epsilon                 $epsilon           \
                        --num_steps               $num_steps         \
                        --channel                 $channel           \
                        --step_size               $step_size         \
                        --gen_epoch               $gen_epoch         \
                        --attack_type             $attack_type       \
                        --perturb_type            $perturb_type      \
                        --train_step              $train_step        \
                        --train_batch_size        256                \
                        --eval_batch_size         256               \
                        --universal_train_target  $universal_train_target\
                        --universal_stop_error    $universal_stop_error\
