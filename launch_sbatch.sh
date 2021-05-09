#!/bin/sh

# --nodelist=napoli114
sbatch --partition=napoli-gpu --gres=gpu:2 --mem=32GB --time=7-00:00:00 ./queue_train.sh
