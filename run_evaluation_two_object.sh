#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 1-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name evaluation
#SBATCH --gres gpu:ampere
#SBATCH -o evaluation_two_object.out


python generate_and_evaluate_image_two_object.py