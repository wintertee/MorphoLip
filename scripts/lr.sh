#!/bin/bash

for lr in 1e-3 3e-3 1e-4
do
   python main.py --morpho infinity --name inf_AdamW_$lr --gpu 0 --optimizer AdamW --lr $lr
done

# python main.py --morpho dilation --name dila_norelu --gpu 1 --dataset CIFAR10 --norelu
