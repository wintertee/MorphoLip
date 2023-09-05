#!/bin/bash

for max_epochs in 200 300
do
   python main.py --morpho infinity --name inf_AdamW_$max_epochs --gpu 0 --optimizer AdamW --lr 3e-3 --max_epochs $max_epochs --gpu 0
done
 
