#!/bin/bash
dataset=sepsis_survival
python main.py --data $dataset --coreset 1 --cuda 6 --psm 0
dataset=AP_Omentum_Ovary
python main.py --data $dataset --coreset 1 --cuda 6 --psm 0
dataset=gisette
python main.py --data $dataset --coreset 1 --cuda -1 --psm 0
dataset=covtype_small
python main.py --data $dataset --coreset 1 --cuda -1 --psm 0
dataset=accelerometer
python main.py --data $dataset --coreset 1 --cuda -1 --psm 0
dataset=covtype
python main.py --data $dataset --coreset 1 --cuda -1 --psm 0
dataset=poker_hand
python main.py --data $dataset --coreset 1 --cuda -1 --psm 0
dataset=BNG_wisconsin
python main.py --data $dataset --coreset 1 --cuda -1 --psm 0
dataset=airlines_train
python main.py --data $dataset --coreset 1 --cuda -1 --psm 0
