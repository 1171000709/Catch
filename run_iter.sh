#!/bin/bash

datasets=("Openml_586" "Openml_589" "Openml_607" "Openml_616" "Openml_618" "Openml_620" "Openml_637" "airfoil" "amazon_employee" "Bikeshare_DC" "credit_a" "default_credit_card" "fertility_Diagnosis" "german_credit_24" "hepatitis" "Housing_Boston" "ionosphere" "lymphography" "megawatt1" "messidor_features" "PimaIndian" "spambase" "spectf" "winequality_red" "winequality_white")
for dataset in ${datasets[@]}
do
  python main.py --data $dataset --coreset 0 --cuda -1 --psm 0
done