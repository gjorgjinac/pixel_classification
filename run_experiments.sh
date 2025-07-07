#!/bin/bash

# Loop over binarize_labels values
for binarize_labels in 1 0
do
  # Loop over do_oversampling values
  for do_oversampling in "none" "over" "under"
  do
    # Loop over fold values
    for fold in {0..9}
    do
      echo "Running with binarize_labels=$binarize_labels, do_oversampling=$do_oversampling, fold=$fold"
      tsp python R2_train_model_single_fold.py --use_mlflow 0 --binarize_labels $binarize_labels --fold $fold --do_oversampling $do_oversampling
    done
  done
done