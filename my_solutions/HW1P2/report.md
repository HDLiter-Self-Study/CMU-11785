# CMU 11785 HW1P2: Phoneme Classification - Project Report

## Introduction
In this project, we used a multilayer perceptron (MLP) to classify phonemes from audio data. We also conducted a comprehensive hyperparameter search to optimize the model's performance. The goal was to achieve high accuracy in phoneme classification tasks.

## Dataset
The dataset is provided by the CMU 11785 course on [Kaggle](https://www.kaggle.com/competitions/11785-hw1p2-f24/submissions). It contains MFCC features extracted from audio recordings of various phonemes.

## Results
The best model achieved an accuracy of 85.2% on the kaggle test set (as a late submission since I am a self-studier). The best hyperparameters were found as below:

**Best Hyperparameters:**

| Hyperparameter         | Value                                 |
|-----------------------|---------------------------------------|
| activation            | `gelu`                                |
| batch normalization   | `true`                                |
| batch size            | 8,192                                 |
| context window        | 25                                    |
| frequency masking     | 0.1                                   |
| hidden dropout        | 0.15                                  |
| hidden shape          | `custom_9_layers_cylinder_1`          |
| label smoothing       | 0.1                                   |
| layer-wise dropout    | `true`                                |
| learning rate         | 0.0008                                |
| max total parameters  | 20,000,000                            |
| number of epochs      | 16                                    |
| number of hidden layers| 9                                    |
| optimizer             | `adamw`                               |
| scheduler             | `plateau`                             |
| time masking          | 0                                     |


`custom_9_layers_cylinder_1` is a cylindrical architecture with 9 hidden layers, each has 1460 neurons (the number of neurons is determined to maximize the model's capacity while keeping the total parameter count within the specified limit).


We also experimented with advanced network architectures and techniques (suggested by my AI assistant) such as: focal loss, mix up and specialized classification heads, but none of them improved the performance significantly. (Actually the results were quite disappointing.😢)


It ranked about 150/300 in the kaggle competition, I believe there is still room for improvement, but I decided to devote more time to the following homework assignments and projects, so I will not continue to optimize this model. Good luck to the future me~😁

## Lessons Learned (mostly on hyperparameter search)
1. **Do not dive too deep until you have covered all the available techniques**: At the beginning, I spent a lot of time on tuning the learning rate and layer depth, but later I realized that I had not even tried batch normalization yet. After adding batch normalization, the model's performance improved significantly, but since batch normalization allows for larger learning rates and deeper networks, I had to re-tune the learning rate and layer depth again.😇 So I suggest to first try all the available techniques and then do a comprehensive hyperparameter search.
2. **Make clean updates and make your code reproducible**: Some changes I made required me to reformat the model's code quite a bit (e.g. using mixup, adding specialized classification heads, etc.), which made it hard to track the changes and reproduce the results.😓 Next time I will try to modularize the code better and make clean updates, so that I can easily reproduce the results and track the updates.
3. **Use a systematic approach for hyperparameter search**: I used a mixture of random search and grid search for hyperparameter tuning, but I found that it was not very efficient. I was always worried and suspicious that I might miss some important hyperparameters or combinations, and the current optimum was a local optimum.😵 So I wasted a lot of time doing redundant experiments. Next time I will try to use a more systematic approach, such as Bayesian optimization or Hyperband, to search for the best hyperparameters more efficiently. (Since my autoDL server cannot connect to wandb and perform wandb sweep, maybe next time I'll try optuna.)
4. **Record every possible hyperparameter**: At the beginning of the hyperparameter search, I didn't record the arguments for scheduler and simply recorded the scheduler type. But later I tuned them for a couple of times and found that they did affect the performance significantly. But I had no record of the arguments I used before, so the actual optimum was lost in the sea of search space.😢 So next time I will make sure to record every hyperparameter and its value, so that I can easily reproduce the results and track the changes. (The reason I didn't record the scheduler's parameters was that every scheduler has its own parameter sets, which made it hard to organize the hyperparameter sweep. Next time I will serialize them in the name of scheduler, like 'plateau_0.25_2_0.025' for  "factor=0.25, patience=2, threshold=0.025").
