# rnn-composer-recognition

Overview
 - Task: recognize a music composer from chord sequences (sequence classification).

Data
 - Sequences of chords / tokens representing chords.

Solution
 - Model: RNN (PyTorch). Important preprocessing step: isolate vocabulary to avoid data leakage.
 - Regularization: L2, LR scheduler, and early stopping for stable validation.
