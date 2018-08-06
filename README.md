# Investigation in Video Analysis
This codebase contains implementations of some ideas in video analysis that I found interesting.  

The structure of the code is inherited from
[cs-230-stanford](https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision)


## Convolutional Drift Networks for Video Classification

This part contains implementation of [CDN](https://arxiv.org/abs/1711.01201)

`driftTransform.py` implements transformation of a video in a dataset into a vector which encodes the video and stores as numpy arrays. Once transformed videos are available.
`python train.py` to train, (check  [cs-230-stanford](https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision) for details)
