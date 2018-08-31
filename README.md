# Flower or Crops CNN classification

### Overview
This repository contains the training/evaluation scripts for a Convolutional Neural Network made for detecting flowers in the middle of a plantation. The goal is to detect [_ipomoea grandifolia_](http://lmgtfy.com/?q=ipomoea+grandifolia), one of the weeds common in sugarcane plantations.

### CNN Architectures
The idea is to reuse popular Imagenet architectures, fine-tuning them with a custom set of flower images (signal) and miscellaneous plantation images (background). 
