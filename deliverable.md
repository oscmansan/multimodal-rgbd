# Deliverable

## 1. Comparison of RGB, HHA and RGB-D baselines

The following table shows train, validation and test average accuracies (and standard deviations) over 5 runs for each case (RGB only, HHA only and RGB-D).

| modality | set   | run 1 | run 2 | run 3 | run 4 | run 5 | avg   | std    |
|----------|-------|-------|-------|-------|-------|-------|-------|--------|
| rgb      | train | 92.21 | 84.70 | 90.87 | 86.85 | 86.44 | 88.21 | 3.1772 |
| rgb      | val   | 77.42 | 76.96 | 78.69 | 77.88 | 78.00 | 77.79 | 0.6496 |
| rgb      | test  | 72.09 | 68.81 | 73.22 | 70.57 | 72.46 | 71.43 | 1.7539 |
| hha      | train | 75.70 | 76.64 | 73.15 | 73.29 | 77.18 | 75.19 | 1.8771 |
| hha      | val   | 65.32 | 66.24 | 65.67 | 67.40 | 66.59 | 66.24 | 0.8123 |
| hha      | test  | 59.48 | 58.16 | 61.18 | 61.18 | 60.93 | 60.19 | 1.3354 |
| rgbd     | train | 93.96 | 93.29 | 94.63 | 92.89 | 93.96 | 93.75 | 0.6734 |
| rgbd     | val   | 80.41 | 80.53 | 81.57 | 82.03 | 81.57 | 81.22 | 0.7130 |
| rgbd     | test  | 74.04 | 75.61 | 74.17 | 74.61 | 74.86 | 74.66 | 0.6264 |

## 2. Description of the improvements of the RGB-D network, experimental results and discussion

### Improvements

* Replaced AlexNet by ResNet-18 as base architecture for extracting embeddings.
* Early fusion: combined output feature maps of the convolutional layerss instead of the output decisions of the fully-connected layers. This is done by concatenating the output feature maps of the last convolutional layer of each base network. The output volume of each stream has shape 512x8x8, giving a total volume of 1024x8x8 after concatenation. Then, a 1x1 convolution is applied to reduce the number of channels to 512 and further combine RGB and depth features. Next, global average pooling is applied in order to obtain a 1D vector of length 512. Finally, that vector is fed to a couple of fully-connected layers which make up the classifier.
* Fine-tuned the layers of the fusion network, while keeping the convolutional layers of the two base networks freezed.

### Experimental results and discussion

| modality      | set   | run 1 | run 2 | run 3 | run 4 | run 5 | avg   | std    |
|---------------|-------|-------|-------|-------|-------|-------|-------|--------|
| rgbd improved | train | 92.48 | 94.63 | 92.08 | 94.63 | 94.09 | 93.58 | 1.2171 |
| rgbd improved | val   | 81.80 | 82.83 | 82.60 | 81.34 | 81.80 | 82.07 | 0.6199 |
| rgbd improved | test  | 79.02 | 77.19 | 79.02 | 77.38 | 79.08 | 78.34 | 0.9639 |

As it can be observed in the above tables, a substantial improvement was obtained in the test set (almost 4%), while only a fair improvement was obtained in the validation set (almost 1%). No improvement was obtained in the training set. The gap between training accuracy and validation accuracy is shortened, meaning that the performance of the RGB-D model was increased while reducing its overfitting to the training set.

### Future work

* Pre-train the two individual stream networks with RGB and HHA data respectively, and then train the fusion network while freezing the weights of the fine-tuned base networks.

## 3. Team work

I did this optional assignment by myself.