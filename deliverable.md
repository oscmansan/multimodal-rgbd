# Deliverable

## 1. Comparison of RGB, HHA and RGB-D baselines

Include a table with the train, validation and test average accuracies (and standard deviations) over 5 runs for each case (RGB only, HHA only and RGB-D).

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

