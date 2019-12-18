# Improving Instance Segmentation for Cell Microscopy

[Mikhail Papkov](https://github.com/papkov), [Allan Kustavus](https://github.com/Akustav), [Roberts Oskars Komarovskis](https://github.com/rokspy)

LOTI.05.037 Digital Image Processing, University of Tartu, fall 2019

Instance segmentation is an essential part of a fluorescent microscopy image processing pipeline. Accurate cell segmentation is necessary for the correctness of any downstream analysis such as cell counting and measuring morphological properties. Although segmentation of e.g. DAPI-stained cell nuclei is a well-known task, many modern methods still suffer from unwanted mergers and splits of separate cell instances. Here, we explore various post-processing methods to improve instance segmentation for the given set of segmentation masks produced by a trained neural network.


## Benchmarking

Methods are evaluated by pixel-wise accuracy, precision, recall, F1-score, object-wise F1-score.

|                     | Accuracy  | Precision  | Recall| PW F1 | OW F1  | Merges | Splits   | t, s/img |
|---------------------|-------|--------|-------|-------|--------|--------|----------|------|
| Baseline            | 0.887 | 0.816  | 0.708 | 0.758 | 0.327  | 17869  | 17159    | -    |
| Watershed           | 0.886 | 0.817  | 0.704 | 0.756 | 0.335  | 9299   | 28833    | 11.7 |
| Concavity detection | 0.885 | 0.819  | 0.693 | 0.751 | 0.3    | 6094   | 33554    | 8.2  |
