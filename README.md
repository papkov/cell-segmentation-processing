# Improving Instance Segmentation for Cell Microscopy

[Mikhail Papkov](https://github.com/papkov), [Allan Kustavus](https://github.com/Akustav), [Roberts Oskars Komarovskis](https://github.com/rokspy)

LOTI.05.037 Digital Image Processing, University of Tartu, fall 2019

Instance segmentation is an essential part of a fluorescent microscopy image processing pipeline. Accurate cell segmentation is necessary for the correctness of any downstream analysis such as cell counting and measuring morphological properties. Although segmentation of e.g. DAPI-stained cell nuclei is a well-known task, many modern methods still suffer from unwanted mergers and splits of separate cell instances. Here, we explore various post-processing methods to improve instance segmentation for the given set of segmentation masks produced by a trained neural network.

## Data

Fluorescent and brightfield microscopy images of HepG2 cell culture segmented with [UNet++](https://arxiv.org/abs/1807.10165)

## Objective

Improve instance segmentation for brightfield images with respect to the fluorescent segmentation (fluorescent images are much easirer to segment, pixel-wise errors are lower up to 5x as was [previously shown](https://www.biorxiv.org/content/10.1101/764894v1.abstract)).
Main error to fix — merge of touching cells into a single object.

## Methods

### Concavity-based contour splitting
Source: [Splitting touching cells based on concave points and ellipse fitting](https://dl.acm.org/citation.cfm?id=1563085)

* Detect contours based on thresholded segmentation probability map
* Detect concavity points
* Fit ellipse with concavity points and contour anchor points
* Separate touching cells 

### Watershed
Source: [Automated basin delineation from digital elevation models using mathematical morphology](https://www.sciencedirect.com/science/article/pii/016516849090127K)

* Threshold segmentation probability map at 0.9 to get seeds
* Assure regions: every object should be represented by a seed
* Run [scikit](https://scikit-image.org/docs/0.7.0/api/skimage.morphology.watershed.html) watershed using inversed probability map
* Separate touching cells by detected contour

### Morphology

Run morphological operations (erosion and dilation).


## Benchmarking

Methods are evaluated by pixel-wise accuracy, precision, recall, F1-score, object-wise F1-score.

|                     | Accuracy  | Precision  | Recall| PW F1 | OW F1  | Merges | Splits   | t, s/img |
|---------------------|-------|--------|-------|-------|--------|--------|----------|------|
| Baseline            | 0.887 | 0.816  | 0.708 | 0.758 | 0.327  | 17869  | 17159    | -    |
| Watershed           | 0.886 | 0.817  | 0.704 | 0.756 | 0.335  | 9299   | 28833    | 11.7 |
| Concavity detection | 0.885 | 0.819  | 0.693 | 0.751 | 0.3    | 6094   | 33554    | 8.2  |
