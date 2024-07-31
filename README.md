# Comparing the Performance of Radiation Oncologists versus Deep Learning 

This repository contains code to reproduce the analysis from our recent paper "Comparing the Performance of Radiation Oncologists versus a Deep Learning Dose Predictor to Estimate Dosimetric Impact of Segmentation Variations for Radiotherapy", presented as an oral talk at MIDL 2024.

Read the paper [here](https://openreview.net/pdf/5f8cbcc7c1bba1e30813f02448e4d7c8be57c3b2.pdf).  

![figure-one.png](images/figure-one.png)

Is a deep learning dose prediction model able to ascertain dosimetric impact of tumor target volume contour changes when compared to radiation oncologists? An experimental study is run with 54 contour variations which are individually re-planned to generate three categories of results: “Worse”, “No Change” and “Better”.

## Abstract
Current evaluation methods for quality control of manual/automated tumor and organs-at- risk segmentation for radiotherapy are driven mostly by geometric correctness. It is however known that geometry-driven segmentation quality metrics cannot characterize potentially detrimental dosimetric effects of sub-optimal tumor segmentation.

In this work, we build on prior studies proposing deep learning-based dose prediction models to extend its use for the task of contour quality evaluation of brain tumor treatment planning. Using a test set of 54 contour variants and their corresponding dose plans, we show that our model can be used to dosimetrically assess the quality of contours and can outperform clinical expert radiation oncologists while estimating sub-optimal situations.

We compare results against three such experts and demonstrate improved accuracy in addition to time savings.

## How to run:

1. Clone the repository.
2. Install the requirements using `pip install -r requirements.txt`
3. Run code/generate-figure-3.py and code/generate-figure-4.py to reproduce the analysis. Data required to generate figures 3 and 4 from the paper is included in radonc-vs-dldp-data.zip.

## More details about this work:

see https://amithjkamath.github.io/projects/2024-midl-radonc-vs-dldp/ for more.