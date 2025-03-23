
========================================
Machine Learning Experiments - HW3

This repository contains the implementation and experiments for HW3, focusing on unsupervised learning techniques, including clustering, dimensionality reduction, and their interactions with supervised learning. The experiments are conducted using two datasets:

Customer Personality Dataset (Market Campaign Data - Task 1, 2, 3, 4, 5)

Spotify 2023 Dataset (Task 1, 2, 3)

Repository Structure

ML-HW3/
│── clustering/              --> Experiments related to Clustering (Task 1, Task 3)
│── configs/                 --> Configuration files for models and experiments (YAML format)
│── datasets/                --> Preprocessed dataset files (No need to rerun data pipeline)
│── dimension_reduction/     --> Experiments related to Dimensionality Reduction (Task 2)
│── model_checkpoints/       --> Saved model checkpoints from previous runs
│── results/                 --> Output results
│── src/                     --> Experiments for Supervised Learning (Task 4, Task 5)
│── environment.yaml         --> Conda environment dependencies for reproducing experiments
│── README.md                --> (This file) Repository documentation

Experiment Files

Clustering Experiments (Task 1, Task 3)

Located in the clustering/ folder:

    clustering_dr_mkt.ipynb - Clustering experiments on the Market Campaign dataset

    clustering_dr_spotify.ipynb - Clustering experiments on the Spotify dataset

    clustering_exploration.ipynb - General clustering exploration

Dimensionality Reduction Experiments (Task 2)

Located in the dimension_reduction/ folder:

    dr_explorations.ipynb - Dimensionality reduction experiments (PCA, ICA, RP, UMAP) applied to both datasets

Supervised Learning with Dimensionality Reduction & Clustering Features (Task 4, Task 5)

Located in the src/ folder:

    nn_dr_experiments.ipynb - Neural network training on datasets processed with dimensionality reduction (Task 4)

    nn_dr_cluster_experiments.ipynb - Neural network training with clustering-derived features added (Task 5)

    neural_network.py - Core neural network model implementation

    neural_network_trainer.py - Training script for the neural network

    mkt_data_processing.py - Preprocessing script for the Market Campaign dataset

Reproducing the Experiments

1. Environment Setup

To install dependencies, create a new Conda environment using:

conda env create -f environment.yaml
conda activate <your_env_name>

2. Running the Experiments

Clustering Experiments: Open and run the notebooks in the clustering/ folder.

Dimensionality Reduction Experiments: Open and run dr_explorations.ipynb in the dimension_reduction/ folder.

Supervised Learning Experiments: Open and run the notebooks in the src/ folder.

3. Notes

No need to rerun the data pipeline; required datasets are preprocessed and stored in datasets/.

To modify model configurations, edit the YAML files inside configs/.

For any issues or clarifications, refer to the HW3 Report or contact the repository maintainer.

Read Only Project Report: https://www.overleaf.com/read/vnxwywvwmjwm#094323