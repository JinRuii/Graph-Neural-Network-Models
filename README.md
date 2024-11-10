# Graph Neural Network Models 🌐🏙️

![Python](https://img.shields.io/badge/python-v3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-v1.10.0-%23E34F26.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This repository provides Graph Neural Network (GNN) models, specifically focusing on **GraphSAGE** for predicting and analyzing urban factors based on neighborhood data. The models are designed to handle large-scale, undirected graphs with node features, making them suitable for complex urban data analysis, such as understanding the influence of environmental and social features on city planning.

## Applications in Urban Planning 🏢🌳

The provided GraphSAGE model can be applied to various urban planning tasks, such as:

- **Predicting Socioeconomic and Environmental Factors**: Using neighborhood features like housing prices, population density, and green spaces, the model predicts levels of socioeconomic well-being or environmental quality.
- **Analyzing Connectivity and Accessibility**: The model utilizes urban infrastructure data (like distance to public transport and access to amenities) to help planners understand the accessibility levels within different neighborhoods.
- **Evaluating Urban Health Indicators**: By training on air quality and other health-related data, the model aids in assessing and forecasting public health conditions based on spatial characteristics.

## Dataset Structure

To run this project, you need two input files containing node features and edge definitions. The default files are:

- `H8-16_variables.csv`: Contains feature values for each node.
- `H8-16_edges.csv`: Defines the undirected edges between nodes.

### Using Sample Data

For quick testing, we have provided two sample files in this repository:

- `Sample_variables.csv`: Contains sample feature values for each node.
- `Sample_edges.csv`: Defines sample undirected edges between nodes.

These sample files allow you to run the model without needing the full dataset. Simply place `Sample_variables.csv` and `Sample_edges.csv` in the `./data/` directory, or adjust the file paths in the code to point to the sample files.

## Installation and Requirements

1. Clone the repository:
   ```bash
   git clone https://github.com/JinRuii/Graph-Neural-Network-Models.git
   cd Graph-Neural-Network-Models

## Key Libraries and Version Requirements

- **Python**: 3.8+
- **PyTorch (CPU version)**: 1.10.0
- **torch_geometric**: 2.0.4
- **torch_scatter**: 2.0.9
- **torch_sparse**: 0.6.12
- **torch_cluster**: 1.5.9
- **torch_spline_conv**: 1.2.1
- **scikit-learn**: 0.24.2
- **pandas**: 1.3.3
- **numpy**: 1.21.2

> **Note**: This project uses the **CPU version** of PyTorch for compatibility with systems without GPU support. If you have a compatible GPU and want to use CUDA, please adjust the installation command to install the appropriate version of PyTorch.


## Installation Instructions

To install the required packages with specified versions, follow these steps:

1. **Create and Activate a Virtual Environment** (recommended):
   ```bash
   python3 -m venv gnn_env
   source gnn_env/bin/activate  # For Linux/macOS
   gnn_env\Scripts\activate  # For Windows

   pip install --upgrade pip setuptools

   pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
   pip install torch_geometric==2.0.4
   pip install torch_scatter==2.0.9 torch_sparse==0.6.12 torch_cluster==1.5.9 torch_spline_conv==1.2.1
   pip install scikit-learn==0.24.2 pandas==1.3.3 numpy==1.21.2


