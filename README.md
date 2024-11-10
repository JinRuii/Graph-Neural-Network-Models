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

The model relies on two main input files located in the `./data/H8-16/` directory:

1. **H8-16_variables.csv** - Contains feature values for each node.
2. **H8-16_edges.csv** - Defines the undirected edges between nodes.

## Installation and Requirements

1. Clone the repository:
   ```bash
   git clone https://github.com/JinRuii/Graph-Neural-Network-Models.git
   cd Graph-Neural-Network-Models
