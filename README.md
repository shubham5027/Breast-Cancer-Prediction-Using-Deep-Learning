# Breast-Cancer-Prediction-Using-Neural-Networks
This project employs an Artificial Neural Network (ANN) to predict breast cancer based on  numerical features. With an accuracy of 96 % without overfitting also it consist preporcessing pipelines using which easier traformation of data, deep learning artchtexture uses Hyperparameter Tuning using Keras Tuner which Enhance Performance.

# Breast Cancer Prediction Using Neural Networks

This repository contains code for a breast cancer prediction project using neural networks. The goal is to predict the diagnosis of breast cancer (Malignant or Benign) based on various features extracted from digitized images of breast cancer biopsies.

## Dataset

The dataset used in this project includes the following features:

- diagnosis
- radius_mean
- texture_mean
- perimeter_mean
- area_mean
- smoothness_mean
- compactness_mean
- concavity_mean
- concave points_mean
- ... (and more)

The dataset is available in the file `breast_cancer_data.csv`. Each row in the dataset corresponds to a breast cancer biopsy, and the "diagnosis" column indicates whether the tumor is malignant (M) or benign (B).

## Getting Started

### Prerequisites

Before running the code, ensure you have the following Python libraries installed:
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras_tuner 
from kerastuner.tuners import RandomSearch
from tensorflow.keras import layers

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


#📈 Results

The trained neural network achieves an impressive accuracy of 96 % on the test set, showcasing its effectiveness in breast cancer prediction.

#📂 Project Structure

Breast_Cancer_Prediction.ipynb: Jupyter notebook containing the main project code.
breast_cancer_prediction.py: Python script equivalent to the notebook.
breast_cancer_data.csv: Dataset file.

🙏 Acknowledgments

The breast cancer dataset is sourced from the UCI Machine Learning Repository.
