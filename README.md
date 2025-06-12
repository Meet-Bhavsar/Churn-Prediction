# Customer Churn Prediction

This repository contains code for predicting customer churn using an Artificial Neural Network (ANN). The project uses data preprocessing techniques, a trained model, and a Streamlit app to predict whether a customer is likely to churn based on various input features.

## Files Overview

### 1. **`preprocessing.ipynb`**
   This Jupyter notebook contains the data preprocessing pipeline:
   - Loading and cleaning data (`Churn_Modelling.csv`)
   - Encoding categorical variables (`Gender` and `Geography`)
   - Splitting the data into training and testing sets
   - Scaling the features using `StandardScaler`
   - Saving preprocessing objects (`LabelEncoder`, `OneHotEncoder`, and `StandardScaler`) using `pickle`

### 2. **`prediction.ipynb`**
   This Jupyter notebook demonstrates how to make predictions with the trained model:
   - Loads the trained ANN model (`model.h5`)
   - Loads the preprocessing objects (`label_encoder_gender.pkl`, `onehot_encoder_geography.pkl`, and `scaler.pkl`)
   - Accepts input data and preprocesses it before passing it to the model for prediction
   - Outputs the likelihood of customer churn

### 3. **`app.py`**
   A Streamlit app that provides a user-friendly interface to make customer churn predictions:
   - Takes user input for customer details, such as credit score, geography, gender, balance, and more
   - Preprocesses the input data using the previously saved preprocessing objects
   - Outputs the prediction whether the customer is likely to churn or not based on the trained model

### 4. **`Churn_Modelling.csv`**
   The dataset used to train the model. It contains customer information and their churn status (Exited).
   - Columns: `RowNumber`, `CustomerId`, `Surname`, `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited`

### 5. **`model.h5`**
   The trained Artificial Neural Network (ANN) model that predicts customer churn. The model is built with Keras and TensorFlow.

### 6. **`preprocessed_data.csv`**
   A preprocessed version of the dataset used for making predictions in `prediction.ipynb`. It contains only the necessary features in numerical form.

---

## Installation

To get started with this project, clone the repository to your local machine and install the necessary dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ann-customer-churn-prediction.git
   cd ann-customer-churn-prediction

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
## How to Use

### 1. **Training the Model (Data Preprocessing)**

Run the `preprocessing.ipynb` notebook to:
   - Clean the data
   - Preprocess categorical variables
   - Train and save the model

This notebook will output the following files:
   - `label_encoder_gender.pkl`
   - `onehot_encoder_geography.pkl`
   - `scaler.pkl`
   - `model.h5`

### 2. **Making Predictions**

Run the `prediction.ipynb` notebook to make predictions for a given customer:
   - Enter customer details in the input dictionary (e.g., credit score, age, balance, etc.)
   - The model will output the probability of the customer churning.

### 3. **Customer Churn Prediction Web App**

To use the Streamlit app (`app.py`), run the following command:

   ```bash
   streamlit run app.py


