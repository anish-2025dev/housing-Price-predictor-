````markdown
# 🏡 Housing Price Predictor

A machine learning project to predict housing prices using a variety of regression models. This notebook performs data cleaning, feature engineering, model training, comparison across multiple regressors, and saves the best performing model for future use.

---

## 📂 Table of Contents

- [📌 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [⚙️ Prerequisites](#️-prerequisites)
- [📓 Notebook Structure](#-notebook-structure)
- [✅ Key Steps](#-key-steps)
- [🧠 Models Used](#-models-used)
- [🚀 Usage](#-usage)
- [📁 Files](#-files)
- [📫 Contributing & Contact](#-contributing--contact)

---

## 📌 Project Overview

The goal is to build a robust housing price prediction system based on various attributes of homes. The pipeline covers:

- Exploratory data analysis
- Handling missing values
- Encoding categorical variables
- Feature selection via model-based importance
- Training multiple regression models
- Evaluating models using MAPE
- Saving the best model and preprocessing pipeline

---

## 📊 Dataset

**Filename**: `HousePrices.csv`  
**Target Variable**: `Property_Sale_Price`  
Includes various features like:

- `Dwell_Type`: Type of dwelling
- `LotArea`: Lot size
- `Neighborhood`: Area classification
- `GarageType`, `PoolQC`, `Fence`, etc.
- Year, condition, and quality attributes

---

## ⚙️ Prerequisites

Ensure the following Python packages are installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
````

---

## 📓 Notebook Structure

The notebook (`housingpricepredictor_cleaned.ipynb`) is organized as follows:

### 1. 📥 Import Libraries

All essential packages are imported for data processing, visualization, and modeling.

### 2. 📊 Load & Inspect Data

Reads `HousePrices.csv`, checks nulls, data types, and descriptive stats.

### 3. 🔧 Handle Missing Values

* Numerical columns → filled with median or 0
* Categorical columns → domain-specific strings like `"No Basement"`, `"No Garage"`
* Visualized with seaborn heatmaps

### 4. 🔢 Feature Encoding

Categorical features are label-encoded using `LabelEncoder`.

### 5. 📈 Feature Importance & Attribute Selection

* A `RandomForestRegressor` is trained
* Top N most relevant features are selected for model input

### 6. 🤖 Model Comparison

Applies and compares multiple regression models:

* Linear Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Gradient Boosting
* Support Vector Regressor

Evaluated using **Mean Absolute Percentage Error (MAPE)**.

### 7. 🏆 Save Best Model & Pipeline

* Saves best model to `best_model.pkl`
* Saves corresponding scaler to `best_scaler.pkl`
* Saves selected feature list and model name

---

## ✅ Key Steps

✔️ **Data Cleaning** – Imputes missing values logically
✔️ **Label Encoding** – Categorical → Numeric
✔️ **Feature Selection** – Based on model-based importance
✔️ **Model Evaluation** – Via MAPE metric
✔️ **Model Persistence** – Pickles saved for reuse

---

## 🧠 Models Used

| Model                    | Notes                             |
| ------------------------ | --------------------------------- |
| Linear Regression        | Baseline for performance          |
| Decision Tree Regressor  | Handles non-linearity             |
| Random Forest Regressor  | Robust, ensemble-based            |
| KNN Regressor            | Simple, non-parametric            |
| Gradient Boosting        | High accuracy for structured data |
| Support Vector Regressor | Works well with scaled features   |

**Metric**:
📏 **MAPE (Mean Absolute Percentage Error)** – Useful for understanding percentage error from actual prices.

---

## 🚀 Usage

1. Clone this repo or download the files.

2. Ensure these are in the same folder:

   * `housingpricepredictor_cleaned.ipynb`
   * `HousePrices.csv`

3. Run the notebook:

```bash
jupyter notebook housingpricepredictor_cleaned.ipynb
```

4. The notebook will:

   * Clean & preprocess the data
   * Train and evaluate multiple models
   * Save:

     * `best_model.pkl`
     * `best_scaler.pkl`
     * `important_features.pkl`

---

## 📁 Files

| File Name                             | Description                          |
| ------------------------------------- | ------------------------------------ |
| `housingpricepredictor_cleaned.ipynb` | Complete ML workflow notebook        |
| `HousePrices.csv`                     | Training dataset                     |
| `best_model.pkl`                      | Saved model with best MAPE score     |
| `best_scaler.pkl`                     | Scaler object for numerical features |
| `important_features.pkl`              | List of selected top attributes      |

---

## 📫 Contributing & Contact

* Want to improve the project or try new models? Fork and PR!
* Found a bug? Create an issue.
* Need help understanding the flow? Reach out via the Issues tab.

---

🧪 Ideal for practicing end-to-end regression ML workflows, from EDA to deployment-ready model saving.

```
