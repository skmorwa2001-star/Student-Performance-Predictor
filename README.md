# 🎓 Student Performance Predictor

An end-to-end Machine Learning project for predicting a student's
**final grade (G3)** using academic, attendance, study, and
lifestyle-related factors.

## 📌 Overview

This project demonstrates a complete ML workflow:

**Data → EDA → Feature Selection → Train/Test Split → Model Training →
Evaluation → Prediction → Streamlit App**

### Objective

Predict a student's final academic grade (`G3`) on a **0--20 scale**.

### Problem Type

**Supervised Learning --- Regression**

------------------------------------------------------------------------

## ✨ Features

-   Data exploration and validation
-   Missing-value checks
-   Statistical analysis
-   Correlation analysis
-   Feature selection
-   Train-test splitting
-   Linear Regression
-   Decision Tree Regression
-   Random Forest Regression
-   MAE, MSE, RMSE and R² evaluation
-   Feature importance analysis
-   Actual vs Predicted visualization
-   Individual student prediction
-   Model saving with Joblib
-   Interactive Streamlit web application

------------------------------------------------------------------------

## 🗂️ Project Structure

``` text
Student Performance Predictor/
│
├── data.ipynb
├── student-mat.csv
├── student_performance_model.pkl
├── features.pkl
├── app.py
└── README.md
```

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `data.ipynb`                        Complete analysis, preprocessing,
                                      training and evaluation

  `student-mat.csv`                   Student Mathematics Performance
                                      dataset

  `student_performance_model.pkl`     Trained Decision Tree model

  `features.pkl`                      Selected feature information

  `app.py`                            Streamlit application

  `README.md`                         Project documentation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 📊 Dataset

The project uses the **Student Performance** dataset from the UCI
Machine Learning Repository.

Dataset source:

https://archive.ics.uci.edu/dataset/320/student%2Bperformance

The Mathematics subset (`student-mat.csv`) contains student academic,
demographic, study, social and performance information.

## Selected Features

  Feature        Description
  -------------- -------------------------------------
  `G1`           First-period grade
  `G2`           Second-period grade
  `studytime`    Weekly study-time category
  `failures`     Number of past class failures
  `absences`     Number of school absences
  `health`       Current health status
  `freetime`     Free time after school
  `traveltime`   Travel time to school
  `goout`        Frequency of going out with friends
  `G3`           Final grade / target

------------------------------------------------------------------------

# 🔄 Machine Learning Workflow

``` text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning & Validation
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
┌─────────────────────────────┐
│ Linear Regression           │
│ Decision Tree Regressor     │
│ Random Forest Regressor     │
└─────────────────────────────┘
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Feature Importance
   ↓
Student Prediction
   ↓
Streamlit Application
```

------------------------------------------------------------------------

# 🎯 Feature Selection

The selected input features are:

``` python
features = [
    'G1',
    'G2',
    'studytime',
    'failures',
    'absences',
    'health',
    'freetime',
    'traveltime',
    'goout'
]

X = df[features]
y = df['G3']
```

------------------------------------------------------------------------

# ✂️ Train-Test Split

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The dataset is divided into:

-   **80% Training**
-   **20% Testing**

------------------------------------------------------------------------

# 🤖 Models

## 1. Linear Regression

Used as the baseline regression model.

## 2. Decision Tree Regressor

Used to capture non-linear relationships between student features and
final grade.

``` python
DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)
```

## 3. Random Forest Regressor

An ensemble model based on multiple decision trees.

``` python
RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
```

------------------------------------------------------------------------

# 📏 Evaluation Metrics

### MAE --- Mean Absolute Error

Average absolute difference between actual and predicted grades.

**Lower is better.**

### MSE --- Mean Squared Error

Penalizes larger errors more strongly.

**Lower is better.**

### RMSE --- Root Mean Squared Error

Error measured on the same scale as the target.

**Lower is better.**

### R² Score

Measures the proportion of target variation explained by the model.

**Higher is better.**

------------------------------------------------------------------------

# 🏆 Model Performance

Results obtained on the test set:

  Model                      MAE ↓       RMSE ↓         R² ↑
  ------------------- ------------ ------------ ------------
  Linear Regression         1.1866       1.8776       0.8348
  **Decision Tree**     **0.9933**   **1.5726**   **0.8841**
  Random Forest             1.0385       1.5784       0.8833

## Best Model

### 🌳 Decision Tree Regressor

-   **R²:** `0.8841`
-   **MAE:** `0.9933`
-   **RMSE:** `1.5726`

On the recorded train/test split, Decision Tree performed slightly
better than Random Forest and Linear Regression.

> Model performance can change with a different train/test split.
> Cross-validation and hyperparameter tuning are recommended for a
> stronger evaluation.

------------------------------------------------------------------------

# 🔍 Feature Importance

The Decision Tree identified the following approximate feature
importance:

  Feature          Importance
  -------------- ------------
  `G2`                 0.8325
  `absences`           0.1419
  `health`             0.0138
  `G1`                 0.0109
  `freetime`           0.0006
  `studytime`          0.0004
  `failures`           0.0000
  `traveltime`         0.0000
  `goout`              0.0000

`G2` is the dominant predictor in this model because it represents an
earlier academic grade that is closely related to the final grade.

------------------------------------------------------------------------

# 📊 Visualizations

The notebook includes:

### Actual vs Predicted Final Grade

A scatter plot comparing:

-   Actual `G3`
-   Predicted `G3`

The diagonal line represents perfect prediction.

### Feature Importance

A bar chart showing which selected features contribute most to the
Decision Tree predictions.

------------------------------------------------------------------------

# 🎓 Prediction Example

Example student input:

``` text
G1            = 15
G2            = 16
Study Time    = 3
Failures      = 0
Absences      = 4
Health        = 5
Free Time     = 3
Travel Time   = 2
Going Out     = 3
```

The model returns an estimated final grade:

``` text
Predicted Final Grade: XX / 20
```

The Streamlit app also displays a performance category based on the
predicted grade.

------------------------------------------------------------------------

# 🌐 Streamlit Application

The project contains an interactive web application.

### Application flow

``` text
Student Details
      ↓
Input Validation
      ↓
DataFrame Creation
      ↓
Trained Decision Tree
      ↓
Predicted Final Grade
      ↓
Performance Feedback
```

The application allows users to enter student details and receive an
immediate prediction.

------------------------------------------------------------------------

# ⚙️ Installation

## 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

## 2. Enter the project directory

``` bash
cd "Student Performance Predictor"
```

## 3. Install dependencies

``` bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

------------------------------------------------------------------------

# ▶️ Run the Notebook

Launch Jupyter:

``` bash
jupyter notebook
```

Open:

``` text
data.ipynb
```

Run the notebook cells in order to reproduce the analysis and model
training.

------------------------------------------------------------------------

# 🚀 Run the Streamlit App

From the project directory:

``` bash
python -m streamlit run app.py
```

The application will open in your default browser.

------------------------------------------------------------------------

# 🛠️ Technologies

  Technology         Usage
  ------------------ ---------------------------------
  Python             Core programming
  Pandas             Data manipulation
  NumPy              Numerical computation
  Matplotlib         Visualization
  Seaborn            Statistical visualization
  Scikit-learn       Machine Learning
  Joblib             Model serialization
  Streamlit          Web application
  Jupyter Notebook   Development and experimentation

------------------------------------------------------------------------

# 📚 ML Concepts Demonstrated

-   Supervised Learning
-   Regression
-   Exploratory Data Analysis
-   Data preprocessing
-   Feature selection
-   Correlation analysis
-   Train-test split
-   Linear Regression
-   Decision Tree Regression
-   Random Forest Regression
-   Model evaluation
-   Feature importance
-   Prediction
-   Model serialization
-   Streamlit deployment

------------------------------------------------------------------------

# ⚠️ Limitations

1.  The current model uses `G1` and `G2`, which are previous academic
    grades.
2.  The reported metrics are based on one train/test split.
3.  Feature importance indicates predictive contribution, not causation.
4.  Dataset-based predictions should not be treated as official academic
    assessments.
5.  Real-world performance may differ from performance on the UCI
    dataset.

------------------------------------------------------------------------

# 🔮 Future Improvements

-   [ ] Cross-validation
-   [ ] Hyperparameter tuning
-   [ ] Compare Gradient Boosting and other algorithms
-   [ ] Build an early-warning model without `G1` and `G2`
-   [ ] Add interactive charts to Streamlit
-   [ ] Add prediction history
-   [ ] Add model confidence/error information
-   [ ] Improve UI/UX
-   [ ] Deploy the application online
-   [ ] Add institutional/student support features

------------------------------------------------------------------------

# ⭐ Future Version: Early Performance Predictor

A stronger version can remove `G1` and `G2` and use only information
available before previous-period grades.

For example:

``` text
Study Time
Absences
Failures
Health
Free Time
Travel Time
Going Out
Family/Social Factors
        ↓
Predicted Final Grade
```

This can evolve the project into an **early academic-support system**
that helps identify students who may need additional support.

------------------------------------------------------------------------

# 🔁 Reproducibility

The project uses:

``` python
random_state = 42
```

To reproduce the reported results:

1.  Use the same dataset.
2.  Use the same selected features.
3.  Use the same 80/20 split.
4.  Use the same model parameters.
5.  Execute the notebook cells in sequence.

------------------------------------------------------------------------

# 👨‍💻 Author

**Sunil Kumar**

B.Tech --- Artificial Intelligence

**Project:** Student Performance Predictor

------------------------------------------------------------------------

# 🙌 Acknowledgements

-   UCI Machine Learning Repository --- Student Performance dataset
-   Scikit-learn --- Machine Learning tools
-   Streamlit --- Web application framework
-   Pandas / NumPy --- Data processing

------------------------------------------------------------------------

## 📜 License

This project is intended for educational and portfolio purposes.

The dataset is provided by the UCI Machine Learning Repository and
remains subject to its applicable terms of use.

------------------------------------------------------------------------

### ⭐ If you find this project useful, consider giving the repository a star!
