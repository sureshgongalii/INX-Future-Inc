# Project Title: Employee Performance Analysis – INX Future Inc.

> **IABAC-Certified Real-Time Data Science Project**

This IABAC-Certified real-time data science project focuses on **INX Future Inc.**, a reputed data analytics and automation company. Despite being consistently ranked among the top employers, the company faced a **decline in employee performance**, leading to an **8% drop in client satisfaction** and a **surge in service delivery issues**.

The primary goal of this project was to:
- Identify key drivers behind performance decline
- Analyze performance trends across departments
- Build a robust machine learning model to **predict employee performance**
- Provide actionable business recommendations to support **strategic HR decisions**  
- Maintain **employee morale** while improving **client satisfaction and delivery quality**


# Employee Performance Analysis - INX Future Inc.

## Overview

This project addresses a critical business challenge faced by INX Future Inc., a leading data analytics and automation solutions provider. Despite being consistently rated as a top  employer, the company experienced declining employee performance metrics and an 8% drop in client satisfaction levels. This comprehensive data science project analyzes employee performance data to identify root causes of performance issues and provides actionable insights for strategic decision-making.

## Business Problem

- **Challenge**: Declining employee performance indexes affecting service delivery and client satisfaction
- **Impact**: 8% decrease in client satisfaction levels and increased service delivery escalations
- **Objective**: Identify core factors affecting employee performance without negatively impacting overall employee morale
- **Stakeholder**: CEO seeking data-driven insights for strategic workforce management decisions

## Dataset Description

**Source**: INX Future Inc. Employee Performance Dataset (IABAC Certified Data Science Project)
- **Format**: Excel file (.xls)
- **Scope**: Comprehensive employee performance metrics across multiple departments
- **Features**: Employee demographics, performance ratings, departmental information, and various performance indicators
- **Business Context**: Real-world corporate dataset reflecting actual HR challenges in performance management

## Technologies Used

- **Programming Language**: Python 3.x
- **Data Manipulation**: pandas, NumPy
- **Data Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, XGBoost
- **Model Persistence**: pickle, joblib
- **Automated EDA**: SweetViz
- **Development Environment**: Jupyter Notebook
- **Data Storage**: CSV, Feather, Pickle formats for optimized data handling

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Comprehensive statistical analysis using dedicated EDA notebook
- Automated profiling report generation with SweetViz
- Department-wise performance distribution analysis
- Correlation analysis between performance factors

### 2. Data Preprocessing
- Systematic data cleaning and transformation pipeline
- Feature engineering for performance indicators
- Data standardization and encoding of categorical variables
- Multiple data format exports (CSV, Feather, Pickle) for efficient processing

### 3. Machine Learning Implementation
- **Multiple Algorithm Evaluation**: Implemented 8 different classification algorithms
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP) Classifier

### 4. Model Selection & Optimization
- Comparative performance analysis across all algorithms
- Hyperparameter tuning for optimal model performance
- **Best Model Selection**: XGBoost identified as top-performing algorithm
- Model persistence using pickle for deployment readiness

## Model & Evaluation

**Primary Objective**: Develop a predictive model to identify employee performance levels for strategic hiring and management decisions.

**Modeling Approach**:
- Implemented 8 classification algorithms for comprehensive comparison
- Systematic evaluation using cross-validation and multiple performance metrics
- **Selected Model**: XGBoost (saved as `xgboost_best_model.pkl`)

**Model Performance**:
- All trained models saved in `saved_models/` directory for reproducibility
- Best model identified through rigorous evaluation process
- Cross-validation implemented for robust performance assessment

## Key Findings & Business Insights

### 1. Department-wise Performance Analysis
- Identified departments with significant performance variations
- Quantified performance gaps across organizational units
- Provided statistical evidence for targeted interventions

### 2. Top 3 Critical Performance Factors
- Systematically identified the most influential factors affecting employee performance
- Ranked factors by statistical significance and business impact
- Provided actionable insights for HR policy improvements

### 3. Predictive Model for Hiring Decisions
- Developed XGBoost model capable of predicting employee performance based on key input factors
- Enabled data-driven hiring decisions to improve future workforce quality
- Model ready for deployment with saved pickle files

### 4. Strategic Recommendations
- Evidence-based recommendations for improving overall employee performance
- Targeted interventions for underperforming departments
- Data-driven approach to minimize impact on employee morale while addressing performance issues

## Project Structure

```
INX-Future-Inc/
│
├── data/
│   ├── INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls
│   ├── INX-ANALYSIS.xls
│   ├── INX-preprocessed_data.csv
│   ├── INX-preprocessed_data.feather
│   └── INX-preprocessed_data.pkl
│
├── notebooks/
│   ├── Employee Performance Analysis INX Future Inc-MAIN.ipynb
│   ├── INX-EXPLORATORY DATA ANALYSIS.ipynb
│   ├── INX-DATA_PREPROCESSING.ipynb
│   ├── INX-PROJECT SUMMARY.ipynb
│   └── SOURCE-MAIN.ipynb
│
├── saved_models/
│   ├── best_model/
│   │   └── xgboost_best_model.pkl
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── mlp_classifier.pkl
│   ├── random_forest.pkl
│   ├── svc.pkl
│   └── xgboost.pkl
│
├── reports/
│   └── sweetviz_report.html
│
├── documentation/
│   └── IINX-PROJECT SUMMARY.doc
│
└── README.md
```

## How This Project Adds Value

**Technical Skills Demonstrated**:
- End-to-end machine learning pipeline development
- Multiple algorithm implementation and comparison
- Advanced data preprocessing and feature engineering
- Model selection and hyperparameter optimization
- Professional code organization and model persistence

**Business Impact**:
- Addressed real-world HR challenges with quantifiable business impact
- Provided predictive capabilities for strategic hiring decisions
- Developed actionable recommendations for performance improvement
- Created reproducible analysis framework for ongoing monitoring

**Professional Relevance**:
- Demonstrates ability to work with stakeholder requirements (CEO-level expectations)
- Shows understanding of business constraints and ethical considerations
- Exhibits capability to translate technical findings into business recommendations
- Proves competency in industry-standard tools and methodologies

## How to Run This Project

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost sweetviz jupyter
```

### Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/sureshgongalii/INX-Future-Inc.git
cd INX-Future-Inc
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Execute notebooks in sequence:
   - Start with `INX-EXPLORATORY DATA ANALYSIS.ipynb`
   - Continue with `INX-DATA_PREPROCESSING.ipynb`
   - Run the main analysis: `Employee Performance Analysis INX Future Inc-MAIN.ipynb`
   - Review summary: `INX-PROJECT SUMMARY.ipynb`

### Loading Pre-trained Models
```python
import pickle

# Load the best performing model
with open('saved_models/best_model/xgboost_best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Load preprocessed data
import pandas as pd
data = pd.read_csv('INX-preprocessed_data.csv')
```

### Viewing Analysis Reports
- Open `sweetviz_report.html` in your browser for automated EDA insights
- Review `IINX-PROJECT SUMMARY.doc` for detailed project documentation

## Future Enhancements

- Implementation of real-time performance monitoring dashboard
- Integration with HR management systems for automated predictions
- Advanced ensemble methods combining multiple top-performing models
- Deployment of best model as a web service for operational use
- A/B testing framework for validating model recommendations

---

**Project Certification**: IABAC Certified Data Science Project
**Business Domain**: Human Resources Analytics & Performance Management
**Technical Domain**: Machine Learning, Statistical Analysis, Business Intelligence
**Technical Domain: Machine Learning, Statistical Analysis, Business Intelligence
