
# Project Title: Developing an Explainable Machine Learning Model for CO2 Prediction in Neonatal Care to Enhance Respiratory Support
### Supervisor: Dr. H.K Lam
### Done by: Gabriella Fawaz

Accurate and real-time monitoring of end-tidal carbon dioxide (etCO₂) levels is critical in neonatal intensive care (NICU) to ensure optimal respiratory support for premature and critically ill newborns.
Traditional CO₂ monitoring methods can be invasive, inaccurate, and lack real-time predictive capabilities.
This project applies Machine Learning (ML) and Explainable Artificial Intelligence (XAI) to develop an interpretable and accurate system for forecasting etCO₂ levels in neonates.

## Project Overview

Developed and compared several time series forecasting models (both traditional and deep learning) for predicting future etCO₂ values.
Implemented explainable AI techniques to make model predictions interpretable for clinical use.
Achieved state-of-the-art performance using XGBoost, outperforming baseline linear regression by about five times with a MAPE of 4.8 percent and no observed overfitting.
Designed the workflow to integrate seamlessly into neonatal intensive care, with a focus on clinical applicability and model transparency.

## Key Features

Data Preprocessing
- Missing value imputation using patient-specific medians
- Encoding categorical features (diagnosis and ventilation mode) with one-hot encoding
- Scaling and normalization for deep learning optimization
- Time feature engineering (cyclical time transformation)

Feature Engineering
- Created medically meaningful features such as Ventilation_Efficiency, Respiration_Health, Oxygen_to_etCO2, and Risk_Factor
- Selected top-performing features using SelectKBest and Random Forest importance

Model Development
- Traditional models: Linear Regression, Autoregression
- Tree-based: XGBoost (best performing)
- Deep learning: DNN, CNN, LSTM, AR-LSTM

Explainable AI
- Applied interpretability techniques to understand feature impact
- Built transparent models suitable for clinical decision support

Evaluation
- Used Mean Absolute Percentage Error (MAPE) as the primary metric
- Performed k-fold cross-validation for robust performance testing
- Analyzed effects of feature engineering, data augmentation, different time-window sizes, and feature selection strategies

## Results

| Model            | Validation MAPE | Test MAPE |
|------------------|-----------------|-----------|
| Linear Regression | 26.2%          | 17.6%     |
| Deep Neural Net   | ~17.6%         | 12.1%     |
| CNN               | ~17.6%         | 12.1%     |
| LSTM              | ~23%           | ~15%      |
| AR-LSTM           | ~22%           | ~14%      |
| XGBoost           | 4.8%           | 5.3%      |

XGBoost achieved the highest accuracy and stability, making it the most clinically viable model for real-time etCO₂ prediction.

## Clinical Impact

Improved neonatal respiratory care by enabling non-invasive, real-time CO₂ prediction.
Reduced reliance on invasive and less accurate monitoring methods.
Provided explainable outputs, giving clinicians confidence in model-driven decisions.
Potential to be integrated into closed-loop ventilation systems to assist in personalized, autonomous neonatal care.

## Tech Stack

Programming: Python
Libraries: TensorFlow/Keras, Scikit-learn, XGBoost, NumPy, Pandas, Matplotlib
Machine Learning: Time Series Forecasting, Gradient Boosting, Neural Networks
Explainability: Feature importance, interpretable ML techniques
Validation: k-Fold Cross Validation, MAPE

## How to Use

1. Clone the repository
```
git clone https://github.com/your-username/neonatal-co2-prediction.git
cd neonatal-co2-prediction
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Run preprocessing
```
python preprocess.py
```

4. Train the model
```
python train.py --model xgboost --window 24
```

5. Evaluate the model
```
python evaluate.py --model xgboost
```

6. Visualize feature importance
```
python explain.py
```

## Future Work

Integration into real-time NICU monitoring systems
Validation with larger, multi-center datasets
Exploration of hybrid models (e.g., XGBoost + LSTM)
Enhanced explainability for clinical adoption

## Ethical Considerations

Rigorous validation to avoid bias and ensure fairness.
Protection of patient privacy and sensitive health data.
Transparency in model decision-making for safe clinical deployment.

## Citation

If you use this work, please cite:

Author. Neonatal CO₂ Prediction using Machine Learning and Explainable AI.
Department of Women & Children’s Health, King’s College London, 2025.
