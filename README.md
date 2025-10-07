
# Developing an Explainable Machine Learning Model for CO2 Prediction in Neonatal Care to Enhance Respiratory Support

#### Supervisor: Dr. H.K Lam; 
#### Reaseacher: Gabriella Fawaz

Neonatal intensive care requires accurate and real-time monitoring of end-tidal carbon dioxide (etCO2) levels to provide optimal respiratory support for vulnerable newborns. Traditional methods for monitoring CO2 are limited by accuracy and invasiveness, emphasising the urgent need for precise, non-invasive, and interpretable solutions. In this study, machine learning (ML) and explainable artificial intelligence (XAI) were explored to improve the prediction of etCO2 levels in neonates. This approach combines the predictive power of ML with the transparency and interpretability of XAI. By carefully analysing data and engineering features and evaluating models, the XGBoost model was found to be the best-performing model with a MAPE of 4.8% and no observable overfitting. The XGBoost model outperformed traditional ML methods like linear regression by five times. These findings suggest that with further research, this model could potentially be implemented into the clinics to enhance neonatal care.

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
