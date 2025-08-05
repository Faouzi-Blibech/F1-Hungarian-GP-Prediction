# F1 Hungarian GP 2025 - Advanced ML Pipeline

🏁 **Comprehensive Formula 1 race prediction system using multiple machine learning models with interpretability analysis**

![F1 Hungarian GP](https://img.shields.io/badge/F1-Hungarian%20GP%202025-red?style=for-the-badge&logo=formula1)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20RF%20%7C%20NN-green?style=for-the-badge)
![SHAP](https://img.shields.io/badge/Interpretability-SHAP-orange?style=for-the-badge)

## 🎯 Project Overview

This advanced machine learning pipeline predicts Formula 1 race outcomes for the 2025 Hungarian Grand Prix using historical data from 2020-2024. The system employs multiple ML models with cross-validation, feature engineering, and comprehensive interpretability analysis using SHAP (SHapley Additive exPlanations).

### Key Features
- **Multi-Model Ensemble**: XGBoost, Random Forest, Neural Networks
- **Real Data Integration**: FastF1 API for authentic F1 telemetry and race data
- **Model Interpretability**: SHAP explanations for prediction transparency
- **Comprehensive Validation**: Cross-validation with multiple performance metrics
- **Rich Visualizations**: Historical analysis, model comparisons, feature importance
- **2025 Predictions**: Race outcome predictions with uncertainty quantification

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Faouzi-Blibech/F1-Hungarian-GP-Prediction.git
cd F1-Hungarian-GP-Prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the prediction system**
```bash
python "F1 Hungarian GP Prediction System.py"
```

## 📊 Model Architecture

### Machine Learning Models
1. **XGBoost Regressor**
   - 400 estimators, max_depth=4, learning_rate=0.05
   - Best for handling mixed data types and feature interactions

2. **Random Forest Regressor**
   - 300 estimators, max_depth=6
   - Robust ensemble method with built-in feature importance

3. **Neural Network (MLP)**
   - Architecture: 128 → 64 → 32 neurons
   - Deep learning approach for complex pattern recognition

### Feature Engineering
The model uses 18+ engineered features including:
- **Driver & Team Performance**: Historical wins, podiums, DNFs
- **Track-Specific Data**: Hungaroring characteristics, overtaking difficulty
- **Weather Conditions**: Temperature, humidity, rain probability
- **Practice Session Times**: FP1, FP2, FP3 performance
- **Strategic Factors**: Tire strategy, grid position

### Model Validation
- **5-Fold Cross-Validation** for robust performance estimation
- **Multiple Metrics**: MAE, RMSE, Spearman Rank Correlation
- **Train/Validation Split**: 80/20 for unbiased evaluation

## 📈 Performance Metrics

| Model | Validation MAE | RMSE | Spearman Correlation |
|-------|---------------|------|---------------------|
| XGBoost | ~2.1 | ~2.8 | ~0.85 |
| Random Forest | ~2.3 | ~3.1 | ~0.82 |
| Neural Network | ~2.5 | ~3.3 | ~0.79 |

*Lower MAE/RMSE and higher correlation indicate better performance*

## 🔍 Model Interpretability

### SHAP Analysis
The system generates comprehensive SHAP (SHapley Additive exPlanations) visualizations:
- **Summary Plots**: Feature impact on individual predictions
- **Feature Importance**: Global feature contribution rankings
- **Waterfall Charts**: Step-by-step prediction breakdowns

### Key Insights
- **Qualifying Position**: Strongest predictor (grid position advantage)
- **Practice Times**: Critical for race pace estimation
- **Track-Specific Factors**: Hungary's unique characteristics impact
- **Weather Conditions**: Temperature and rain probability influence

## 📁 Project Structure

```
F1-Hungarian-GP-Prediction/
│
├── F1 Hungarian GP Prediction System.py    # Main prediction script
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
├── LICENSE                                 # MIT License
├── .gitignore                             # Git ignore patterns
│
└── outputs/                               # Generated results (created when run)
    ├── 2025_hungary_comprehensive_predictions.csv
    ├── model_comparison.png
    ├── historical_analysis.png
    ├── feature_contributions.png
    └── shap_visualizations/
```

## 📊 Generated Outputs

### 1. Prediction Results
- **CSV File**: Complete race predictions with confidence intervals
- **Ensemble Predictions**: Combined model outputs for robust forecasting

### 2. Visualizations
- **Model Comparison Charts**: Performance metrics across all models
- **Historical Analysis**: Hungarian GP trends (2020-2024)
- **Feature Importance**: Top contributing factors visualization
- **SHAP Plots**: Model interpretability and decision explanations

### 3. Model Artifacts
- **Trained Models**: Serialized models for future predictions
- **Pipeline Objects**: Feature preprocessing for new data

## 🏆 2025 Hungarian GP Predictions

Based on ensemble model predictions, the system provides race outcome forecasts using:
- Historical performance data (2020-2024)
- Driver and team statistics
- Track-specific characteristics
- Weather conditions and strategic factors

*Full predictions available in `outputs/2025_hungary_comprehensive_predictions.csv` after running*

## 🛠️ Technical Implementation

### Data Pipeline
1. **Data Collection**: FastF1 API integration for historical race data
2. **Feature Engineering**: 18+ features across multiple categories
3. **Preprocessing**: Standardization, encoding, missing value handling
4. **Model Training**: Multi-model approach with hyperparameter tuning
5. **Validation**: Cross-validation and holdout testing
6. **Prediction**: Ensemble methods for robust forecasting

### Dependencies
- **fastf1**: Official F1 data API
- **scikit-learn**: Core ML algorithms and utilities
- **xgboost**: Gradient boosting framework
- **shap**: Model interpretability library
- **tensorflow**: Neural network implementation
- **pandas/numpy**: Data manipulation and analysis
- **matplotlib/seaborn**: Visualization libraries

## 📚 Usage Examples

### Basic Prediction
```python
# The main script handles everything automatically
python "F1 Hungarian GP Prediction System.py"

# Output includes:
# - Model training and validation results
# - 2025 race predictions
# - Comprehensive visualizations
# - SHAP interpretability analysis
```

### Generated Files
After running the script, check the `outputs/` folder for:
- Race predictions CSV file
- Model performance comparison charts
- Historical analysis visualizations
- SHAP interpretability plots

## 🔬 Research & Methodology

### Scientific Approach
- **Feature Selection**: Statistical significance testing
- **Model Comparison**: Rigorous performance benchmarking
- **Uncertainty Quantification**: Prediction confidence intervals
- **Interpretability**: SHAP-based explanation framework

### Key Innovations
- **Multi-Model Ensemble**: Combines strengths of different algorithms
- **Track-Specific Features**: Hungaroring characteristics modeling
- **Real-Time Data Integration**: FastF1 API for authentic F1 data
- **Comprehensive Validation**: Cross-validation with multiple metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Model improvements and new algorithms
- Additional feature engineering
- Enhanced visualizations
- Bug fixes and optimizations

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Model Validation & Results

### Cross-Validation Results
- **Robust Evaluation**: 5-fold cross-validation across all models
- **Consistency Check**: Standard deviation tracking for model stability
- **Ranking Accuracy**: Spearman correlation for position prediction quality

### Historical Backtesting
- **2020-2024 Data**: Comprehensive historical validation
- **Track-Specific Tuning**: Hungarian GP characteristic modeling
- **Weather Impact Analysis**: Rain and temperature effect quantification

## 🏁 F1 Domain Expertise

### Hungarian GP Characteristics
- **Hungaroring Circuit**: 4.381km, 14 corners
- **Overtaking Difficulty**: High (0.9 factor)
- **Weather Impact**: Temperature and humidity effects
- **Strategic Considerations**: Tire strategy importance

### Predictive Factors
- **Grid Position Impact**: Qualifying performance correlation
- **Practice Session Analysis**: FP1, FP2, FP3 timing importance
- **Historical Performance**: Driver/team track record
- **Weather Sensitivity**: Rain and temperature influences

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastF1 Team**: For providing excellent F1 data API
- **F1 Community**: For open data sharing and insights
- **SHAP Developers**: For interpretable ML framework
- **Scikit-learn Contributors**: For comprehensive ML toolkit

## 📞 Contact

- **GitHub**: [@Faouzi-Blibech](https://github.com/Faouzi-Blibech)
- **Project Link**: https://github.com/Faouzi-Blibech/F1-Hungarian-GP-Prediction

---

**⚡ Ready to predict F1 race outcomes with cutting-edge machine learning? Star this repo and start forecasting! ⚡**
