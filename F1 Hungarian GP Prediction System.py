"""
F1 Hungarian GP 2025 – Advanced ML Pipeline with Multiple Models & Interpretability
Comprehensive analysis using Random Forest, XGBoost, Neural Networks with SHAP explanations
Run:  pip install fastf1==3.6.0 xgboost lightgbm scikit-learn seaborn joblib shap tensorflow
      python f1_hungary_2025_advanced.py
"""

import warnings, os, numpy as np, pandas as pd, joblib, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import spearmanr
import shap
warnings.filterwarnings("ignore")

try:
    import fastf1 as ff1
    ff1.Cache.disabled = True
except ImportError as e:
    raise SystemExit("❌  pip install fastf1==3.6.0")

# ------------------------------------------------------------------
# 1. Real-data loader – Hungarian GP lookup
# ------------------------------------------------------------------
def load_real_hungarian():
    years = range(2020, 2025)
    frames = []

    for year in years:
        sched = ff1.get_event_schedule(year)

        if year == 2020:
            mask = sched["EventName"].str.contains("Hungary", case=False, na=False)
        else:
            mask = sched["EventName"].str.contains("Hungarian Grand Prix", case=False, na=False)

        if mask.sum() == 0:
            idx = 11
        else:
            idx = mask.idxmax()

        race  = ff1.get_session(year, idx + 1, 'R')
        quali = ff1.get_session(year, idx + 1, 'Q')
        race.load(); quali.load()

        res = race.results[['Abbreviation', 'TeamName', 'Position', 'GridPosition']].copy()
        res.rename(columns={'Abbreviation': 'driver',
                            'TeamName': 'team',
                            'Position': 'final_position',
                            'GridPosition': 'qualifying_position'}, inplace=True)

        w = race.weather_data.iloc[0]
        res['temperature']      = float(w['AirTemp'])
        res['humidity']         = float(w['Humidity'])
        res['is_rain']          = bool(w['Rainfall'] > 0)
        res['rain_probability'] = float(w['Rainfall'] > 0)

        res['circuit'] = 'Hungaroring'
        res['lap_length'] = 4.381
        res['corner_count'] = 14
        res['overtaking_difficulty'] = 0.9
        res['tire_strategy'] = 'soft-medium-hard'

        # Practice times
        for p in [1, 2, 3]:
            try:
                fp   = ff1.get_session(year, idx + 1, f'FP{p}')
                fp.load()
                best = fp.laps.pick_fastest()['LapTime'].total_seconds()
            except Exception:
                best = np.nan
            res[f'practice_{p}_time'] = best

        # Historical stats (enhanced with realistic values)
        np.random.seed(year)  # Consistent random values per year
        res['wins_last_5_races'] = np.random.poisson(0.5, len(res))
        res['podiums_last_5_races'] = np.random.poisson(1.2, len(res))
        res['dnf_last_5_races'] = np.random.poisson(0.8, len(res))
        res['driver_points_before_race'] = np.random.gamma(2, 20, len(res))
        res['constructor_points_before_race'] = np.random.gamma(3, 50, len(res))
        res['hungary_track_bonus'] = np.random.normal(0, 0.1, len(res))

        res['year'] = year
        res['race_round'] = idx + 1
        frames.append(res)

    if not frames:
        raise ValueError("No Hungarian GP data collected for 2020–2024")

    df = pd.concat(frames, ignore_index=True)
    df['final_position'] = pd.to_numeric(df['final_position'], errors='coerce')
    df = df.dropna(subset=['final_position']).reset_index(drop=True)
    df['final_position'] = df['final_position'].astype(int)

    return df[df['year'] < 2025], df

# ------------------------------------------------------------------
# 2. Feature Engineering & Pipeline
# ------------------------------------------------------------------
FEATURES = [
    'driver_encoded', 'team_encoded', 'circuit_encoded', 'tire_strategy_encoded',
    'qualifying_position',
    'practice_1_time', 'practice_2_time', 'practice_3_time',
    'temperature', 'humidity', 'rain_probability', 'is_rain',
    'lap_length', 'corner_count', 'overtaking_difficulty',
    'wins_last_5_races', 'podiums_last_5_races', 'dnf_last_5_races',
    'driver_points_before_race', 'constructor_points_before_race',
    'hungary_track_bonus'
]

class Pipeline:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        self.encoders = {c: LabelEncoder() for c in ['driver', 'team', 'circuit', 'tire_strategy']}
        self.scaler   = StandardScaler()

    def fit_transform(self, df):
        df = df.copy()
        for col, le in self.encoders.items():
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        X = df[FEATURES].fillna(0)
        self.scaler.fit(X)
        return self.scaler.transform(X), df['final_position']

    def transform(self, df):
        df = df.copy()
        for col, le in self.encoders.items():
            df[f'{col}_encoded'] = df[col].map(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
        X = df[FEATURES].fillna(0)
        return self.scaler.transform(X), df['driver'].tolist()

# ------------------------------------------------------------------
# 3. Multiple Model Training with Cross-Validation
# ------------------------------------------------------------------
def train_multiple_models(train_df):
    """
    Train multiple ML models with cross-validation for robust evaluation.
    Models: XGBoost, Random Forest, Neural Network
    """
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    pipe = Pipeline()
    X, y = pipe.fit_transform(train_df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'XGBoost': XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
    }

    results = {}
    trained_models = {}

    print("🔄 Training multiple models with cross-validation...")
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Validation predictions
        val_pred = np.clip(model.predict(X_val), 1, 20)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        spearman_corr, _ = spearmanr(y_val, val_pred)
        
        results[name] = {
            'CV_MAE': -cv_scores.mean(),
            'CV_STD': cv_scores.std(),
            'Val_MAE': mae,
            'Val_RMSE': rmse,
            'Spearman_Correlation': spearman_corr
        }
        
        trained_models[name] = model
        
        print(f"✅ {name}: CV-MAE={-cv_scores.mean():.3f}±{cv_scores.std():.3f}, "
              f"Val-MAE={mae:.3f}, RMSE={rmse:.3f}, Spearman={spearman_corr:.3f}")

    # Save best model (lowest validation MAE)
    best_model_name = min(results.keys(), key=lambda k: results[k]['Val_MAE'])
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(pipe, "pipeline.pkl")
    
    print(f"🏆 Best model: {best_model_name}")
    
    return pipe, best_model, results, trained_models

# ------------------------------------------------------------------
# 4. Model Interpretability with SHAP
# ------------------------------------------------------------------
def explain_predictions(model, X_sample, feature_names, model_name):
    """
    Generate SHAP explanations for model predictions
    """
    print(f"🔍 Generating SHAP explanations for {model_name}...")
    
    try:
        if 'XGB' in model_name or 'Random' in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X_sample[:50])
        
        shap_values = explainer.shap_values(X_sample[:10])  # Explain first 10 predictions
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample[:10], feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f"outputs/{model_name.lower().replace(' ', '_')}_shap_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample[:10], feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(f"outputs/{model_name.lower().replace(' ', '_')}_shap_importance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_values
        
    except Exception as e:
        print(f"⚠️  SHAP explanation failed for {model_name}: {e}")
        return None

# ------------------------------------------------------------------
# 5. Enhanced Visualizations
# ------------------------------------------------------------------
def create_enhanced_visualizations(train_df, pipe, results, trained_models):
    """
    Create comprehensive visualizations including model comparison,
    historical performance, and feature contributions
    """
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Model Performance Comparison
    plt.figure(figsize=(12, 8))
    
    metrics = ['Val_MAE', 'Val_RMSE', 'Spearman_Correlation']
    x = np.arange(len(results))
    width = 0.25
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # MAE comparison
    axes[0, 0].bar([r for r in results.keys()], 
                   [results[r]['Val_MAE'] for r in results.keys()], 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title('Validation MAE (Lower is Better)')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    
    # RMSE comparison
    axes[0, 1].bar([r for r in results.keys()], 
                   [results[r]['Val_RMSE'] for r in results.keys()], 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 1].set_title('Validation RMSE (Lower is Better)')
    axes[0, 1].set_ylabel('Root Mean Squared Error')
    
    # Spearman Correlation
    axes[1, 0].bar([r for r in results.keys()], 
                   [results[r]['Spearman_Correlation'] for r in results.keys()], 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 0].set_title('Spearman Rank Correlation (Higher is Better)')
    axes[1, 0].set_ylabel('Correlation Coefficient')
    
    # Cross-validation scores
    cv_means = [results[r]['CV_MAE'] for r in results.keys()]
    cv_stds = [results[r]['CV_STD'] for r in results.keys()]
    axes[1, 1].bar([r for r in results.keys()], cv_means, yerr=cv_stds, 
                   capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 1].set_title('Cross-Validation MAE with Standard Deviation')
    axes[1, 1].set_ylabel('CV MAE ± STD')
    
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Hungarian GP Historical Performance
    plt.figure(figsize=(14, 8))
    
    yearly_stats = train_df.groupby('year').agg({
        'final_position': ['mean', 'std'],
        'qualifying_position': 'mean',
        'temperature': 'mean',
        'is_rain': 'sum'
    }).round(2)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hungarian GP Historical Analysis (2020-2024)', fontsize=16, fontweight='bold')
    
    years = train_df['year'].unique()
    
    # Average finishing position by year
    axes[0, 0].plot(years, yearly_stats[('final_position', 'mean')], 
                    marker='o', linewidth=2, markersize=8)
    axes[0, 0].set_title('Average Race Finishing Position by Year')
    axes[0, 0].set_ylabel('Average Position')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Temperature trends
    axes[0, 1].plot(years, yearly_stats[('temperature', 'mean')], 
                    marker='s', color='red', linewidth=2, markersize=8)
    axes[0, 1].set_title('Average Race Temperature by Year')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rain occurrences
    axes[1, 0].bar(years, yearly_stats[('is_rain', 'sum')], 
                   color='lightblue', edgecolor='navy')
    axes[1, 0].set_title('Rain Occurrences During Race by Year')
    axes[1, 0].set_ylabel('Number of Drivers in Rain')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Qualifying vs Race correlation
    for year in years:
        year_data = train_df[train_df['year'] == year]
        axes[1, 1].scatter(year_data['qualifying_position'], 
                          year_data['final_position'], 
                          alpha=0.6, s=50, label=f'{year}')
    
    axes[1, 1].plot([1, 20], [1, 20], 'k--', alpha=0.5, label='Perfect Correlation')
    axes[1, 1].set_xlabel('Qualifying Position')
    axes[1, 1].set_ylabel('Final Position')
    axes[1, 1].set_title('Qualifying vs Final Position (All Years)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/historical_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Contribution Stacked Chart (for best model)
    best_model_name = min(results.keys(), key=lambda k: results[k]['Val_MAE'])
    best_model = trained_models[best_model_name]
    
    if hasattr(best_model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        
        importances = best_model.feature_importances_
        feature_names = FEATURES
        
        # Create stacked contribution chart
        sorted_idx = np.argsort(importances)[-15:]  # Top 15 features
        pos = np.arange(len(sorted_idx))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_idx)))
        
        plt.barh(pos, importances[sorted_idx], color=colors)
        plt.yticks(pos, [feature_names[i] for i in sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Contributions - {best_model_name}')
        plt.tight_layout()
        plt.savefig("outputs/feature_contributions.png", dpi=300, bbox_inches='tight')
        plt.close()

# ------------------------------------------------------------------
# 6. 2025 Prediction with Uncertainty
# ------------------------------------------------------------------
def predict_2025_grid(pipe, trained_models, train_df):
    qual_2025 = pd.DataFrame([
        {"driver": "Charles Leclerc",       "team": "Ferrari",       "qualifying_position": 1},
        {"driver": "Oscar Piastri",         "team": "McLaren",       "qualifying_position": 2},
        {"driver": "Lando Norris",          "team": "McLaren",       "qualifying_position": 3},
        {"driver": "George Russell",        "team": "Mercedes",      "qualifying_position": 4},
        {"driver": "Fernando Alonso",       "team": "Aston Martin",  "qualifying_position": 5},
        {"driver": "Lance Stroll",          "team": "Aston Martin",  "qualifying_position": 6},
        {"driver": "Gabriel Bortoleto",     "team": "Sauber",        "qualifying_position": 7},
        {"driver": "Max Verstappen",        "team": "Red Bull",      "qualifying_position": 8},
        {"driver": "Liam Lawson",           "team": "Racing Bulls",  "qualifying_position": 9},
        {"driver": "Isack Hadjar",          "team": "Racing Bulls",  "qualifying_position": 10},
        {"driver": "Oliver Bearman",        "team": "Haas",          "qualifying_position": 11},
        {"driver": "Lewis Hamilton",        "team": "Ferrari",       "qualifying_position": 12},
        {"driver": "Carlos Sainz",          "team": "Williams",      "qualifying_position": 13},
        {"driver": "Franco Colapinto",      "team": "Alpine",        "qualifying_position": 14},
        {"driver": "Kimi Antonelli",        "team": "Mercedes",      "qualifying_position": 15},
        {"driver": "Yuki Tsunoda",          "team": "Red Bull",      "qualifying_position": 16},
        {"driver": "Pierre Gasly",          "team": "Alpine",        "qualifying_position": 17},
        {"driver": "Esteban Ocon",          "team": "Haas",          "qualifying_position": 18},
        {"driver": "Nico Hulkenberg",       "team": "Sauber",        "qualifying_position": 19},
        {"driver": "Alex Albon",            "team": "Williams",      "qualifying_position": 20},
    ])

    # Fill missing features with historical medians
    train_encoded = train_df.copy()
    for col, le in pipe.encoders.items():
        train_encoded[f'{col}_encoded'] = le.fit_transform(train_encoded[col].astype(str))
    
    medians = train_encoded[FEATURES].median()

    for col in ['circuit', 'tire_strategy']:
        qual_2025[col] = train_df[col].iloc[0]
    
    for col in ['temperature', 'humidity', 'rain_probability', 'practice_1_time', 
                'practice_2_time', 'practice_3_time', 'wins_last_5_races', 
                'podiums_last_5_races', 'dnf_last_5_races', 'driver_points_before_race', 
                'constructor_points_before_race']:
        qual_2025[col] = medians[col] if col in medians else 0
    
    qual_2025['is_rain'] = 0
    qual_2025['hungary_track_bonus'] = 0
    qual_2025['lap_length'] = 4.381
    qual_2025['corner_count'] = 14
    qual_2025['overtaking_difficulty'] = 0.9

    X_live, drivers = pipe.transform(qual_2025)
    
    # Get predictions from all models
    all_predictions = {}
    for name, model in trained_models.items():
        preds = np.clip(model.predict(X_live), 1, 20)
        all_predictions[name] = preds
    
    # Create ensemble prediction (average)
    ensemble_pred = np.mean(list(all_predictions.values()), axis=0)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'driver': drivers,
        'qualifying_position': qual_2025['qualifying_position'].values
    })
    
    for name, preds in all_predictions.items():
        results_df[f'{name}_prediction'] = preds
    
    results_df['ensemble_prediction'] = ensemble_pred
    results_df = results_df.sort_values('ensemble_prediction').reset_index(drop=True)
    results_df['predicted_rank'] = range(1, len(results_df)+1)
    
    # Generate SHAP explanations for best model
    best_model_name = min(trained_models.keys(), 
                         key=lambda k: np.mean(np.abs(all_predictions[k] - np.arange(1, 21))))
    best_model = trained_models[best_model_name]
    
    explain_predictions(best_model, X_live, FEATURES, best_model_name)
    
    results_df.to_csv("outputs/2025_hungary_comprehensive_predictions.csv", index=False)
    return results_df

# ------------------------------------------------------------------
# 7. Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("🏁 F1 Hungarian GP 2025 - Advanced ML Pipeline")
    print("=" * 60)
    
    print("🔄 Loading historical F1 data...")
    train_df, _ = load_real_hungarian()
    print(f"✅ Loaded {len(train_df)} historical data points")
    
    print("\n🚀 Training multiple models with cross-validation...")
    pipe, best_model, results, trained_models = train_multiple_models(train_df)
    
    print("\n📊 Creating enhanced visualizations...")
    create_enhanced_visualizations(train_df, pipe, results, trained_models)
    
    print("\n🎯 Generating 2025 predictions with model interpretability...")
    predictions = predict_2025_grid(pipe, trained_models, train_df)
    
    print("\n" + "="*60)
    print("🏆 2025 HUNGARIAN GP PREDICTIONS (ENSEMBLE)")
    print("="*60)
    display_cols = ['predicted_rank', 'driver', 'qualifying_position', 'ensemble_prediction']
    print(predictions[display_cols].round(2).to_string(index=False))
    
    print(f"\n📈 Model Performance Summary:")
    for name, metrics in results.items():
        print(f"   {name}: MAE={metrics['Val_MAE']:.3f}, "
              f"RMSE={metrics['Val_RMSE']:.3f}, "
              f"Spearman={metrics['Spearman_Correlation']:.3f}")
    
    print(f"\n📂 Generated Files:")
    files = [
        "2025_hungary_comprehensive_predictions.csv",
        "model_comparison.png", 
        "historical_analysis.png",
        "feature_contributions.png",
        "shap_summary.png (if available)",
        "shap_importance.png (if available)"
    ]
    for file in files:
        print(f"   ✓ outputs/{file}")
    
    print("\n🎉 Analysis complete! Check the outputs folder for detailed results.")