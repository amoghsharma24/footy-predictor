import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
from datetime import datetime


def load_training_data():
    """Loading enhanced training data"""
    print("Loading enhanced training data...")
    df = pd.read_csv('results/training_data_enhanced.csv')
    
    exclude_cols = ['Team', 'Year', 'Next_Year_Position']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['Next_Year_Position']
    
    print(f"Samples: {len(X)}, Features: {len(feature_cols)}")
    
    return X, y, feature_cols


def evaluate_model(model, X, y, model_name):
    """Evaluating model using cross-validation and test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print('='*60)
    
    # Cross-validation (5-fold)
    print("Running 5-fold cross-validation...")
    cv_mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))
    
    print(f"CV MAE: {cv_mae_scores.mean():.3f} (+/- {cv_mae_scores.std():.3f})")
    print(f"CV RMSE: {cv_rmse_scores.mean():.3f} (+/- {cv_rmse_scores.std():.3f})")
    
    # Train/test split evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining on 80% and testing on 20%...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    
    return {
        'model_name': model_name,
        'model': model,
        'cv_mae_mean': cv_mae_scores.mean(),
        'cv_mae_std': cv_mae_scores.std(),
        'cv_rmse_mean': cv_rmse_scores.mean(),
        'cv_rmse_std': cv_rmse_scores.std(),
        'test_mae': test_mae,
        'test_rmse': test_rmse
    }


def compare_models(X, y):
    """Comparing multiple regression models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'Random Forest (Deep)': RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = []
    
    for name, model in models.items():
        result = evaluate_model(model, X, y, name)
        results.append(result)
    
    return results


def create_ensemble(results, X, y):
    """Creating ensemble from top models"""
    print("\n" + "="*80)
    print("ENSEMBLE MODEL")
    print("="*80)
    
    # Getting top 3 models by CV MAE
    sorted_results = sorted(results, key=lambda x: x['cv_mae_mean'])
    top_models = sorted_results[:3]
    
    print(f"\nCombining top 3 models:")
    for i, r in enumerate(top_models, 1):
        print(f"  {i}. {r['model_name']} (CV MAE: {r['cv_mae_mean']:.3f})")
    
    # Simple averaging ensemble
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training all models
    for result in top_models:
        result['model'].fit(X_train, y_train)
    
    # Making predictions
    predictions = []
    for result in top_models:
        pred = result['model'].predict(X_test)
        predictions.append(pred)
    
    # Averaging predictions
    ensemble_pred = np.mean(predictions, axis=0)
    
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    print(f"\nEnsemble Performance:")
    print(f"  Test MAE: {ensemble_mae:.3f}")
    print(f"  Test RMSE: {ensemble_rmse:.3f}")
    
    ensemble_result = {
        'model_name': 'Ensemble (Top 3)',
        # Storing all models
        'model': top_models,  
        'cv_mae_mean': np.mean([r['cv_mae_mean'] for r in top_models]),
        'cv_mae_std': 0,
        'cv_rmse_mean': np.mean([r['cv_rmse_mean'] for r in top_models]),
        'cv_rmse_std': 0,
        'test_mae': ensemble_mae,
        'test_rmse': ensemble_rmse
    }
    
    return ensemble_result


def display_results(results, ensemble_result):
    """Displaying comparison results"""
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    all_results = results + [ensemble_result]
    
    df_results = pd.DataFrame([{
        'Model': r['model_name'],
        'CV MAE': f"{r['cv_mae_mean']:.3f} ± {r['cv_mae_std']:.3f}",
        'Test MAE': f"{r['test_mae']:.3f}",
        'Test RMSE': f"{r['test_rmse']:.3f}"
    } for r in all_results])
    
    # Sorting by Test MAE
    df_results = df_results.sort_values('Test MAE')
    
    print("\n" + df_results.to_string(index=False))
    
    # Finding best model
    best = min(all_results, key=lambda x: x['test_mae'])
    
    print("\n" + "="*80)
    print(f"🏆 BEST MODEL: {best['model_name']}")
    print(f"   Test MAE: {best['test_mae']:.3f} positions")
    print(f"   Test RMSE: {best['test_rmse']:.3f} positions")
    print("="*80)
    
    return best, df_results


def save_best_model(best_result, feature_cols):
    """Saving the best model"""
    print("\nSaving best model...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_info = {
        'model': best_result['model'],
        'model_name': best_result['model_name'],
        'feature_cols': feature_cols,
        'test_mae': best_result['test_mae'],
        'test_rmse': best_result['test_rmse'],
        'timestamp': timestamp
    }
    
    model_path = f'results/best_model_{timestamp}.pkl'
    joblib.dump(model_info, model_path)
    
    # Also save as latest
    joblib.dump(model_info, 'results/best_model_latest.pkl')
    
    print(f"✓ Saved to {model_path}")
    print("✓ Saved to results/best_model_latest.pkl")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AFL PREDICTION - MODEL COMPARISON")
    print("="*80)
    
    # Loading data
    X, y, feature_cols = load_training_data()
    
    # Comparing models
    results = compare_models(X, y)
    
    # Creating ensemble
    ensemble_result = create_ensemble(results, X, y)
    
    # Displaying results
    best, df_results = display_results(results, ensemble_result)
    
    # Saving results
    df_results.to_csv('results/model_comparison_results.csv', index=False)
    print("\n✓ Saved comparison results to results/model_comparison_results.csv")
    
    # Saving best model
    save_best_model(best, feature_cols)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("1. Update predict.py to use the best model")
    print("2. Generate final 2026 predictions")
    print("3. Build production API")
    print("="*80 + "\n")
