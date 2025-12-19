import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def prepare_training_data(use_advanced=True):
    """Loading and preparing training data for the model"""
    print("Loading training data...")
    
    if use_advanced:
        df = pd.read_csv('results/training_data_advanced.csv')
        feature_cols = ['Position', 'Wins', 'Draws', 'Losses', 'Win_Rate', 
                        'Percentage', 'Avg_Points_For', 'Avg_Points_Against', 
                        'Premiership_Points', 'form_last_5', 'home_win_rate',
                        'away_win_rate', 'home_away_diff', 'scoring_trend',
                        'consistency', 'big_wins_rate', 'big_losses_rate']
        print("Using ADVANCED features")
    else:
        df = pd.read_csv('results/training_data.csv')
        feature_cols = ['Position', 'Wins', 'Draws', 'Losses', 'Win_Rate', 
                        'Percentage', 'Avg_Points_For', 'Avg_Points_Against', 
                        'Premiership_Points']
        print("Using BASIC features")
    
    X = df[feature_cols]
    y = df['Next_Year_Position']
    
    print(f"Training samples: {len(X)}")
    print(f"Features: {len(feature_cols)}")
    
    return X, y, feature_cols

def train_model(X, y):
    """Training Random Forest model and evaluating performance"""
    print("\nSplitting data for validation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluating on test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"\nModel Performance:")
    print(f"  Mean Absolute Error: {mae:.2f} positions")
    print(f"  Root Mean Squared Error: {rmse:.2f} positions")
    
    return model

def predict_2026(model, feature_cols):
    """Predicting 2026 ladder based on 2025 stats"""
    print("\nLoading 2025 data for prediction...")
    df_2025 = pd.read_csv('results/prediction_features_2025.csv')
    
    # Preparing features
    X_pred = df_2025[feature_cols]
    
    # Making predictions
    predicted_positions = model.predict(X_pred)
    
    # Creating results dataframe
    results = pd.DataFrame({
        'Team': df_2025['Team'],
        '2025_Position': df_2025['Position'],
        'Predicted_2026_Position': predicted_positions
    })
    
    # Sorting by predicted position
    results = results.sort_values('Predicted_2026_Position')
    results['Predicted_2026_Position'] = results['Predicted_2026_Position'].round(1)
    
    return results

if __name__ == "__main__":
    print("\n" + "="*80)
    print("AFL 2026 LADDER PREDICTION")
    print("="*80)
    
    # Preparing training data
    X, y, feature_cols = prepare_training_data()
    
    # Training model
    model = train_model(X, y)
    
    # Feature importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"  {row['Feature']:<25} {row['Importance']:.4f}")
    
    # Making 2026 predictions
    predictions = predict_2026(model, feature_cols)
    
    print("\n" + "="*80)
    print("PREDICTED 2026 AFL LADDER")
    print("="*80)
    print(f"\n{'Pos':<6} {'Team':<25} {'2025 Pos':<10} {'Predicted 2026':<15}")
    print("-"*80)
    
    for idx, (_, row) in enumerate(predictions.iterrows(), 1):
        change = row['2025_Position'] - row['Predicted_2026_Position']
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"{idx:<6} {row['Team']:<25} {int(row['2025_Position']):<10} {row['Predicted_2026_Position']:<15.1f} {arrow}")
    
    print("="*80)
    print("\nNote: Predictions are based on 2015-2024 historical data")
    print("="*80 + "\n")
    
    # Saving predictions
    predictions.to_csv('results/predicted_2026_ladder.csv', index=False)
    print("✓ Saved results/predicted_2026_ladder.csv\n")