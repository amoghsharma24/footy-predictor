import pandas as pd
from ladder_calc import calculate_ladder

def generate_all_ladders(start_year, end_year):
    """Generating ladder for each season from start to end year"""
    ladders = {}
    
    for year in range(start_year, end_year + 1):
        matches = pd.read_csv(f'data/afl_{year}.csv')
        ladder = calculate_ladder(matches)
        ladders[year] = ladder
        print(f"Generated {year} ladder")
    
    return ladders

def create_training_data(ladders_dict, start_year, end_year):
    """Creating training data by pairing consecutive seasons"""
    training_data = []
    
    for year in range(start_year, end_year):
        next_year = year + 1
        
        # Getting features from current year
        current_features = create_features(ladders_dict[year], year)
        
        # Getting next year positions (target)
        next_ladder = ladders_dict[next_year]
        
        # Matching teams between years
        for _, current_row in current_features.iterrows():
            team = current_row['Team']
            
            # Finding this team's position next year
            next_position = next_ladder[next_ladder['Team'] == team]['Position']
            
            if not next_position.empty:
                training_row = current_row.to_dict()
                training_row['Next_Year_Position'] = next_position.values[0]
                training_data.append(training_row)
    
    return pd.DataFrame(training_data)

def create_features(ladder_df, year):
    """Creating features from a season's ladder"""
    features = []
    
    for _, row in ladder_df.iterrows():
        # Calculating per-game averages
        games_played = row['Wins'] + row['Draws'] + row['Losses']
        
        feature_row = {
            'Team': row['Team'],
            'Year': year,
            'Position': row['Position'],
            'Wins': row['Wins'],
            'Draws': row['Draws'],
            'Losses': row['Losses'],
            'Win_Rate': row['Wins'] / games_played if games_played > 0 else 0,
            'Percentage': row['Percentage'],
            'Avg_Points_For': row['Points For'] / games_played if games_played > 0 else 0,
            'Avg_Points_Against': row['Points Against'] / games_played if games_played > 0 else 0,
            'Premiership_Points': row['Premiership Points']
        }
        
        features.append(feature_row)
    
    return pd.DataFrame(features)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("AFL FEATURE ENGINEERING")
    print("="*80)
    
    # Generating ladders for all years
    print("\nStep 1: Generating ladders for all seasons...")
    ladders = generate_all_ladders(2015, 2025)
    
    # Creating training dataset from 2015-2024 (predicting 2016-2025)
    print("\nStep 2: Creating training dataset...")
    training_data = create_training_data(ladders, 2015, 2024)
    
    print(f"\nTraining dataset shape: {training_data.shape}")
    print(f"Features: {list(training_data.columns)}")
    
    # Saving training data
    training_data.to_csv('results/training_data.csv', index=False)
    print("\n✓ Saved results/training_data.csv")
    
    # Preparing 2025 data for prediction
    print("\nStep 3: Preparing 2025 data for prediction...")
    prediction_features = create_features(ladders[2025], 2025)
    prediction_features.to_csv('results/prediction_features_2025.csv', index=False)
    print("✓ Saved results/prediction_features_2025.csv")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Training samples: {len(training_data)}")
    print(f"Years covered: 2015-2024 → 2016-2025")
    print(f"Prediction input: 2025 stats → 2026 ladder")
    print("\nReady for model training!")
    print("="*80 + "\n")