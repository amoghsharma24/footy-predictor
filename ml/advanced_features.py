import pandas as pd
import numpy as np
from ml.ladder_calc import calculate_ladder


def calculate_form(matches_df, team, last_n=5):
    """Calculating win rate over last N games"""
    team_matches = matches_df[
        (matches_df['Home Team'] == team) | 
        (matches_df['Away Team'] == team)
    ]
    
    # Getting last N matches
    recent_matches = team_matches.tail(last_n)
    
    if len(recent_matches) == 0:
        return 0
    
    wins = 0
    for _, match in recent_matches.iterrows():
        if match['Home Team'] == team:
            if match['Home Score'] > match['Away Score']:
                wins += 1
        else:
            if match['Away Score'] > match['Home Score']:
                wins += 1
    
    return wins / len(recent_matches)


def calculate_home_away_split(matches_df, team):
    """Calculating separate home and away performance"""
    home_matches = matches_df[matches_df['Home Team'] == team]
    away_matches = matches_df[matches_df['Away Team'] == team]
    
    # Home stats
    home_wins = len(home_matches[home_matches['Home Score'] > home_matches['Away Score']])
    home_win_rate = home_wins / len(home_matches) if len(home_matches) > 0 else 0
    home_avg_score = home_matches['Home Score'].mean() if len(home_matches) > 0 else 0
    
    # Away stats
    away_wins = len(away_matches[away_matches['Away Score'] > away_matches['Home Score']])
    away_win_rate = away_wins / len(away_matches) if len(away_matches) > 0 else 0
    away_avg_score = away_matches['Away Score'].mean() if len(away_matches) > 0 else 0
    
    return {
        'home_win_rate': home_win_rate,
        'away_win_rate': away_win_rate,
        'home_away_diff': home_win_rate - away_win_rate,
        'home_avg_score': home_avg_score,
        'away_avg_score': away_avg_score
    }


def calculate_scoring_trend(matches_df, team):
    """Calculating if team is improving or declining through season"""
    team_matches = matches_df[
        (matches_df['Home Team'] == team) | 
        (matches_df['Away Team'] == team)
    ].copy()
    
    if len(team_matches) < 3:
        return 0
    
    # Getting team's scores in order
    scores = []
    for _, match in team_matches.iterrows():
        if match['Home Team'] == team:
            scores.append(match['Home Score'])
        else:
            scores.append(match['Away Score'])
    
    # Simple linear trend: comparing first half to second half
    mid = len(scores) // 2
    first_half_avg = np.mean(scores[:mid])
    second_half_avg = np.mean(scores[mid:])
    
    trend = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
    return trend


def calculate_consistency(matches_df, team):
    """Calculating how consistent team's performance is"""
    team_matches = matches_df[
        (matches_df['Home Team'] == team) | 
        (matches_df['Away Team'] == team)
    ]
    
    # Getting all margins
    margins = []
    for _, match in team_matches.iterrows():
        if match['Home Team'] == team:
            margin = match['Home Score'] - match['Away Score']
        else:
            margin = match['Away Score'] - match['Home Score']
        margins.append(margin)
    
    if len(margins) == 0:
        return 0
    
    # Lower standard deviation = more consistent
    consistency = np.std(margins)
    return consistency


def calculate_big_wins_losses(matches_df, team, margin_threshold=40):
    """Counting blowout wins and heavy losses"""
    team_matches = matches_df[
        (matches_df['Home Team'] == team) | 
        (matches_df['Away Team'] == team)
    ]
    
    big_wins = 0
    big_losses = 0
    
    for _, match in team_matches.iterrows():
        if match['Home Team'] == team:
            margin = match['Home Score'] - match['Away Score']
        else:
            margin = match['Away Score'] - match['Home Score']
        
        if margin >= margin_threshold:
            big_wins += 1
        elif margin <= -margin_threshold:
            big_losses += 1
    
    total_games = len(team_matches)
    
    return {
        'big_wins_rate': big_wins / total_games if total_games > 0 else 0,
        'big_losses_rate': big_losses / total_games if total_games > 0 else 0
    }


def create_advanced_features(matches_df, ladder_df, year):
    """Creating advanced features combining match data and ladder"""
    features = []
    
    for _, row in ladder_df.iterrows():
        team = row['Team']
        games_played = row['Wins'] + row['Draws'] + row['Losses']
        
        # Basic ladder features
        basic_features = {
            'Team': team,
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
        
        # Advanced features from match data
        form = calculate_form(matches_df, team, last_n=5)
        home_away = calculate_home_away_split(matches_df, team)
        trend = calculate_scoring_trend(matches_df, team)
        consistency = calculate_consistency(matches_df, team)
        big_games = calculate_big_wins_losses(matches_df, team)
        
        # Combining all features
        advanced_features = {
            **basic_features,
            'form_last_5': form,
            'home_win_rate': home_away['home_win_rate'],
            'away_win_rate': home_away['away_win_rate'],
            'home_away_diff': home_away['home_away_diff'],
            'scoring_trend': trend,
            'consistency': consistency,
            'big_wins_rate': big_games['big_wins_rate'],
            'big_losses_rate': big_games['big_losses_rate']
        }
        
        features.append(advanced_features)
    
    return pd.DataFrame(features)


def generate_all_ladders_with_matches(start_year, end_year):
    """Generating ladders and keeping match data for advanced features"""
    ladders = {}
    matches_dict = {}
    
    for year in range(start_year, end_year + 1):
        matches = pd.read_csv(f'data/afl_{year}.csv')
        ladder = calculate_ladder(matches)
        
        ladders[year] = ladder
        matches_dict[year] = matches
        print(f"Loaded {year} data")
    
    return ladders, matches_dict


def create_advanced_training_data(ladders, matches_dict, start_year, end_year):
    """Creating training data with advanced features"""
    training_data = []
    
    for year in range(start_year, end_year):
        next_year = year + 1
        
        print(f"Processing {year} → {next_year}...")
        
        # Creating features from current year
        current_features = create_advanced_features(
            matches_dict[year], 
            ladders[year], 
            year
        )
        
        # Getting next year positions
        next_ladder = ladders[next_year]
        
        # Matching teams
        for _, current_row in current_features.iterrows():
            team = current_row['Team']
            
            next_position = next_ladder[next_ladder['Team'] == team]['Position']
            
            if not next_position.empty:
                training_row = current_row.to_dict()
                training_row['Next_Year_Position'] = next_position.values[0]
                training_data.append(training_row)
    
    return pd.DataFrame(training_data)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*80)
    
    # Generating ladders and loading match data
    print("\nLoading data...")
    ladders, matches = generate_all_ladders_with_matches(2015, 2025)
    
    # Creating advanced training dataset
    print("\nCreating advanced features...")
    training_data = create_advanced_training_data(ladders, matches, 2015, 2024)
    
    print(f"\nTraining dataset shape: {training_data.shape}")
    print(f"\nFeatures: {list(training_data.columns)}")
    
    # Saving
    training_data.to_csv('results/training_data_advanced.csv', index=False)
    print("\n✓ Saved results/training_data_advanced.csv")
    
    # Preparing 2025 for prediction
    print("\nPreparing 2025 data...")
    prediction_features = create_advanced_features(matches[2025], ladders[2025], 2025)
    prediction_features.to_csv('results/prediction_features_2025_advanced.csv', index=False)
    print("✓ Saved results/prediction_features_2025_advanced.csv")
    
    print("\n" + "="*80)
    print(f"Training samples: {len(training_data)}")
    print(f"Total features: {len(training_data.columns) - 1}")  # -1 for target
    print(f"New advanced features: 8")
    print("="*80 + "\n")
