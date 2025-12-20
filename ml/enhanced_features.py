import pandas as pd
import numpy as np
from ml.ladder_calc import calculate_ladder


def calculate_rolling_stats(matches_df, team, windows=[3, 5, 10]):
    """Calculating rolling averages for different window sizes"""
    team_matches = matches_df[
        (matches_df['Home Team'] == team) | 
        (matches_df['Away Team'] == team)
    ].copy()
    
    # Getting team's stats in chronological order
    stats = []
    for _, match in team_matches.iterrows():
        if match['Home Team'] == team:
            score = match['Home Score']
            opp_score = match['Away Score']
            result = 1 if score > opp_score else 0 if score < opp_score else 0.5
        else:
            score = match['Away Score']
            opp_score = match['Home Score']
            result = 1 if score > opp_score else 0 if score < opp_score else 0.5
        
        stats.append({
            'score': score,
            'opp_score': opp_score,
            'result': result,
            'margin': score - opp_score
        })
    
    if len(stats) == 0:
        return {f'rolling_{w}': {} for w in windows}
    
    df_stats = pd.DataFrame(stats)
    rolling_features = {}
    
    # Calculating rolling averages for each window
    for window in windows:
        if len(df_stats) >= window:
            rolling_features[f'rolling_{window}_wins'] = df_stats['result'].tail(window).mean()
            rolling_features[f'rolling_{window}_score'] = df_stats['score'].tail(window).mean()
            rolling_features[f'rolling_{window}_conceded'] = df_stats['opp_score'].tail(window).mean()
            rolling_features[f'rolling_{window}_margin'] = df_stats['margin'].tail(window).mean()
        else:
            rolling_features[f'rolling_{window}_wins'] = 0
            rolling_features[f'rolling_{window}_score'] = 0
            rolling_features[f'rolling_{window}_conceded'] = 0
            rolling_features[f'rolling_{window}_margin'] = 0
    
    return rolling_features


def calculate_streak(matches_df, team):
    """Calculating current win/loss streak at end of season"""
    team_matches = matches_df[
        (matches_df['Home Team'] == team) | 
        (matches_df['Away Team'] == team)
    ]
    
    if len(team_matches) == 0:
        return {'win_streak': 0, 'loss_streak': 0, 'current_streak': 0}
    
    # Getting results in order
    results = []
    for _, match in team_matches.iterrows():
        if match['Home Team'] == team:
            if match['Home Score'] > match['Away Score']:
                results.append('W')
            elif match['Home Score'] < match['Away Score']:
                results.append('L')
            else:
                results.append('D')
        else:
            if match['Away Score'] > match['Home Score']:
                results.append('W')
            elif match['Away Score'] < match['Home Score']:
                results.append('L')
            else:
                results.append('D')
    
    # Calculating current streak
    current_streak = 0
    if results:
        last_result = results[-1]
        for result in reversed(results):
            if result == last_result:
                current_streak += 1 if last_result == 'W' else -1 if last_result == 'L' else 0
            else:
                break
    
    # Finding longest streaks
    win_streak = 0
    loss_streak = 0
    current_win = 0
    current_loss = 0
    
    for result in results:
        if result == 'W':
            current_win += 1
            current_loss = 0
            win_streak = max(win_streak, current_win)
        elif result == 'L':
            current_loss += 1
            current_win = 0
            loss_streak = max(loss_streak, current_loss)
        else:
            current_win = 0
            current_loss = 0
    
    return {
        'win_streak': win_streak,
        'loss_streak': loss_streak,
        'current_streak': current_streak
    }


def calculate_opponent_strength(matches_df, team, ladder_df):
    """Calculating average strength of opponents faced"""
    team_matches = matches_df[
        (matches_df['Home Team'] == team) | 
        (matches_df['Away Team'] == team)
    ]
    
    opponent_positions = []
    top_8_count = 0
    bottom_8_count = 0
    
    for _, match in team_matches.iterrows():
        opponent = match['Away Team'] if match['Home Team'] == team else match['Home Team']
        
        # Finding opponent's ladder position
        opp_position = ladder_df[ladder_df['Team'] == opponent]['Position']
        
        if not opp_position.empty:
            pos = opp_position.values[0]
            opponent_positions.append(pos)
            
            if pos <= 8:
                top_8_count += 1
            else:
                bottom_8_count += 1
    
    if len(opponent_positions) == 0:
        return {
            'avg_opponent_position': 9.5,
            'top_8_opponents_rate': 0.5,
            'bottom_8_opponents_rate': 0.5,
            'schedule_difficulty': 0
        }
    
    total_games = len(opponent_positions)
    
    return {
        'avg_opponent_position': np.mean(opponent_positions),
        'top_8_opponents_rate': top_8_count / total_games,
        'bottom_8_opponents_rate': bottom_8_count / total_games,
        # Positive = harder schedule
        'schedule_difficulty': top_8_count - bottom_8_count
    }


def calculate_momentum(matches_df, team):
    """Calculating momentum by comparing early vs late season performance"""
    team_matches = matches_df[
        (matches_df['Home Team'] == team) | 
        (matches_df['Away Team'] == team)
    ]
    
    if len(team_matches) < 6:
        return {'momentum': 0}
    
    # Getting win rates for different parts of season
    total = len(team_matches)
    first_third = team_matches.head(total // 3)
    last_third = team_matches.tail(total // 3)
    
    def get_win_rate(matches):
        wins = 0
        for _, match in matches.iterrows():
            if match['Home Team'] == team:
                if match['Home Score'] > match['Away Score']:
                    wins += 1
            else:
                if match['Away Score'] > match['Home Score']:
                    wins += 1
        return wins / len(matches) if len(matches) > 0 else 0
    
    early_win_rate = get_win_rate(first_third)
    late_win_rate = get_win_rate(last_third)
    
    # Momentum is improvement from early to late season
    momentum = late_win_rate - early_win_rate
    
    return {'momentum': momentum}


def create_enhanced_features(matches_df, ladder_df, year):
    """Creating comprehensive feature set with all enhancements"""
    features = []
    
    for _, row in ladder_df.iterrows():
        team = row['Team']
        games_played = row['Wins'] + row['Draws'] + row['Losses']
        
        # Basic features
        basic = {
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
        
        # Rolling averages
        rolling = calculate_rolling_stats(matches_df, team)
        
        # Streaks
        streaks = calculate_streak(matches_df, team)
        
        # Opponent strength
        opp_strength = calculate_opponent_strength(matches_df, team, ladder_df)
        
        # Momentum
        momentum = calculate_momentum(matches_df, team)
        
        # Combining all features
        enhanced = {
            **basic,
            **rolling,
            **streaks,
            **opp_strength,
            **momentum
        }
        
        features.append(enhanced)
    
    return pd.DataFrame(features)


def generate_all_ladders_with_matches(start_year, end_year):
    """Generating ladders and keeping match data"""
    ladders = {}
    matches_dict = {}
    
    for year in range(start_year, end_year + 1):
        matches = pd.read_csv(f'data/afl_{year}.csv')
        ladder = calculate_ladder(matches)
        
        ladders[year] = ladder
        matches_dict[year] = matches
        print(f"Loaded {year} data")
    
    return ladders, matches_dict


def create_enhanced_training_data(ladders, matches_dict, start_year, end_year):
    """Creating training data with enhanced features"""
    training_data = []
    
    for year in range(start_year, end_year):
        next_year = year + 1
        
        print(f"Processing {year} → {next_year}...")
        
        # Creating enhanced features
        current_features = create_enhanced_features(
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
    print("ENHANCED FEATURE ENGINEERING")
    print("="*80)
    
    # Loading data
    print("\nLoading data...")
    ladders, matches = generate_all_ladders_with_matches(2015, 2025)
    
    # Creating enhanced features
    print("\nCreating enhanced features...")
    training_data = create_enhanced_training_data(ladders, matches, 2015, 2024)
    
    print(f"\nTraining dataset shape: {training_data.shape}")
    print(f"\nFeature count: {len(training_data.columns) - 1}")
    
    # Saving
    training_data.to_csv('results/training_data_enhanced.csv', index=False)
    print("\n✓ Saved results/training_data_enhanced.csv")
    
    # Preparing 2025
    print("\nPreparing 2025 data...")
    prediction_features = create_enhanced_features(matches[2025], ladders[2025], 2025)
    prediction_features.to_csv('results/prediction_features_2025_enhanced.csv', index=False)
    print("✓ Saved results/prediction_features_2025_enhanced.csv")
    
    print("\n" + "="*80)
    print("NEW FEATURES ADDED:")
    print("- Rolling averages (3, 5, 10 games)")
    print("- Win/loss streaks")
    print("- Opponent strength metrics")
    print("- Schedule difficulty")
    print("- Season momentum")
    print("="*80 + "\n")
