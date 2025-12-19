import pandas as pd
from collections import defaultdict


def calculate_ladder(matches_df):
    """Calculating AFL ladder from match results"""
    # Tracking each team's stats throughout the season
    team_stats = defaultdict(lambda: {
        'wins': 0,
        'draws': 0,
        'losses': 0,
        'points_for': 0,
        'points_against': 0
    })

    # Processing each match in the DataFrame
    for _, match in matches_df.iterrows():
        home_team = match['Home Team']
        away_team = match['Away Team']
        home_score = match['Home Score']
        away_score = match['Away Score']

        # Updating points for both teams
        team_stats[home_team]['points_for'] += home_score
        team_stats[away_team]['points_for'] += away_score
        team_stats[home_team]['points_against'] += away_score
        team_stats[away_team]['points_against'] += home_score

        # Determining match result
        if home_score > away_score:
            team_stats[home_team]['wins'] += 1
            team_stats[away_team]['losses'] += 1
        elif home_score < away_score:
            team_stats[home_team]['losses'] += 1
            team_stats[away_team]['wins'] += 1
        else:
            team_stats[home_team]['draws'] += 1
            team_stats[away_team]['draws'] += 1

    # Building the ladder from collected stats
    ladder = []
    for team, stats in team_stats.items():
        premiership_points = (stats['wins'] * 4) + (stats['draws'] * 2)
        percentage = (stats['points_for'] / stats['points_against']) * 100 if stats['points_against'] > 0 else 0
        
        ladder.append({
            'Team': team,
            'Wins': stats['wins'],
            'Draws': stats['draws'],
            'Losses': stats['losses'],
            'Points For': stats['points_for'],
            'Points Against': stats['points_against'],
            'Percentage': round(percentage, 2),
            'Premiership Points': premiership_points
        })
    
    ladder_df = pd.DataFrame(ladder)
    
    # Sorting by premiership points then percentage
    ladder_df = ladder_df.sort_values(
        by=['Premiership Points', 'Percentage'],
        ascending=[False, False]
    ).reset_index(drop=True)
    
    # Adding position column
    ladder_df.insert(0, 'Position', range(1, len(ladder_df) + 1))
    
    return ladder_df


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AFL LADDER CALCULATOR")
    print("="*80)
    
    year = input("\nWhich year are you looking for (2015-2025): ").strip()
    
    try:
        year_int = int(year)
        if year_int < 2015 or year_int > 2025:
            print(f"Error: Year must be between 2015 and 2025")
            exit(1)
    except ValueError:
        print(f"Error: Please enter a valid year")
        exit(1)
    
    try:
        matches = pd.read_csv(f'afl_{year}.csv')
    except FileNotFoundError:
        print(f"Error: No data found for {year}. Run scraper.py first.")
        exit(1)
    
    ladder = calculate_ladder(matches)
    
    print(f"\n{year} AFL LADDER")
    print("="*80)
    
    # Creating a cleaner display format
    print(f"\n{'Pos':<4} {'Team':<25} {'W':<3} {'D':<3} {'L':<3} {'PF':<6} {'PA':<6} {'%':<8} {'Pts':<4}")
    print("-"*80)
    
    for _, row in ladder.iterrows():
        print(f"{row['Position']:<4} {row['Team']:<25} {row['Wins']:<3} {row['Draws']:<3} {row['Losses']:<3} "
              f"{row['Points For']:<6} {row['Points Against']:<6} {row['Percentage']:<8.2f} {row['Premiership Points']:<4}")
    
    print("="*80)
    print(f"\nLegend: Pos=Position, W=Wins, D=Draws, L=Losses, PF=Points For, PA=Points Against, %=Percentage, Pts=Premiership Points\n")