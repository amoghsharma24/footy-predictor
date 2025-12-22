import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from datetime import datetime


# Team name mappings for AFL Tables URLs
TEAM_MAPPINGS = {
    'Adelaide': 'adelaide',
    'Brisbane Lions': 'brisbanel',
    'Carlton': 'carlton',
    'Collingwood': 'collingwood',
    'Essendon': 'essendon',
    'Fremantle': 'fremantle',
    'Geelong': 'geelong',
    'Gold Coast': 'goldcoast',
    'GWS': 'gws',
    'Hawthorn': 'hawthorn',
    'Melbourne': 'melbourne',
    'North Melbourne': 'kangaroos',
    'Port Adelaide': 'padelaide',
    'Richmond': 'richmond',
    'St Kilda': 'stkilda',
    'Sydney': 'swans',
    'West Coast': 'westcoast',
    'Western Bulldogs': 'bullldogs'
}


def scrape_season_player_stats(year):
    """Scrape detailed player statistics for all teams in a season from AFL Tables"""
    
    url = f"https://afltables.com/afl/stats/{year}.html"
    print(f"\nScraping {year} season player stats from {url}")
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching season stats for {year}: {e}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    all_players = []
    current_team = None
    
    # Find all tables - each team's stats are in a separate table
    tables = soup.find_all('table')
    
    for table in tables:
        rows = table.find_all('tr')
        
        # Look for team headers - they typically have team names in bold or links
        for row in rows:
            # Check if this is a team header row
            header_text = row.get_text(strip=True)
            
            # Team headers typically include "[Players Game by Game][Team Game by Game]"
            if '[Players Game by Game]' in header_text or '[Team Game by Game]' in header_text:
                # Extract team name from the header
                team_name = header_text.split('[')[0].strip()
                current_team = team_name
                print(f"  Found team: {current_team}")
                continue
            
            # Skip header rows with column names
            if 'Player' in header_text and 'GM' in header_text:
                continue
            
            # Parse player stat rows
            cells = row.find_all('td')
            
            if len(cells) >= 20 and current_team:  # Player rows have many columns
                try:
                    # Extract stats - columns are: #, Player, GM, KI, MK, HB, DI, DA, GL, BH, HO, TK, RB, IF, CL, CG, FF, FA, BR, CP, UP, CM, MI, 1%, BO, GA, %P, SU
                    
                    jumper = cells[0].get_text(strip=True)
                    player_name = cells[1].get_text(strip=True)
                    
                    # Skip summary rows like "37 players used"
                    if 'players used' in player_name.lower() or player_name == '':
                        continue
                    
                    def safe_int(val):
                        try:
                            return int(val.replace(' ', '')) if val.strip() else 0
                        except:
                            return 0
                    
                    def safe_float(val):
                        try:
                            return float(val.replace(' ', '')) if val.strip() else 0.0
                        except:
                            return 0.0
                    
                    games = safe_int(cells[2].get_text(strip=True))
                    kicks = safe_int(cells[3].get_text(strip=True))
                    marks = safe_int(cells[4].get_text(strip=True))
                    handballs = safe_int(cells[5].get_text(strip=True))
                    disposals = safe_int(cells[6].get_text(strip=True))
                    disposal_avg = safe_float(cells[7].get_text(strip=True))
                    goals = safe_int(cells[8].get_text(strip=True))
                    behinds = safe_int(cells[9].get_text(strip=True))
                    hitouts = safe_int(cells[10].get_text(strip=True))
                    tackles = safe_int(cells[11].get_text(strip=True))
                    rebound_50s = safe_int(cells[12].get_text(strip=True))
                    inside_50s = safe_int(cells[13].get_text(strip=True))
                    clearances = safe_int(cells[14].get_text(strip=True))
                    clangers = safe_int(cells[15].get_text(strip=True))
                    free_kicks_for = safe_int(cells[16].get_text(strip=True))
                    free_kicks_against = safe_int(cells[17].get_text(strip=True))
                    brownlow_votes = safe_int(cells[18].get_text(strip=True))
                    contested_poss = safe_int(cells[19].get_text(strip=True))
                    uncontested_poss = safe_int(cells[20].get_text(strip=True))
                    
                    # Optional columns (may not exist for all rows)
                    contested_marks = safe_int(cells[21].get_text(strip=True)) if len(cells) > 21 else 0
                    marks_inside_50 = safe_int(cells[22].get_text(strip=True)) if len(cells) > 22 else 0
                    one_percenters = safe_int(cells[23].get_text(strip=True)) if len(cells) > 23 else 0
                    bounces = safe_int(cells[24].get_text(strip=True)) if len(cells) > 24 else 0
                    goal_assists = safe_int(cells[25].get_text(strip=True)) if len(cells) > 25 else 0
                    
                    all_players.append({
                        'team': current_team,
                        'year': year,
                        'jumper_number': jumper,
                        'player_name': player_name,
                        'games': games,
                        'kicks': kicks,
                        'marks': marks,
                        'handballs': handballs,
                        'disposals': disposals,
                        'disposal_avg': disposal_avg,
                        'goals': goals,
                        'behinds': behinds,
                        'hitouts': hitouts,
                        'tackles': tackles,
                        'rebound_50s': rebound_50s,
                        'inside_50s': inside_50s,
                        'clearances': clearances,
                        'clangers': clangers,
                        'free_kicks_for': free_kicks_for,
                        'free_kicks_against': free_kicks_against,
                        'brownlow_votes': brownlow_votes,
                        'contested_possessions': contested_poss,
                        'uncontested_possessions': uncontested_poss,
                        'contested_marks': contested_marks,
                        'marks_inside_50': marks_inside_50,
                        'one_percenters': one_percenters,
                        'bounces': bounces,
                        'goal_assists': goal_assists
                    })
                    
                except (ValueError, IndexError, AttributeError) as e:
                    # Skip malformed rows
                    continue
    
    if len(all_players) == 0:
        print(f"  No player stats found for {year}")
        return None
    
    print(f"  Total players found: {len(all_players)}")
    return pd.DataFrame(all_players)


def aggregate_team_season_metrics(players_df):
    """Aggregate season player stats to team-level metrics"""
    
    if players_df is None or len(players_df) == 0:
        return None
    
    team_metrics = []
    
    for team in players_df['team'].unique():
        team_players = players_df[players_df['team'] == team]
        
        total_team_games = team_players['games'].sum()
        
        metrics = {
            'team': team,
            'year': team_players['year'].iloc[0],
            
            # Total team stats
            'total_disposals': team_players['disposals'].sum(),
            'total_goals': team_players['goals'].sum(),
            'total_behinds': team_players['behinds'].sum(),
            'total_tackles': team_players['tackles'].sum(),
            'total_clearances': team_players['clearances'].sum(),
            'total_inside_50s': team_players['inside_50s'].sum(),
            'total_marks': team_players['marks'].sum(),
            'total_contested_marks': team_players['contested_marks'].sum(),
            'total_hitouts': team_players['hitouts'].sum(),
            
            # Average stats per player
            'avg_disposal_per_player': team_players['disposals'].mean(),
            'avg_goals_per_player': team_players['goals'].mean(),
            'avg_tackles_per_player': team_players['tackles'].mean(),
            
            # Elite player counts (season-based)
            'players_20plus_goals': len(team_players[team_players['goals'] >= 20]),
            'players_30plus_goals': len(team_players[team_players['goals'] >= 30]),
            'players_40plus_goals': len(team_players[team_players['goals'] >= 40]),
            'players_300plus_disposals': len(team_players[team_players['disposals'] >= 300]),
            'players_400plus_disposals': len(team_players[team_players['disposals'] >= 400]),
            'players_500plus_disposals': len(team_players[team_players['disposals'] >= 500]),
            'players_100plus_tackles': len(team_players[team_players['tackles'] >= 100]),
            
            # List depth
            'total_players_used': len(team_players),
            'players_with_10plus_games': len(team_players[team_players['games'] >= 10]),
            'players_with_15plus_games': len(team_players[team_players['games'] >= 15]),
            'players_with_20plus_games': len(team_players[team_players['games'] >= 20]),
            
            # Brownlow contention
            'total_brownlow_votes': team_players['brownlow_votes'].sum(),
            'players_with_brownlow_votes': len(team_players[team_players['brownlow_votes'] > 0]),
            'top_brownlow_vote_getter': team_players['brownlow_votes'].max(),
            
            # Scoring efficiency
            'goal_accuracy': (team_players['goals'].sum() / (team_players['goals'].sum() + team_players['behinds'].sum()) * 100) if (team_players['goals'].sum() + team_players['behinds'].sum()) > 0 else 0,
            
            # Contested vs uncontested
            'total_contested_poss': team_players['contested_possessions'].sum(),
            'total_uncontested_poss': team_players['uncontested_possessions'].sum(),
            'contested_poss_ratio': (team_players['contested_possessions'].sum() / team_players['disposals'].sum() * 100) if team_players['disposals'].sum() > 0 else 0,
            
            # Discipline
            'total_clangers': team_players['clangers'].sum(),
            'free_kick_differential': team_players['free_kicks_for'].sum() - team_players['free_kicks_against'].sum(),
        }
        
        team_metrics.append(metrics)
    
    return pd.DataFrame(team_metrics)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AFL SEASON PLAYER STATISTICS SCRAPER")
    print("="*80)
    
    # Scrape multiple years
    start_year = 2015
    end_year = 2025
    
    all_player_stats = []
    all_team_metrics = []
    
    for year in range(start_year, end_year + 1):
        print(f"\n{'='*80}")
        print(f"SCRAPING {year}")
        print(f"{'='*80}")
        
        players_df = scrape_season_player_stats(year)
        
        if players_df is not None and not players_df.empty:
            # Save individual year data
            players_df.to_csv(f'data/season_player_stats_{year}.csv', index=False)
            print(f"✓ Saved data/season_player_stats_{year}.csv ({len(players_df)} players)")
            
            all_player_stats.append(players_df)
            
            # Aggregate to team metrics
            team_metrics = aggregate_team_season_metrics(players_df)
            if team_metrics is not None:
                team_metrics.to_csv(f'data/team_season_metrics_{year}.csv', index=False)
                print(f"✓ Saved data/team_season_metrics_{year}.csv ({len(team_metrics)} teams)")
                all_team_metrics.append(team_metrics)
        
        # Be polite to the server
        time.sleep(2)
    
    # Combine all years
    if all_player_stats:
        combined_players = pd.concat(all_player_stats, ignore_index=True)
        combined_players.to_csv('data/season_player_stats_all_years.csv', index=False)
        print(f"\n{'='*80}")
        print(f"✓ Saved data/season_player_stats_all_years.csv")
        print(f"  Total player-season records: {len(combined_players)}")
        print(f"{'='*80}\n")
    
    if all_team_metrics:
        combined_metrics = pd.concat(all_team_metrics, ignore_index=True)
        combined_metrics.to_csv('data/team_season_metrics_all_years.csv', index=False)
        print(f"✓ Saved data/team_season_metrics_all_years.csv")
        print(f"  Total team-season records: {len(combined_metrics)}")
        print(f"  Years: {start_year}-{end_year}")
        print(f"{'='*80}\n")
        
        print("NEW METRICS ADDED:")
        print("- Total team disposals, goals, tackles, clearances")
        print("- Elite player counts (20+, 30+, 40+ goals)")
        print("- High disposal players (300+, 400+, 500+)")
        print("- Brownlow medal contention metrics")
        print("- Goal accuracy percentage")
        print("- Contested vs uncontested possession ratio")
        print("- Free kick differential (discipline)")
        print("- List depth (players with 10+, 15+, 20+ games)")
        print("="*80 + "\n")

