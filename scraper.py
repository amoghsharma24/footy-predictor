import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


def scrape_season(year):
    """Scraping AFL match data from afltables.com for a given season"""
    url = f"https://afltables.com/afl/seas/{year}.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    matches = []
    
    # afltables uses simple tables with match data so no special classes will be needed
    for table in soup.find_all('table'):
        text = table.get_text()
        
        # Quick filter: match tables have "won by" text and score patterns
        if 'won by' not in text or '.' not in text:
            continue
            
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        # Each match table has exactly 2 lines: home team and then away team
        if len(lines) != 2:
            continue
        
        home_line = lines[0]
        away_line = lines[1]
        
        # Extracting team names they are always before the first number
        home_match = re.match(r'([A-Za-z\s]+?)\s+\d', home_line)
        away_match = re.match(r'([A-Za-z\s]+?)\s+\d', away_line)
        
        if not home_match or not away_match:
            continue
        
        home_team = home_match.group(1).strip()
        away_team = away_match.group(1).strip()
        
        # Getting final scores last number before the date/result text
        home_score = re.search(r'(\d+)(?=\w{3}\s+\d{2}-)', home_line)
        away_score = re.search(r'(\d+)(?=\w+\s+won)', away_line)
        
        if not home_score or not away_score:
            continue
        
        # Extracting date and venue
        date_match = re.search(r'(\w{3}\s+\d{2}-\w{3}-\d{4})', home_line)
        date = date_match.group(1) if date_match else "Unknown"
        
        venue_match = re.search(r'Venue:\s*(.+?)$', home_line)
        venue = venue_match.group(1).strip() if venue_match else "Unknown"
        
        matches.append({
            'Date': date,
            'Home Team': home_team,
            'Away Team': away_team,
            'Home Score': int(home_score.group(1)),
            'Away Score': int(away_score.group(1)),
            'Venue': venue
        })
    
    return matches


if __name__ == "__main__":
    # Scraping seasons from 2015 to 2025 and saving each to CSV
    for year in range(2015, 2026):
        print(f"Scraping {year} season...")
        matches = scrape_season(year)
        
        if matches:
            df = pd.DataFrame(matches)
            filename = f'afl_{year}.csv'
            df.to_csv(filename, index=False)
            print(f"Saved {len(matches)} matches to {filename}\n")
        else:
            print(f"No matches found for {year}\n")
