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
    
    # Parsing tables with tr and td elements
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        
        # Match tables have exactly 2 rows
        if len(rows) != 2:
            continue
        
        # Each row should have 4 cells: team, quarters, score, details
        row1_cells = rows[0].find_all('td')
        row2_cells = rows[1].find_all('td')
        
        if len(row1_cells) != 4 or len(row2_cells) != 4:
            continue
        
        # Extracting data from cells
        home_team = row1_cells[0].get_text(strip=True)
        home_score = row1_cells[2].get_text(strip=True)
        away_team = row2_cells[0].get_text(strip=True)
        away_score = row2_cells[2].get_text(strip=True)
        
        # Getting date and venue from row 1, cell 3
        details = row1_cells[3].get_text(strip=True)
        
        # Extracting date
        date_match = re.search(r'(\w{3}\s+\d{2}-\w{3}-\d{4})', details)
        date = date_match.group(1) if date_match else "Unknown"
        
        # Extracting venue
        venue_match = re.search(r'Venue:\s*(.+?)$', details)
        venue = venue_match.group(1).strip() if venue_match else "Unknown"
        
        # Validating scores are numbers
        try:
            home_score_int = int(home_score)
            away_score_int = int(away_score)
        except ValueError:
            continue
        
        matches.append({
            'Date': date,
            'Home Team': home_team,
            'Away Team': away_team,
            'Home Score': home_score_int,
            'Away Score': away_score_int,
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
            filename = f'data/afl_{year}.csv'
            df.to_csv(filename, index=False)
            print(f"Saved {len(matches)} matches to {filename}\n")
        else:
            print(f"No matches found for {year}\n")
