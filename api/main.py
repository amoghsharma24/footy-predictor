"""
FastAPI application for AFL ladder prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Adding parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.models import (
    LadderResponse, PredictionResponse, ModelComparisonResponse,
    HealthResponse, TeamStats, PredictionTeam, ModelMetrics,
    TeamsListResponse, TeamInfo, TeamHistoryResponse, TeamHistoryEntry,
    MatchesResponse, Match, FeaturesResponse, FeatureImportance
)
from ml.ladder_calc import calculate_ladder
from ml.enhanced_features import create_enhanced_features

# Global variables for model and data
model_data = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    try:
        # Loading the best model
        model_path = Path(__file__).parent.parent / "results" / "best_model_latest.pkl"
        model_info = joblib.load(model_path)
        
        model_data["model"] = model_info["model"]
        model_data["features"] = model_info["feature_cols"]
        model_data["model_name"] = model_info["model_name"]
        model_data["test_mae"] = model_info["test_mae"]
        model_data["timestamp"] = model_info["timestamp"]
        model_data["loaded"] = True
        
        print(f"✓ Model loaded: {model_info['model_name']}")
        print(f"✓ MAE: {model_info['test_mae']:.3f}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        model_data["loaded"] = False
    
    yield
    
    # Cleanup (if needed)
    model_data.clear()


app = FastAPI(
    title="AFL Ladder Predictor API",
    description="Predict AFL ladder positions using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    # TODO: In production specify your frontend URL
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AFL Ladder Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "historical": "/historical/{year}",
            "compare": "/compare",
            "teams": "/teams",
            "team_history": "/teams/{team}/history",
            "team_prediction": "/predict/team/{team}",
            "matches": "/matches/{year}",
            "team_matches": "/teams/{team}/matches/{year}",
            "features": "/features"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_data.get("loaded", False) else "degraded",
        model_loaded=model_data.get("loaded", False),
        timestamp=datetime.now()
    )


@app.get("/teams", response_model=TeamsListResponse, tags=["teams"])
async def get_teams():
    """Get list of all AFL teams"""
    teams = [
        "Adelaide", "Brisbane Lions", "Carlton", "Collingwood", "Essendon",
        "Fremantle", "Geelong", "Gold Coast", "GWS", "Hawthorn",
        "Melbourne", "North Melbourne", "Port Adelaide", "Richmond",
        "St Kilda", "Sydney", "West Coast", "Western Bulldogs"
    ]
    
    team_infos = [
        TeamInfo(name=team, seasons_available=list(range(2010, 2026)))
        for team in teams
    ]
    
    return TeamsListResponse(
        teams=team_infos,
        total_teams=len(teams)
    )


@app.get("/historical/{year}", response_model=LadderResponse, tags=["data"])
async def get_historical_ladder(year: int):
    """
    Get historical ladder for a specific year
    
    Args:
        year: Year between 2015 and 2025
        
    Returns:
        LadderResponse with ladder data for that year
    """
    if year < 2015 or year > 2025:
        raise HTTPException(status_code=400, detail="Year must be between 2015 and 2025")
    
    try:
        # Loading match data
        data_path = Path(__file__).parent.parent / "data" / f"afl_{year}.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Data not found for year {year}")
        
        matches_df = pd.read_csv(data_path)
        
        # Calculating ladder
        ladder_df = calculate_ladder(matches_df)
        
        # Converting to response model
        ladder_stats = []
        for idx, row in ladder_df.iterrows():
            ladder_stats.append(TeamStats(
                team=row['Team'],
                position=idx + 1,
                wins=row['Wins'],
                draws=row['Draws'],
                losses=row['Losses'],
                points_for=row['Points For'],
                points_against=row['Points Against'],
                percentage=round(row['Percentage'], 2),
                premiership_points=row['Premiership Points']
            ))
        
        return LadderResponse(year=year, ladder=ladder_stats)
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error calculating ladder: {str(e)}")


@app.get("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_2026_ladder():
    """
    Predict 2026 AFL ladder using the best model
    
    Returns:
        PredictionResponse with predicted positions for all teams
    """
    if not model_data.get("loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Loading 2025 ladder (most recent season)
        data_path = Path(__file__).parent.parent / "data" / "afl_2025.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="2025 data not found")
        
        matches_df = pd.read_csv(data_path)
        ladder_2025 = calculate_ladder(matches_df)
        
        # Creating features for prediction
        feature_df = create_enhanced_features(matches_df, ladder_2025, 2025)
        
        # Getting model and features
        model = model_data["model"]
        feature_cols = model_data["features"]
        
        # Making predictions
        X = feature_df[feature_cols]
        predictions = model.predict(X)
        
        # Generating realistic W/L/D based on predicted positions
        # Using historical averages from 2015-2025 data
        def estimate_wins_from_position(position):
            """Estimate wins based on ladder position using historical averages"""
            # AFL typically has 23 games per season
            # Top teams win ~16-19 games, bottom teams win ~4-7 games
            position_to_wins = {
                1: 18, 2: 17, 3: 16, 4: 15, 5: 14, 6: 13,
                7: 12, 8: 12, 9: 11, 10: 10, 11: 9, 12: 9,
                13: 8, 14: 7, 15: 7, 16: 6, 17: 5, 18: 4
            }
            pos_int = max(1, min(18, round(position)))
            return position_to_wins.get(pos_int, 11)
        
        # Creating response
        prediction_teams = []
        for idx, row in feature_df.iterrows():
            team = row['Team']
            current_pos = int(row['Position'])
            pred_pos = float(predictions[idx])
            
            # Determine change arrow
            if pred_pos < current_pos - 0.5:
                change = "↑"
            elif pred_pos > current_pos + 0.5:
                change = "↓"
            else:
                change = "→"
            
            prediction_teams.append(PredictionTeam(
                team=team,
                predicted_position=round(pred_pos, 2),
                current_position=current_pos,
                change=change
            ))
        
        # Sorting by predicted position
        prediction_teams.sort(key=lambda x: x.predicted_position)
        
        return PredictionResponse(
            predictions=prediction_teams,
            model_name=model_data["model_name"],
            model_mae=model_data["test_mae"],
            timestamp=datetime.now()
        )
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@app.get("/predict/ladder", response_model=LadderResponse, tags=["prediction"])
async def predict_2026_full_ladder():
    """
    Predict full 2026 AFL ladder with W/L/D stats normalized to 23 games
    
    Returns:
        LadderResponse with predicted ladder including realistic W/L/D counts
    """
    if not model_data.get("loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Loading 2025 ladder (most recent season)
        data_path = Path(__file__).parent.parent / "data" / "afl_2025.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="2025 data not found")
        
        matches_df = pd.read_csv(data_path)
        ladder_2025 = calculate_ladder(matches_df)
        
        # Creating features for prediction
        feature_df = create_enhanced_features(matches_df, ladder_2025, 2025)
        
        # Getting model and features
        model = model_data["model"]
        feature_cols = model_data["features"]
        
        # Making predictions
        X = feature_df[feature_cols]
        predictions = model.predict(X)
        
        # Generating realistic W/L/D based on predicted positions
        # Using historical averages from 2015-2025 data
        def estimate_stats_from_position(position):
            """Estimate W/L/D and percentage based on ladder position"""
            # Historical averages for each position
            position_stats = {
                1: {'wins': 18, 'draws': 1, 'losses': 4, 'percentage': 136.0},
                2: {'wins': 17, 'draws': 0, 'losses': 6, 'percentage': 128.0},
                3: {'wins': 16, 'draws': 1, 'losses': 6, 'percentage': 122.0},
                4: {'wins': 15, 'draws': 1, 'losses': 7, 'percentage': 116.0},
                5: {'wins': 14, 'draws': 1, 'losses': 8, 'percentage': 112.0},
                6: {'wins': 13, 'draws': 1, 'losses': 9, 'percentage': 108.0},
                7: {'wins': 12, 'draws': 1, 'losses': 10, 'percentage': 104.0},
                8: {'wins': 12, 'draws': 0, 'losses': 11, 'percentage': 101.0},
                9: {'wins': 11, 'draws': 1, 'losses': 11, 'percentage': 98.0},
                10: {'wins': 10, 'draws': 1, 'losses': 12, 'percentage': 95.0},
                11: {'wins': 9, 'draws': 1, 'losses': 13, 'percentage': 92.0},
                12: {'wins': 9, 'draws': 0, 'losses': 14, 'percentage': 89.0},
                13: {'wins': 8, 'draws': 1, 'losses': 14, 'percentage': 86.0},
                14: {'wins': 7, 'draws': 1, 'losses': 15, 'percentage': 82.0},
                15: {'wins': 7, 'draws': 0, 'losses': 16, 'percentage': 78.0},
                16: {'wins': 6, 'draws': 0, 'losses': 17, 'percentage': 74.0},
                17: {'wins': 5, 'draws': 1, 'losses': 17, 'percentage': 69.0},
                18: {'wins': 4, 'draws': 0, 'losses': 19, 'percentage': 63.0}
            }
            pos_int = max(1, min(18, round(position)))
            return position_stats.get(pos_int, {'wins': 11, 'draws': 1, 'losses': 11, 'percentage': 100.0})
        
        # Creating predicted ladder with realistic stats
        ladder_teams = []
        for idx, row in feature_df.iterrows():
            team = row['Team']
            pred_pos = float(predictions[idx])
            
            # Get estimated stats for this predicted position
            stats = estimate_stats_from_position(pred_pos)
            wins = stats['wins']
            draws = stats['draws']
            losses = stats['losses']
            percentage = stats['percentage']
            points = (wins * 4) + (draws * 2)
            
            ladder_teams.append({
                'team': team,
                'predicted_position': pred_pos,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'percentage': percentage,
                'points': points
            })
        
        # Sort by predicted position
        ladder_teams.sort(key=lambda x: x['predicted_position'])
        
        # Convert to TeamStats objects
        ladder_stats = []
        for idx, team_data in enumerate(ladder_teams):
            ladder_stats.append(TeamStats(
                team=team_data['team'],
                position=idx + 1,
                wins=team_data['wins'],
                draws=team_data['draws'],
                losses=team_data['losses'],
                points_for=0,  # Not predicted
                points_against=0,  # Not predicted
                percentage=team_data['percentage'],
                premiership_points=team_data['points']
            ))
        
        return LadderResponse(year=2026, ladder=ladder_stats)
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@app.get("/compare", response_model=ModelComparisonResponse, tags=["comparison"])
async def compare_models():
    """
    Get model comparison results
    
    Returns:
        ModelComparisonResponse with metrics for all tested models
    """
    try:
        # Loading model comparison results
        results_path = Path(__file__).parent.parent / "results" / "model_comparison_results.csv"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Model comparison results not found")
        
        results_df = pd.read_csv(results_path)
        
        # Converting to response model
        model_metrics = []
        for _, row in results_df.iterrows():
            model_metrics.append(ModelMetrics(
                model_name=row['Model'],
                mae=row['CV MAE Mean'],
                cv_mae=row['CV MAE Mean'],
                cv_std=row['CV MAE Std'],
                test_mae=row.get('Test MAE', None)
            ))
        
        # Finding best model (lowest MAE)
        best_model = results_df.loc[results_df['CV MAE Mean'].idxmin(), 'Model']
        
        return ModelComparisonResponse(
            models=model_metrics,
            best_model=best_model,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error loading comparison: {str(e)}")


@app.get("/teams", response_model=TeamsListResponse, tags=["teams"])
async def get_teams():
    """
    Get list of all AFL teams with available data
    
    Returns:
        TeamsListResponse with all teams and their available seasons
    """
    try:
        data_path = Path(__file__).parent.parent / "data"
        teams_dict = {}
        
        # Scan all CSV files to find teams
        for year in range(2015, 2026):
            csv_path = data_path / f"afl_{year}.csv"
            if csv_path.exists():
                matches_df = pd.read_csv(csv_path)
                ladder_df = calculate_ladder(matches_df)
                
                for team in ladder_df['Team'].unique():
                    if team not in teams_dict:
                        teams_dict[team] = []
                    teams_dict[team].append(year)
        
        # Create response
        teams_list = [
            TeamInfo(name=team, seasons_available=sorted(years))
            for team, years in sorted(teams_dict.items())
        ]
        
        return TeamsListResponse(
            teams=teams_list,
            total_teams=len(teams_list),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching teams: {str(e)}")


@app.get("/teams/{team}/history", response_model=TeamHistoryResponse, tags=["teams"])
async def get_team_history(team: str):
    """
    Get historical performance for a specific team across all seasons
    
    Args:
        team: Team name (e.g., "Brisbane Lions", "Hawthorn")
        
    Returns:
        TeamHistoryResponse with team's performance across all available seasons
    """
    try:
        data_path = Path(__file__).parent.parent / "data"
        history = []
        
        # Collect data for all years
        for year in range(2015, 2026):
            csv_path = data_path / f"afl_{year}.csv"
            if csv_path.exists():
                matches_df = pd.read_csv(csv_path)
                ladder_df = calculate_ladder(matches_df)
                
                # Find team in ladder
                team_row = ladder_df[ladder_df['Team'] == team]
                if not team_row.empty:
                    row = team_row.iloc[0]
                    history.append(TeamHistoryEntry(
                        year=year,
                        position=ladder_df[ladder_df['Team'] == team].index[0] + 1,
                        wins=row['Wins'],
                        draws=row['Draws'],
                        losses=row['Losses'],
                        points_for=row['Points For'],
                        points_against=row['Points Against'],
                        percentage=round(row['Percentage'], 2),
                        premiership_points=row['Premiership Points']
                    ))
        
        if not history:
            raise HTTPException(status_code=404, detail=f"Team '{team}' not found")
        
        # Calculate stats
        positions = [h.position for h in history]
        
        return TeamHistoryResponse(
            team=team,
            history=history,
            seasons=len(history),
            best_position=min(positions),
            worst_position=max(positions),
            avg_position=round(sum(positions) / len(positions), 2),
            timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching team history: {str(e)}")


@app.get("/predict/team/{team}", tags=["prediction"])
async def predict_team(team: str):
    """
    Get 2026 prediction for a specific team
    
    Args:
        team: Team name (e.g., "Brisbane Lions", "Hawthorn")
        
    Returns:
        Prediction details for the specified team
    """
    if not model_data.get("loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get full predictions
        data_path = Path(__file__).parent.parent / "data" / "afl_2025.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="2025 data not found")
        
        matches_df = pd.read_csv(data_path)
        ladder_2025 = calculate_ladder(matches_df)
        feature_df = create_enhanced_features(matches_df, ladder_2025, 2025)
        
        # Find team
        team_row = feature_df[feature_df['Team'] == team]
        if team_row.empty:
            raise HTTPException(status_code=404, detail=f"Team '{team}' not found")
        
        # Make prediction
        model = model_data["model"]
        feature_cols = model_data["features"]
        X = team_row[feature_cols]
        prediction = model.predict(X)[0]
        
        current_pos = int(team_row['Position'].iloc[0])
        
        # Determine change
        if prediction < current_pos - 0.5:
            change = "↑"
        elif prediction > current_pos + 0.5:
            change = "↓"
        else:
            change = "→"
        
        return {
            "team": team,
            "current_position": current_pos,
            "predicted_position": round(prediction, 2),
            "change": change,
            "change_value": round(current_pos - prediction, 2),
            "model_name": model_data["model_name"],
            "model_mae": model_data["test_mae"],
            "timestamp": datetime.now()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@app.get("/matches/{year}", response_model=MatchesResponse, tags=["data"])
async def get_matches(year: int):
    """
    Get all matches for a specific year
    
    Args:
        year: Year between 2015 and 2025
        
    Returns:
        MatchesResponse with all matches for that year
    """
    if year < 2015 or year > 2025:
        raise HTTPException(status_code=400, detail="Year must be between 2015 and 2025")
    
    try:
        data_path = Path(__file__).parent.parent / "data" / f"afl_{year}.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Data not found for year {year}")
        
        matches_df = pd.read_csv(data_path)
        
        # Parse matches
        matches = []
        for _, row in matches_df.iterrows():
            winner = None
            if row['Home Score'] > row['Away Score']:
                winner = row['Home Team']
            elif row['Away Score'] > row['Home Score']:
                winner = row['Away Team']
            else:
                winner = "Draw"
            
            matches.append(Match(
                date=row['Date'],
                home_team=row['Home Team'],
                away_team=row['Away Team'],
                home_score=int(row['Home Score']),
                away_score=int(row['Away Score']),
                venue=row.get('Venue', None),
                winner=winner
            ))
        
        return MatchesResponse(
            year=year,
            matches=matches,
            total_matches=len(matches),
            timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching matches: {str(e)}")


@app.get("/teams/{team}/matches/{year}", response_model=MatchesResponse, tags=["teams"])
async def get_team_matches(team: str, year: int):
    """
    Get all matches for a specific team in a specific year
    
    Args:
        team: Team name
        year: Year between 2015 and 2025
        
    Returns:
        MatchesResponse with all matches for that team in that year
    """
    if year < 2015 or year > 2025:
        raise HTTPException(status_code=400, detail="Year must be between 2015 and 2025")
    
    try:
        data_path = Path(__file__).parent.parent / "data" / f"afl_{year}.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Data not found for year {year}")
        
        matches_df = pd.read_csv(data_path)
        
        # Filter for team
        team_matches = matches_df[
            (matches_df['Home Team'] == team) | (matches_df['Away Team'] == team)
        ]
        
        if team_matches.empty:
            raise HTTPException(status_code=404, detail=f"No matches found for team '{team}' in {year}")
        
        # Parse matches
        matches = []
        for _, row in team_matches.iterrows():
            winner = None
            if row['Home Score'] > row['Away Score']:
                winner = row['Home Team']
            elif row['Away Score'] > row['Home Score']:
                winner = row['Away Team']
            else:
                winner = "Draw"
            
            matches.append(Match(
                date=row['Date'],
                home_team=row['Home Team'],
                away_team=row['Away Team'],
                home_score=int(row['Home Score']),
                away_score=int(row['Away Score']),
                venue=row.get('Venue', None),
                winner=winner
            ))
        
        return MatchesResponse(
            year=year,
            team=team,
            matches=matches,
            total_matches=len(matches),
            timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching team matches: {str(e)}")


@app.get("/features", response_model=FeaturesResponse, tags=["model"])
async def get_features():
    """
    Get model features and their importance
    
    Returns:
        FeaturesResponse with all features ranked by importance
    """
    if not model_data.get("loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model = model_data["model"]
        feature_cols = model_data["features"]
        
        # Get feature importance
        importances = model.feature_importances_
        
        # Create ranked list
        features = []
        for rank, (feature, importance) in enumerate(
            sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True),
            start=1
        ):
            features.append(FeatureImportance(
                feature=feature,
                importance=round(float(importance), 4),
                rank=rank
            ))
        
        return FeaturesResponse(
            features=features,
            total_features=len(features),
            model_name=model_data["model_name"],
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching features: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
