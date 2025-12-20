"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class TeamStats(BaseModel):
    """Team statistics for a given season"""
    team: str
    position: int
    wins: int
    draws: int
    losses: int
    points_for: int
    points_against: int
    percentage: float
    premiership_points: int


class LadderResponse(BaseModel):
    """Historical ladder response"""
    year: int
    ladder: List[TeamStats]
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionTeam(BaseModel):
    """Predicted team position"""
    team: str
    predicted_position: float
    current_position: int
    change: str  # "↑", "↓", or "→"


class PredictionResponse(BaseModel):
    """Prediction response for 2026 ladder"""
    year: int = 2026
    predictions: List[PredictionTeam]
    model_name: str
    model_mae: float
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_name: str
    mae: float
    cv_mae: float
    cv_std: float
    test_mae: Optional[float] = None


class ModelComparisonResponse(BaseModel):
    """Model comparison response"""
    models: List[ModelMetrics]
    best_model: str
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class TeamInfo(BaseModel):
    """Basic team information"""
    name: str
    seasons_available: List[int]


class TeamsListResponse(BaseModel):
    """List of all teams"""
    teams: List[TeamInfo]
    total_teams: int
    timestamp: datetime = Field(default_factory=datetime.now)


class TeamHistoryEntry(BaseModel):
    """Team's performance in a single season"""
    year: int
    position: int
    wins: int
    draws: int
    losses: int
    points_for: int
    points_against: int
    percentage: float
    premiership_points: int


class TeamHistoryResponse(BaseModel):
    """Historical performance for a team"""
    team: str
    history: List[TeamHistoryEntry]
    seasons: int
    best_position: int
    worst_position: int
    avg_position: float
    timestamp: datetime = Field(default_factory=datetime.now)


class Match(BaseModel):
    """Single match information"""
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    venue: Optional[str] = None
    winner: Optional[str] = None


class MatchesResponse(BaseModel):
    """List of matches"""
    year: Optional[int] = None
    team: Optional[str] = None
    matches: List[Match]
    total_matches: int
    timestamp: datetime = Field(default_factory=datetime.now)


class FeatureImportance(BaseModel):
    """Feature importance from the model"""
    feature: str
    importance: float
    rank: int


class FeaturesResponse(BaseModel):
    """Model features and their importance"""
    features: List[FeatureImportance]
    total_features: int
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
