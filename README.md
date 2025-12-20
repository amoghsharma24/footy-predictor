# AFL Ladder Predictor 🏈

Machine learning-powered AFL ladder predictions with a production-ready REST API and modern web interface.

## 🏗️ Project Structure

```
footy-predictor/
├── api/                    # FastAPI REST API
│   ├── main.py            # API server with 11 endpoints
│   └── models.py          # Pydantic request/response models
├── ml/                     # Machine Learning pipeline
│   ├── ladder_calc.py     # AFL ladder calculation logic
│   ├── preprocessing.py   # Basic feature engineering
│   ├── advanced_features.py    # Advanced feature engineering
│   ├── enhanced_features.py    # Enhanced features (29 total)
│   ├── predict.py         # Main prediction script
│   └── model_comparison.py     # Model evaluation framework
├── scripts/                # Utility scripts
│   └── scraper.py         # Web scraper for AFL data
├── data/                   # AFL match data (2015-2025)
├── results/                # Trained models & predictions
└── tests/                  # Test scripts

## 🚀 Quick Start

### Backend API
```bash
# Activate virtual environment
.venv\Scripts\activate

# Start API server
python -m uvicorn api.main:app --reload

# Visit http://127.0.0.1:8000/docs for API docs
```

## 📊 API Endpoints

- `GET /health` - Health check
- `GET /predict` - Predict 2026 ladder
- `GET /historical/{year}` - Get ladder for any year (2015-2025)
- `GET /compare` - Compare model performance
- `GET /teams` - List all AFL teams
- `GET /teams/{team}/history` - Team historical performance
- `GET /predict/team/{team}` - Single team prediction
- `GET /matches/{year}` - All matches for a year
- `GET /teams/{team}/matches/{year}` - Team-specific matches
- `GET /features` - Model feature importance

## 🤖 Model Performance

**Best Model:** Random Forest Regressor
- **MAE:** 3.655 ladder positions
- **Features:** 29 engineered features
- **Training Data:** 2015-2025 seasons

### Key Features:
- Percentage (22% importance)
- Average opponent position (9.6%)
- Rolling 10-game margin (5.1%)
- Win streaks, home/away splits, momentum metrics

## 🛠️ Tech Stack

**Backend:**
- FastAPI (REST API)
- scikit-learn (ML models)
- pandas, numpy (data processing)
- uvicorn (ASGI server)

**Data:**
- Web scraping with BeautifulSoup
- Historical AFL data (2015-2025)
- 216 matches per season

## 📈 Model Development

1. **Data Collection** - Scraped 11 years of AFL data
2. **Feature Engineering** - 3 iterations (basic → advanced → enhanced)
3. **Model Selection** - Tested Random Forest, XGBoost, Gradient Boosting
4. **Validation** - 5-fold cross-validation + test set

## 🎯 2026 Predictions

Top predictions:
1. Hawthorn (↑ from 5th)
2. GWS (↑ from 7th)
3. Sydney (↓ from 3rd)
...
8. Brisbane Lions (↓ from 1st - defending champions)

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use pip-tools for reproducible builds
pip-compile requirements.in
pip-sync requirements.txt
```

## 🏃 Running Scripts

```bash
# Scrape new season data
python scripts/scraper.py

# Calculate ladder
python ml/ladder_calc.py

# Train model
python ml/predict.py

# Compare models
python ml/model_comparison.py
```

## 🎨 Frontend

*Coming soon* - React-based web interface

## 📄 License

MIT