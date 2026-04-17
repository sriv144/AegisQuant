import os
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, text
import pandas as pd

app = FastAPI(title="AegisQuant Web UI")

# Check if static directory exists
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DB_PATH = Path("aegisquant_live.db")

def get_engine():
    db_url = os.getenv("POSTGRES_URL", f"sqlite:///{DB_PATH}")
    return create_engine(db_url)

@app.get("/")
def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/api/portfolio")
def get_portfolio():
    """Return historical portfolio values and current metrics."""
    try:
        engine = get_engine()
        query = text("SELECT date, total_portfolio_value, drawdown, total_pnl FROM daily_pnl ORDER BY date ASC LIMIT 100")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        # If empty, return mock data to prevent errors
        if df.empty:
            return {
                "history": [],
                "current_value": 0.0,
                "drawdown": 0.0,
                "total_pnl": 0.0
            }
            
        latest = df.iloc[-1]
        
        return {
            "history": df.to_dict(orient="records"),
            "current_value": float(latest["total_portfolio_value"]),
            "drawdown": float(latest["drawdown"]),
            "total_pnl": float(latest["total_pnl"])
        }
    except Exception as e:
        print(e)
        return {"error": str(e), "history": [], "current_value": 0.0}

@app.get("/api/positions")
def get_positions():
    """Return active internal positions."""
    try:
        engine = get_engine()
        query = text("SELECT ticker, quantity, entry_price, pnl_pct, trade_type FROM open_positions WHERE status='OPEN'")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        if df.empty:
            return []
            
        return df.to_dict(orient="records")
    except Exception as e:
        print(e)
        return []

@app.get("/api/decisions")
def get_decisions():
    """Return recent AI decisions."""
    try:
        engine = get_engine()
        query = text("SELECT timestamp, model_version, final_weights, transaction_costs, circuit_breaker_status FROM decisions ORDER BY id DESC LIMIT 50")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        if df.empty:
            return []
            
        # Parse weights safely
        def safe_json_load(x):
            try:
                return json.loads(x)
            except:
                return []
                
        df['final_weights'] = df['final_weights'].apply(safe_json_load)
        return df.to_dict(orient="records")
    except Exception as e:
        print(e)
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.webapp.server:app", host="127.0.0.1", port=8000, reload=True)
