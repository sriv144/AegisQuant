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
    """Return recent AI decisions with trade reasoning."""
    try:
        engine = get_engine()
        query = text("""
            SELECT timestamp, model_version, ticker_universe, final_weights,
                   transaction_costs, circuit_breaker_status, trade_reasoning
            FROM decisions ORDER BY id DESC LIMIT 50
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            return []

        def safe_json_load(x):
            try:
                return json.loads(x) if x else []
            except Exception:
                return []

        df['final_weights'] = df['final_weights'].apply(safe_json_load)
        df['ticker_universe'] = df['ticker_universe'].apply(safe_json_load)
        df['trade_reasoning'] = df['trade_reasoning'].apply(
            lambda x: json.loads(x) if x and x != "{}" else {}
        )
        return df.to_dict(orient="records")
    except Exception as e:
        print(e)
        return []


@app.get("/api/latest-run")
def get_latest_run():
    """
    Returns the ticker-level breakdown from the most recent decision:
    which tickers were longed, shorted, or skipped, at what weight,
    and WHY (agent reasoning from the pipeline).
    """
    try:
        engine = get_engine()
        query = text("""
            SELECT timestamp, ticker_universe, rl_output, final_weights,
                   circuit_breaker_status, model_version, trade_reasoning
            FROM decisions
            ORDER BY id DESC
            LIMIT 1
        """)
        with engine.connect() as conn:
            row = conn.execute(query).fetchone()

        if not row:
            return {"positions": [], "summary": {}}

        timestamp, tickers_raw, rl_raw, weights_raw, cb_status, model, reasoning_raw = row
        tickers = json.loads(tickers_raw or "[]")
        weights = json.loads(weights_raw or "[]")
        try:
            reasoning = json.loads(reasoning_raw) if reasoning_raw else {}
        except Exception:
            reasoning = {}

        pv_query = text("SELECT total_portfolio_value FROM daily_pnl ORDER BY date DESC LIMIT 1")
        with engine.connect() as conn:
            pv_row = conn.execute(pv_query).fetchone()
        portfolio_value = float(pv_row[0]) if pv_row else 250_000.0

        latest_prices = {}
        try:
            import yfinance as yf
            active = [t for t, w in zip(tickers, weights) if abs(w) >= 0.001]
            if active:
                data = yf.download(active, period="1d", auto_adjust=True, progress=False)
                close = data["Close"] if "Close" in data else pd.DataFrame()
                if not close.empty:
                    for t in active:
                        try:
                            latest_prices[t] = float(close[t].dropna().iloc[-1])
                        except Exception:
                            pass
        except Exception:
            pass

        positions = []
        for ticker, w in zip(tickers, weights):
            if abs(w) < 0.001:
                continue
            direction = "LONG" if w > 0 else "SHORT"
            rupees = abs(w) * portfolio_value
            price = latest_prices.get(ticker)
            est_shares = int(rupees / price) if price and price > 0 else None
            ticker_reasoning = reasoning.get(ticker, {})
            positions.append({
                "ticker": ticker,
                "weight_pct": round(w * 100, 2),
                "direction": direction,
                "rupees": round(rupees, 0),
                "last_price": round(price, 2) if price else None,
                "est_shares": est_shares,
                "reasoning": ticker_reasoning,
            })

        positions.sort(key=lambda x: -abs(x["weight_pct"]))

        longs = [p for p in positions if p["direction"] == "LONG"]
        shorts = [p for p in positions if p["direction"] == "SHORT"]
        gross = sum(abs(w) for w in weights)
        net = sum(weights)

        return {
            "timestamp": timestamp,
            "model_version": model,
            "circuit_breaker": cb_status,
            "universe_size": len(tickers),
            "positions": positions,
            "summary": {
                "long_count": len(longs),
                "short_count": len(shorts),
                "gross_exposure_pct": round(gross * 100, 1),
                "net_exposure_pct": round(net * 100, 1),
                "cash_pct": round((1 - gross) * 100, 1),
                "portfolio_value": portfolio_value,
            },
        }
    except Exception as e:
        print(e)
        return {"error": str(e), "positions": [], "summary": {}}

@app.get("/api/decisions/{decision_id}/reasoning")
def get_decision_reasoning(decision_id: int):
    """
    Return full trade reasoning for a specific decision.
    Each ticker gets: research signals (per-agent), committee verdict,
    allocation sizing rationale, and risk officer approval/rejection.
    """
    try:
        engine = get_engine()
        query = text("""
            SELECT timestamp, ticker_universe, final_weights, trade_reasoning,
                   model_version, circuit_breaker_status
            FROM decisions WHERE id = :did
        """)
        with engine.connect() as conn:
            row = conn.execute(query, {"did": decision_id}).fetchone()

        if not row:
            return {"error": "Decision not found", "tickers": []}

        timestamp, tickers_raw, weights_raw, reasoning_raw, model, cb = row
        tickers = json.loads(tickers_raw or "[]")
        weights = json.loads(weights_raw or "[]")
        try:
            reasoning = json.loads(reasoning_raw) if reasoning_raw else {}
        except Exception:
            reasoning = {}

        ticker_details = []
        for ticker, w in zip(tickers, weights):
            r = reasoning.get(ticker, {})
            ticker_details.append({
                "ticker": ticker,
                "weight_pct": round(w * 100, 2),
                "direction": "LONG" if w > 0 else ("SHORT" if w < 0 else "FLAT"),
                "trade_type": r.get("trade_type", "SKIP"),
                "research_signals": r.get("research_signals", []),
                "committee": r.get("committee", {}),
                "allocation": r.get("allocation", {}),
                "risk": r.get("risk", {}),
            })

        return {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "model_version": model,
            "circuit_breaker": cb,
            "tickers": ticker_details,
        }
    except Exception as e:
        print(e)
        return {"error": str(e), "tickers": []}


@app.get("/api/positions/detailed")
def get_positions_detailed():
    """
    Return open positions with entry reasoning, P&L, and holding duration.
    Combines open_positions table with the latest decision reasoning.
    """
    try:
        engine = get_engine()
        query = text("""
            SELECT ticker, quantity, entry_price, entry_date, trade_type,
                   strategy, stop_loss_pct, take_profit_pct, max_hold_days, sector
            FROM open_positions WHERE status = 'OPEN'
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            return []

        from datetime import datetime
        today = datetime.now()

        positions = []
        for _, row in df.iterrows():
            entry_date = row["entry_date"]
            try:
                days_held = (today - datetime.fromisoformat(entry_date)).days
            except Exception:
                days_held = 0

            positions.append({
                "ticker": row["ticker"],
                "quantity": int(row["quantity"]),
                "entry_price": float(row["entry_price"]),
                "entry_date": entry_date,
                "days_held": days_held,
                "trade_type": row["trade_type"],
                "strategy": row["strategy"],
                "stop_loss_pct": float(row["stop_loss_pct"]),
                "take_profit_pct": float(row["take_profit_pct"]),
                "max_hold_days": int(row["max_hold_days"]),
                "sector": row["sector"],
            })

        return positions
    except Exception as e:
        print(e)
        return []


@app.get("/api/trades/closed")
def get_closed_trades():
    """Return recently closed trades with P&L and exit reasoning."""
    try:
        engine = get_engine()
        query = text("""
            SELECT ticker, entry_price, exit_price, entry_date, exit_date,
                   quantity, trade_type, strategy, pnl_pct, exit_reason, sector
            FROM open_positions
            WHERE status = 'CLOSED' AND exit_date IS NOT NULL
            ORDER BY exit_date DESC
            LIMIT 100
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            return []

        trades = []
        for _, row in df.iterrows():
            entry_p = float(row["entry_price"])
            exit_p = float(row["exit_price"]) if row["exit_price"] else entry_p
            qty = int(row["quantity"])
            realized_pnl = (exit_p - entry_p) * qty

            trades.append({
                "ticker": row["ticker"],
                "entry_price": entry_p,
                "exit_price": exit_p,
                "entry_date": row["entry_date"],
                "exit_date": row["exit_date"],
                "quantity": qty,
                "trade_type": row["trade_type"],
                "strategy": row["strategy"],
                "pnl_pct": float(row["pnl_pct"]) if row["pnl_pct"] else 0.0,
                "realized_pnl": round(realized_pnl, 2),
                "exit_reason": row["exit_reason"],
                "sector": row["sector"],
            })

        return trades
    except Exception as e:
        print(e)
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.webapp.server:app", host="127.0.0.1", port=8000, reload=True)
