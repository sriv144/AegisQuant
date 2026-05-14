"""
AegisQuant Web Dashboard — FastAPI Backend
===========================================
Features:
  - WebSocket real-time push (/ws/live) with HTTP polling fallback
  - Market-aware benchmark overlay (/api/benchmark) — S&P 500 (US) or Nifty 50 (India)
  - JWT authentication (/api/auth/login, token-gated routes)
  - Per-trade P&L lifecycle (/api/positions/detailed, /api/trades/closed)
  - Trade reasoning drill-down (/api/decisions/{id}/reasoning)
  - Market configuration via MARKET env var (US or IN)
"""

import os
import json
import asyncio
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, text
import pandas as pd

# ── Market configuration ────────────────────────────────────────────────────
MARKET = os.getenv("MARKET", "US").upper()
BENCHMARK_SYMBOL = "^GSPC" if MARKET == "US" else "^NSEI"
BENCHMARK_LABEL = "S&P 500" if MARKET == "US" else "Nifty 50"
CURRENCY_SYMBOL = "$" if MARKET == "US" else "₹"
DEFAULT_CAPITAL = 100_000.0 if MARKET == "US" else DEFAULT_CAPITAL

logger = logging.getLogger(__name__)

# ── JWT-like token management (lightweight, no external deps) ─────────────────
# Set AEGIS_PASSWORD env var to enable auth; if unset, auth is disabled (dev mode)
_AUTH_PASSWORD = os.getenv("AEGIS_PASSWORD", "")
_TOKEN_SECRET = os.getenv("AEGIS_TOKEN_SECRET", secrets.token_hex(32))
_TOKEN_TTL_HOURS = 24
_active_tokens: dict = {}  # token -> expiry datetime

security = HTTPBearer(auto_error=False)


def _hash_password(pw: str) -> str:
    return hashlib.sha256((pw + _TOKEN_SECRET[:16]).encode()).hexdigest()


def _create_token() -> str:
    token = secrets.token_urlsafe(48)
    _active_tokens[token] = datetime.utcnow() + timedelta(hours=_TOKEN_TTL_HOURS)
    # Prune expired tokens
    now = datetime.utcnow()
    expired = [t for t, exp in _active_tokens.items() if exp < now]
    for t in expired:
        _active_tokens.pop(t, None)
    return token


def _verify_token(token: str) -> bool:
    exp = _active_tokens.get(token)
    if not exp:
        return False
    if datetime.utcnow() > exp:
        _active_tokens.pop(token, None)
        return False
    return True


async def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Dependency: enforces auth if AEGIS_PASSWORD is set, otherwise passes through."""
    if not _AUTH_PASSWORD:
        return True  # Auth disabled in dev mode
    if not credentials or not _verify_token(credentials.credentials):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return True


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="AegisQuant Web UI")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

DB_PATH = Path("aegisquant_live.db")

_engine_cache = None


def get_engine():
    global _engine_cache
    if _engine_cache is None:
        db_url = os.getenv("POSTGRES_URL", f"sqlite:///{DB_PATH}")
        _engine_cache = create_engine(db_url)
    return _engine_cache


# Mount static AFTER API routes are defined (order matters for catch-all)
# We'll do it at the bottom of this file.


# ── Auth endpoints ───────────────────────────────────────────────────────────
@app.post("/api/auth/login")
def login(body: dict):
    """Authenticate with password. Returns JWT-like bearer token."""
    if not _AUTH_PASSWORD:
        return {"token": "dev-mode", "expires_in": 999999, "auth_enabled": False}

    password = body.get("password", "")
    if _hash_password(password) != _hash_password(_AUTH_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid password")

    token = _create_token()
    return {"token": token, "expires_in": _TOKEN_TTL_HOURS * 3600, "auth_enabled": True}


@app.get("/health")
def health():
    """Liveness probe for Docker / load balancers."""
    return {"status": "ok", "market": MARKET}


@app.get("/api/auth/status")
def auth_status():
    """Check if auth is enabled (so frontend knows whether to show login screen)."""
    return {"auth_enabled": bool(_AUTH_PASSWORD)}


@app.get("/api/market-config")
def market_config():
    """Return market configuration for frontend rendering."""
    return {
        "market": MARKET,
        "currency_symbol": CURRENCY_SYMBOL,
        "currency_code": "USD" if MARKET == "US" else "INR",
        "locale": "en-US" if MARKET == "US" else "en-IN",
        "timezone": "America/New_York" if MARKET == "US" else "Asia/Kolkata",
        "benchmark_label": BENCHMARK_LABEL,
        "default_capital": DEFAULT_CAPITAL,
    }


# ── WebSocket real-time push ─────────────────────────────────────────────────
class ConnectionManager:
    """Manages active WebSocket connections and broadcasts updates."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws) if ws in self.active else None

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()


@app.websocket("/ws/live")
async def websocket_endpoint(ws: WebSocket):
    """
    Real-time push: sends portfolio snapshot every 5 seconds to connected clients.
    Falls back gracefully — frontend uses HTTP polling if WS fails.
    """
    # Optional: check auth token in query params
    token = ws.query_params.get("token", "")
    if _AUTH_PASSWORD and not _verify_token(token) and token != "dev-mode":
        await ws.close(code=4001)
        return

    await ws_manager.connect(ws)
    try:
        while True:
            # Build live snapshot
            snapshot = _build_live_snapshot()
            await ws.send_json(snapshot)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)


def _build_live_snapshot() -> dict:
    """Build the real-time data packet sent over WebSocket."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Portfolio value
            pv_row = conn.execute(text(
                "SELECT date, total_portfolio_value, drawdown, total_pnl "
                "FROM daily_pnl ORDER BY date DESC LIMIT 1"
            )).fetchone()

            # Open positions count
            pos_row = conn.execute(text(
                "SELECT COUNT(*) FROM open_positions WHERE status = 'OPEN'"
            )).fetchone()

            # Latest decision timestamp
            dec_row = conn.execute(text(
                "SELECT timestamp, circuit_breaker_status FROM decisions ORDER BY id DESC LIMIT 1"
            )).fetchone()

        pv = float(pv_row[1]) if pv_row else DEFAULT_CAPITAL
        dd = float(pv_row[2]) if pv_row else 0.0
        pnl = float(pv_row[3]) if pv_row else 0.0

        return {
            "type": "snapshot",
            "ts": datetime.utcnow().isoformat(),
            "portfolio_value": pv,
            "drawdown": dd,
            "total_pnl": pnl,
            "open_positions": pos_row[0] if pos_row else 0,
            "last_decision_ts": dec_row[0] if dec_row else None,
            "circuit_breaker": dec_row[1] if dec_row else "OK",
        }
    except Exception as e:
        return {"type": "error", "message": str(e)}


# ── Portfolio API ────────────────────────────────────────────────────────────
@app.get("/api/portfolio")
def get_portfolio(_auth=Depends(require_auth)):
    """Return historical portfolio values and current metrics."""
    try:
        engine = get_engine()
        query = text(
            "SELECT date, total_portfolio_value, drawdown, total_pnl "
            "FROM daily_pnl ORDER BY date ASC LIMIT 365"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            return {"history": [], "current_value": 0.0, "drawdown": 0.0, "total_pnl": 0.0}

        latest = df.iloc[-1]
        return {
            "history": df.to_dict(orient="records"),
            "current_value": float(latest["total_portfolio_value"]),
            "drawdown": float(latest["drawdown"]),
            "total_pnl": float(latest["total_pnl"]),
        }
    except Exception as e:
        logger.error(f"get_portfolio: {e}")
        return {"error": str(e), "history": [], "current_value": 0.0}


# ── Benchmark API ────────────────────────────────────────────────────────────
@app.get("/api/benchmark")
def get_benchmark(_auth=Depends(require_auth)):
    """
    Return benchmark normalized performance to overlay on portfolio chart.
    US: S&P 500 (^GSPC), India: Nifty 50 (^NSEI).
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            dates = conn.execute(text(
                "SELECT MIN(date), MAX(date) FROM daily_pnl"
            )).fetchone()

        if not dates or not dates[0]:
            return {"benchmark": [], "label": BENCHMARK_LABEL}

        start_date, end_date = dates
        try:
            from datetime import datetime as dt
            start = (dt.fromisoformat(start_date) - timedelta(days=5)).strftime("%Y-%m-%d")
            end = (dt.fromisoformat(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            start, end = start_date, end_date

        import yfinance as yf
        bench_data = yf.download(BENCHMARK_SYMBOL, start=start, end=end, auto_adjust=True, progress=False)
        if bench_data.empty:
            return {"benchmark": [], "label": BENCHMARK_LABEL}

        close = bench_data["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]

        with engine.connect() as conn:
            init_row = conn.execute(text(
                "SELECT total_portfolio_value FROM daily_pnl ORDER BY date ASC LIMIT 1"
            )).fetchone()
        initial_pv = float(init_row[0]) if init_row else DEFAULT_CAPITAL

        first_close = float(close.iloc[0])
        records = []
        for date, val in close.items():
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10]
            normalized = initial_pv * (float(val) / first_close)
            records.append({"date": date_str, "value": round(normalized, 2)})

        return {"benchmark": records, "label": f"{BENCHMARK_LABEL} (normalized)"}
    except Exception as e:
        logger.error(f"get_benchmark: {e}")
        return {"benchmark": [], "label": BENCHMARK_LABEL, "error": str(e)}


# ── Positions API ────────────────────────────────────────────────────────────
@app.get("/api/positions")
def get_positions(_auth=Depends(require_auth)):
    """Return active internal positions."""
    try:
        engine = get_engine()
        query = text(
            "SELECT ticker, quantity, entry_price, pnl_pct, trade_type "
            "FROM open_positions WHERE status='OPEN'"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df.to_dict(orient="records") if not df.empty else []
    except Exception as e:
        logger.error(f"get_positions: {e}")
        return []


@app.get("/api/positions/detailed")
def get_positions_detailed(_auth=Depends(require_auth)):
    """Open positions with entry context, P&L, and holding duration."""
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
        logger.error(f"get_positions_detailed: {e}")
        return []


@app.get("/api/trades/closed")
def get_closed_trades(_auth=Depends(require_auth)):
    """Recently closed trades with realized P&L and exit reasoning."""
    try:
        engine = get_engine()
        query = text("""
            SELECT ticker, entry_price, exit_price, entry_date, exit_date,
                   quantity, trade_type, strategy, pnl_pct, exit_reason, sector
            FROM open_positions
            WHERE status = 'CLOSED' AND exit_date IS NOT NULL
            ORDER BY exit_date DESC LIMIT 100
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
                "realized_pnl": round((exit_p - entry_p) * qty, 2),
                "exit_reason": row["exit_reason"],
                "sector": row["sector"],
            })
        return trades
    except Exception as e:
        logger.error(f"get_closed_trades: {e}")
        return []


# ── Decisions API ────────────────────────────────────────────────────────────
@app.get("/api/decisions")
def get_decisions(_auth=Depends(require_auth)):
    """Return recent AI decisions with trade reasoning."""
    try:
        engine = get_engine()
        query = text("""
            SELECT id, timestamp, model_version, ticker_universe, final_weights,
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
        logger.error(f"get_decisions: {e}")
        return []


@app.get("/api/latest-run")
def get_latest_run(_auth=Depends(require_auth)):
    """
    Ticker-level breakdown from the most recent decision with per-position reasoning.
    """
    try:
        engine = get_engine()
        query = text("""
            SELECT timestamp, ticker_universe, rl_output, final_weights,
                   circuit_breaker_status, model_version, trade_reasoning
            FROM decisions ORDER BY id DESC LIMIT 1
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
        portfolio_value = float(pv_row[0]) if pv_row else DEFAULT_CAPITAL

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
            capital_alloc = abs(w) * portfolio_value
            price = latest_prices.get(ticker)
            est_shares = int(capital_alloc / price) if price and price > 0 else None
            positions.append({
                "ticker": ticker,
                "weight_pct": round(w * 100, 2),
                "direction": direction,
                "capital": round(capital_alloc, 0),
                "rupees": round(capital_alloc, 0),  # backward compat
                "last_price": round(price, 2) if price else None,
                "est_shares": est_shares,
                "reasoning": reasoning.get(ticker, {}),
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
        logger.error(f"get_latest_run: {e}")
        return {"error": str(e), "positions": [], "summary": {}}


@app.get("/api/decisions/{decision_id}/reasoning")
def get_decision_reasoning(decision_id: int, _auth=Depends(require_auth)):
    """Full trade reasoning for a specific decision."""
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
        logger.error(f"get_decision_reasoning: {e}")
        return {"error": str(e), "tickers": []}


# ── Static files & root ─────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.webapp.server:app", host="127.0.0.1", port=8000, reload=True)
