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

# Load .env before anything else so ALPACA_API_KEY etc. are available when the server starts.
# Works whether launched with `uvicorn` directly or via `python -m src.webapp.server`.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set externally

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
DEFAULT_CAPITAL = 100_000.0 if MARKET == "US" else 1_000_000.0

logger = logging.getLogger(__name__)

# ── JWT-like token management (lightweight, no external deps) ─────────────────
# Set AEGIS_PASSWORD env var to enable auth; if unset, auth is disabled (dev mode)
_AUTH_PASSWORD = os.getenv("AEGIS_PASSWORD", "")
_API_KEY = os.getenv("AEGIS_API_KEY", "")
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
    if _API_KEY:
        if not credentials or credentials.credentials != _API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        return True
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
_engine_cache_url = None


def get_engine():
    global _engine_cache, _engine_cache_url
    db_url = os.getenv("POSTGRES_URL", f"sqlite:///{DB_PATH}")
    if _engine_cache is None or _engine_cache_url != db_url:
        _engine_cache = create_engine(db_url)
        _engine_cache_url = db_url
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
        "kill_switch": os.getenv("KILL_SWITCH", "false").lower() in ("true", "1", "yes"),
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

            # Open positions count — also check live Alpaca if DB is empty
            pos_row = conn.execute(text(
                "SELECT COUNT(*) FROM open_positions WHERE status = 'OPEN'"
            )).fetchone()

            # Latest decision timestamp
            dec_row = conn.execute(text(
                "SELECT timestamp, circuit_breaker_status FROM decisions ORDER BY id DESC LIMIT 1"
            )).fetchone()

        # Use DB values if they're real pipeline output; otherwise fall back to live Alpaca.
        # A row is a placeholder if: missing, exactly at default capital, or far outside
        # the expected range for this market (e.g. India ₹250K rows in a US $100K account).
        db_pv = float(pv_row[1]) if pv_row else None
        db_is_placeholder = (
            db_pv is None
            or db_pv == DEFAULT_CAPITAL
            or abs(db_pv - DEFAULT_CAPITAL) / DEFAULT_CAPITAL > 0.5
        )
        live = _alpaca_portfolio_live()
        if db_is_placeholder or not live:
            pv = live.get("current_value", DEFAULT_CAPITAL) if live else (db_pv or DEFAULT_CAPITAL)
            dd = live.get("drawdown", 0.0) if live else (float(pv_row[2]) if pv_row else 0.0)
            pnl = live.get("total_pnl", 0.0) if live else (float(pv_row[3]) if pv_row else 0.0)
        else:
            # DB has real data — still prefer live Alpaca equity for freshness
            pv = live.get("current_value", db_pv)
            dd = live.get("drawdown", float(pv_row[2]))
            pnl = live.get("total_pnl", float(pv_row[3]))

        # Open positions: prefer DB count; fall back to live Alpaca count
        open_count = pos_row[0] if pos_row and pos_row[0] > 0 else 0
        if open_count == 0:
            try:
                live_positions = _alpaca_positions_live()
                open_count = len(live_positions)
            except Exception:
                pass

        return {
            "type": "snapshot",
            "ts": datetime.utcnow().isoformat(),
            "portfolio_value": pv,
            "drawdown": dd,
            "total_pnl": pnl,
            "open_positions": open_count,
            "last_decision_ts": dec_row[0] if dec_row else None,
            "circuit_breaker": dec_row[1] if dec_row else "OK",
        }
    except Exception as e:
        return {"type": "error", "message": str(e)}


# ── Portfolio API ────────────────────────────────────────────────────────────
def _alpaca_portfolio_live() -> dict:
    """Fetch live equity / P&L from Alpaca when daily_pnl DB table is empty."""
    try:
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not api_key or not secret_key:
            return {}
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper="paper" in base_url.lower(),
        )
        acct = client.get_account()
        equity = float(acct.equity)
        last_equity = float(getattr(acct, "last_equity", equity) or equity)
        total_pnl = equity - DEFAULT_CAPITAL
        daily_pnl = equity - last_equity
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "history": [
                {"date": today, "total_portfolio_value": equity, "drawdown": 0.0, "total_pnl": total_pnl}
            ],
            "current_value": equity,
            "drawdown": max(0.0, (DEFAULT_CAPITAL - equity) / DEFAULT_CAPITAL) if equity < DEFAULT_CAPITAL else 0.0,
            "total_pnl": total_pnl,
            "daily_pnl": daily_pnl,
            "source": "alpaca_live",
        }
    except Exception as e:
        logger.warning(f"_alpaca_portfolio_live fallback failed: {e}")
        return {}


def _alpaca_history_live() -> list:
    """
    Fetch 1-month daily equity history from Alpaca portfolio history API.
    Returns list of {date, total_portfolio_value, drawdown, total_pnl} dicts.
    """
    try:
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not api_key or not secret_key:
            return []
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper="paper" in base_url.lower(),
        )
        # alpaca-py uses keyword args; period="1M" gives ~30 days of daily bars
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest
            hist = client.get_portfolio_history(
                GetPortfolioHistoryRequest(period="1M", timeframe="1D", extended_hours=False)
            )
        except Exception:
            # Older alpaca-py versions
            hist = client.get_portfolio_history(period="1M", timeframe="1D")

        records = []
        for ts, equity in zip(hist.timestamp or [], hist.equity or []):
            if equity is None or equity <= 0:
                continue
            date_str = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
            eq = float(equity)
            records.append({
                "date": date_str,
                "total_portfolio_value": round(eq, 2),
                "drawdown": round(max(0.0, (DEFAULT_CAPITAL - eq) / DEFAULT_CAPITAL), 4),
                "total_pnl": round(eq - DEFAULT_CAPITAL, 2),
            })
        return records
    except Exception as e:
        logger.warning(f"_alpaca_history_live: {e}")
        return []


@app.get("/api/portfolio")
def get_portfolio(_auth=Depends(require_auth)):
    """Return historical portfolio values and current metrics. Falls back to live Alpaca."""
    try:
        engine = get_engine()
        query = text(
            "SELECT date, total_portfolio_value, drawdown, total_pnl "
            "FROM daily_pnl ORDER BY date ASC LIMIT 365"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        # Treat DB as placeholder if empty OR all rows are exactly at starting capital
        # (means the pipeline hasn't run yet or daily_pnl was never properly updated)
        placeholder = df.empty or (
            len(df) <= 2
            and float(df["total_portfolio_value"].max()) == DEFAULT_CAPITAL
        )

        if placeholder:
            live = _alpaca_portfolio_live()
            if live:
                live_hist = _alpaca_history_live()
                return {**live, "history": live_hist if live_hist else live["history"]}
            return {"history": [], "current_value": DEFAULT_CAPITAL, "drawdown": 0.0, "total_pnl": 0.0}

        latest = df.iloc[-1]
        db_total_pnl = float(latest["total_pnl"])
        db_drawdown = float(latest["drawdown"])

        # Enrich with live Alpaca for current equity + daily change
        # Also override total_pnl / drawdown if the DB shows stale zeros
        live = {}
        try:
            live = _alpaca_portfolio_live()
        except Exception:
            pass

        current_value = live.get("current_value", float(latest["total_portfolio_value"]))
        daily_pnl = live.get("daily_pnl", 0.0)

        # Prefer live Alpaca total_pnl/drawdown when DB values are clearly stale
        # (DB shows 0 P&L but equity ≠ initial capital → pipeline hasn't updated yet)
        total_pnl = db_total_pnl
        drawdown = db_drawdown
        if live and abs(db_total_pnl) < 1.0 and abs(current_value - DEFAULT_CAPITAL) > 100:
            total_pnl = live.get("total_pnl", db_total_pnl)
            drawdown = live.get("drawdown", db_drawdown)

        return {
            "history": df.to_dict(orient="records"),
            "current_value": current_value,
            "drawdown": drawdown,
            "total_pnl": total_pnl,
            "daily_pnl": daily_pnl,
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
    Falls back to last 35 days when daily_pnl DB is empty (before first pipeline run).
    """
    try:
        import math
        import yfinance as yf

        engine = get_engine()
        with engine.connect() as conn:
            dates = conn.execute(text(
                "SELECT MIN(date), MAX(date) FROM daily_pnl"
            )).fetchone()

        # Always show the last 90 days of benchmark so the chart is useful even when
        # the DB only has a handful of stale rows. The end date is always today.
        end = datetime.now().strftime("%Y-%m-%d")
        if not dates or not dates[0]:
            start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        else:
            start_date, _ = dates
            try:
                db_start = datetime.fromisoformat(start_date)
                # Use whichever is earlier: DB start or 90 days ago
                ninety_ago = datetime.now() - timedelta(days=90)
                start = min(db_start, ninety_ago).strftime("%Y-%m-%d")
            except Exception:
                start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        initial_pv = DEFAULT_CAPITAL

        bench_data = yf.download(BENCHMARK_SYMBOL, start=start, end=end, auto_adjust=True, progress=False)
        if bench_data.empty:
            return {"benchmark": [], "label": BENCHMARK_LABEL}

        close = bench_data["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]

        first_close = float(close.iloc[0])
        if first_close == 0:
            return {"benchmark": [], "label": BENCHMARK_LABEL}

        records = []
        for date, val in close.items():
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10]
            normalized = initial_pv * (float(val) / first_close)
            # Guard against NaN / Inf which break JSON serialization
            if math.isnan(normalized) or math.isinf(normalized):
                continue
            records.append({"date": date_str, "value": round(normalized, 2)})

        return {"benchmark": records, "label": f"{BENCHMARK_LABEL} (normalized)"}
    except Exception as e:
        logger.error(f"get_benchmark: {e}")
        return {"benchmark": [], "label": BENCHMARK_LABEL, "error": str(e)}


# ── Positions API ────────────────────────────────────────────────────────────
def _alpaca_positions_live() -> list:
    """
    Fallback: fetch positions directly from Alpaca API when DB is empty.
    Returns a list in the same shape as the DB query so callers are agnostic.
    """
    try:
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not api_key or not secret_key:
            return []
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper="paper" in base_url.lower(),
        )
        positions = client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "quantity": int(p.qty),
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "unrealized_pnl_pct": round(float(p.unrealized_plpc) * 100, 2),
                "pnl_pct": round(float(p.unrealized_plpc) * 100, 2),
                "trade_type": "CNC",
                "strategy": "alpaca_live",
                "side": p.side.value,
            }
            for p in positions
        ]
    except Exception as e:
        logger.warning(f"_alpaca_positions_live fallback failed: {e}")
        return []


@app.get("/api/positions")
def get_positions(_auth=Depends(require_auth)):
    """Return active positions. DB-first; falls back to live Alpaca if DB is empty."""
    try:
        engine = get_engine()
        query = text(
            "SELECT ticker, quantity, entry_price, pnl_pct, trade_type "
            "FROM open_positions WHERE status='OPEN'"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        if not df.empty:
            return df.to_dict(orient="records")
        # DB empty → pull live from Alpaca so the UI shows real positions immediately
        logger.info("open_positions DB empty — falling back to live Alpaca positions")
        return _alpaca_positions_live()
    except Exception as e:
        logger.error(f"get_positions: {e}")
        return []


@app.get("/api/positions/detailed")
def get_positions_detailed(_auth=Depends(require_auth)):
    """Open positions with entry context, P&L, and holding duration. Falls back to Alpaca."""
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
            # Fallback: build rich detail from live Alpaca positions
            live = _alpaca_positions_live()
            today_str = datetime.now().strftime("%Y-%m-%d")
            return [
                {
                    "ticker": p["ticker"],
                    "quantity": p["quantity"],
                    "entry_price": p["entry_price"],
                    "current_price": p.get("current_price", p["entry_price"]),
                    "market_value": p.get("market_value", 0.0),
                    "unrealized_pnl": p.get("unrealized_pnl", 0.0),
                    "unrealized_pnl_pct": p.get("unrealized_pnl_pct", 0.0),
                    "entry_date": today_str,
                    "days_held": 0,
                    "trade_type": "CNC",
                    "strategy": "alpaca_live",
                    "stop_loss_pct": 0.08,
                    "take_profit_pct": 0.20,
                    "max_hold_days": 90,
                    "sector": "US",
                }
                for p in live
            ]

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


@app.get("/api/performance")
def get_performance(_auth=Depends(require_auth)):
    """Latest benchmark truth layer row plus recent history."""
    try:
        engine = get_engine()
        query = text("SELECT * FROM performance_daily ORDER BY date ASC LIMIT 365")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        if df.empty:
            return {"latest": {}, "history": []}
        return {"latest": df.iloc[-1].to_dict(), "history": df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"get_performance: {e}")
        return {"latest": {}, "history": [], "error": str(e)}


@app.get("/api/watchlist")
def get_watchlist(_auth=Depends(require_auth)):
    """Group latest run reasoning into a small attention list."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            run_row = conn.execute(text(
                "SELECT run_id FROM agent_reasoning WHERE run_id IS NOT NULL "
                "ORDER BY id DESC LIMIT 1"
            )).fetchone()
            if not run_row:
                return []
            rows = conn.execute(text(
                "SELECT ticker, agent_name, action, confidence, rationale "
                "FROM agent_reasoning WHERE run_id = :run_id ORDER BY id ASC"
            ), {"run_id": run_row[0]}).mappings().all()

        grouped = {}
        for row in rows:
            item = grouped.setdefault(row["ticker"], {
                "ticker": row["ticker"],
                "run_id": run_row[0],
                "agent_count": 0,
                "attention_score": 0.0,
                "strongest_agent": "",
                "beginner_reason": "",
            })
            item["agent_count"] += 1
            conf = float(row["confidence"] or 0.0)
            item["attention_score"] += conf
            if conf >= float(item.get("_max_conf", -1)):
                item["_max_conf"] = conf
                item["strongest_agent"] = row["agent_name"]
                item["beginner_reason"] = f"strongest agent view: {row['rationale']}"

        out = []
        for item in grouped.values():
            item["attention_score"] = round(item["attention_score"] / max(1, item["agent_count"]), 4)
            item.pop("_max_conf", None)
            out.append(item)
        return sorted(out, key=lambda x: -x["attention_score"])
    except Exception as e:
        logger.error(f"get_watchlist: {e}")
        return []


@app.get("/api/decision-detail/{run_id}")
def get_decision_detail(run_id: str, _auth=Depends(require_auth)):
    """Audit drill-down for one run."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            observations = conn.execute(text(
                "SELECT * FROM market_observations WHERE run_id = :run_id ORDER BY id ASC"
            ), {"run_id": run_id}).mappings().all()
            reasoning = conn.execute(text(
                "SELECT * FROM agent_reasoning WHERE run_id = :run_id ORDER BY id ASC"
            ), {"run_id": run_id}).mappings().all()

        tickers = sorted({row["ticker"] for row in reasoning})
        return {
            "run_id": run_id,
            "observations": [dict(row) for row in observations],
            "reasoning": [dict(row) for row in reasoning],
            "beginner_explanation": {
                "headline": f"{len(tickers)} ticker(s) reviewed by the agent team.",
                "ticker_explanations": {
                    ticker: f"Agent team recorded {sum(1 for r in reasoning if r['ticker'] == ticker)} view(s)."
                    for ticker in tickers
                },
            },
        }
    except Exception as e:
        logger.error(f"get_decision_detail: {e}")
        return {"run_id": run_id, "observations": [], "reasoning": [], "error": str(e)}


@app.get("/api/rl")
def get_rl_lab(_auth=Depends(require_auth)):
    """Recent RL/meta-allocation evaluations."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT * FROM rl_model_evaluations ORDER BY id DESC LIMIT 20"
            )).mappings().all()
        return {"evaluations": [dict(row) for row in rows]}
    except Exception as e:
        logger.error(f"get_rl_lab: {e}")
        return {"evaluations": [], "error": str(e)}


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

            # Normalize reasoning keys so frontend always finds what it expects
            r_data = dict(reasoning.get(ticker, {}))
            # Map analyst_reasoning → reasoning  (app.js looks for r.reasoning)
            if not r_data.get("reasoning"):
                r_data["reasoning"] = r_data.get("analyst_reasoning", "")
            # Map agent_signals → research_signals (app.js looks for agent_name field)
            if "agent_signals" in r_data and "research_signals" not in r_data:
                r_data["research_signals"] = [
                    {
                        "agent_name": s.get("agent", "?"),
                        "action": s.get("action", "HOLD"),
                        "confidence": float(s.get("confidence", 0)),
                    }
                    for s in r_data["agent_signals"]
                ]
            # Build a committee stub if not present so r.committee.reasoning resolves
            if not r_data.get("committee") and r_data.get("reasoning"):
                r_data["committee"] = {
                    "reasoning": r_data["reasoning"],
                    "strategy": r_data.get("strategy_used", "--"),
                }

            positions.append({
                "ticker": ticker,
                "weight_pct": round(w * 100, 2),
                "direction": direction,
                "capital": round(capital_alloc, 0),
                "rupees": round(capital_alloc, 0),  # backward compat
                "last_price": round(price, 2) if price else None,
                "est_shares": est_shares,
                "reasoning": r_data,
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


# ── Circuit Breakers API ────────────────────────────────────────────────────
@app.get("/api/circuits")
def get_circuits(_auth=Depends(require_auth)):
    """Return configured circuit breaker rules and their status."""
    try:
        circuits = [
            {"name": "LongOnlyRule", "trip": "w < 0", "status": "armed",
             "description": "Zeroes any negative (short) weights"},
            {"name": "DrawdownCB", "trip": "-15%", "status": "armed",
             "description": "Portfolio drawdown hard cutoff"},
            {"name": "VolatilityCB", "trip": "VIX > 35", "status": "armed",
             "description": "Reduces exposure during high VIX"},
            {"name": "MaxPositionRule", "trip": "10%", "status": "armed",
             "description": "Per-ticker concentration cap"},
            {"name": "TimeWindowRule", "trip": "09:35-15:55 ET", "status": "armed",
             "description": "NYSE trading hours only"},
            {"name": "PositionStopLoss", "trip": "-15% core / -7% tactical", "status": "armed",
             "description": "Per-position stop loss (Buffett-style: wide for core, tight for tactical)"},
        ]
        return {"circuits": circuits, "total": len(circuits), "all_armed": True}
    except Exception as e:
        logger.error(f"get_circuits: {e}")
        return {"circuits": [], "total": 0, "error": str(e)}


# ── Sleeves API (v2 architecture) ───────────────────────────────────────────
@app.get("/api/sleeves")
def get_sleeves(_auth=Depends(require_auth)):
    """
    Return the latest sleeve-based pipeline snapshot. Written by main_us_v2.py
    on each cycle. Returns {"available": false} if v2 has never run.
    """
    try:
        snap_path = Path(__file__).resolve().parents[2] / ".cache" / "sleeve_snapshots" / "latest.json"
        if not snap_path.exists():
            return {"available": False, "message": "v2 pipeline has not run yet. Execute: python main_us_v2.py"}
        with open(snap_path) as f:
            data = json.load(f)
        data["available"] = True
        return data
    except Exception as e:
        logger.error(f"get_sleeves: {e}")
        return {"available": False, "error": str(e)}


@app.get("/api/factors/snapshot")
def get_factors_snapshot(_auth=Depends(require_auth)):
    """
    Compute live factor top-decile for the SP100 universe. Cached server-side
    by the data provider so subsequent calls are fast.

    For the dashboard's Factors tab — shows where each ticker ranks across
    the 5 factor dimensions.
    """
    try:
        from src.factors import (
            ValueFactor, QualityFactor, MomentumFactor,
            DefensiveFactor, TrendFactor, sp100_tickers,
        )
        universe = sp100_tickers()
        out = {}
        for name, factor_cls in [
            ("value", ValueFactor),
            ("quality", QualityFactor),
            ("momentum", MomentumFactor),
            ("defensive", DefensiveFactor),
            ("trend", TrendFactor),
        ]:
            try:
                res = factor_cls().compute(universe)
                out[name] = {
                    "top_10": [
                        {"ticker": t, "score": round(res.scores[t], 3),
                         "confidence": round(res.confidence.get(t, 1.0), 3)}
                        for t in res.top_n(10)
                    ],
                    "n_scored": len(res.scores),
                    "notes": res.notes,
                }
            except Exception as e:
                out[name] = {"error": str(e), "top_10": []}
        return {"available": True, "factors": out, "universe_size": len(universe)}
    except Exception as e:
        logger.error(f"get_factors_snapshot: {e}")
        return {"available": False, "error": str(e)}


# ── Static files & root ─────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.webapp.server:app", host="127.0.0.1", port=8000, reload=True)
