"""
Alerting Engine
===============
Sends drawdown breach / regime-shift / model-degradation alerts via:
  - Slack webhook (set SLACK_WEBHOOK_URL in .env)
  - Email (set ALERT_EMAIL + SMTP_* vars in .env)
  - File log (always active)

Usage:
    from src.engine.alerting import AlertEngine
    alerts = AlertEngine()
    alerts.check_drawdown(current_dd=0.12, threshold=0.10, context="SPY portfolio")
"""

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from src import config  # noqa: F401

try:
    from slack_sdk.webhook import WebClient, WebhookClient
except ImportError:
    WebhookClient = None

logger = logging.getLogger(__name__)

ALERT_LOG = Path("alert_history.jsonl")


class AlertEngine:
    """
    Monitors portfolio metrics and fires alerts when thresholds are breached.

    All thresholds are loaded from environment variables (set in .env):
        SLACK_WEBHOOK_URL          — Slack incoming webhook
        MAX_DRAWDOWN_THRESHOLD     — float, e.g. 0.10 for 10%  (default 0.15)
        MIN_SHARPE_THRESHOLD       — float rolling Sharpe floor (default 0.0)
        ALERT_EMAIL                — recipient email address
        SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS — email sender config
    """

    def __init__(self):
        self.slack_url       = os.environ.get("SLACK_WEBHOOK_URL", "")
        self.dd_threshold    = float(os.environ.get("MAX_DRAWDOWN_THRESHOLD", "0.15"))
        self.sharpe_floor    = float(os.environ.get("MIN_SHARPE_THRESHOLD", "0.0"))
        self.alert_email     = os.environ.get("ALERT_EMAIL", "")
        self.smtp_host       = os.environ.get("SMTP_HOST", "")
        self.smtp_port       = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user       = os.environ.get("SMTP_USER", "")
        self.smtp_pass       = os.environ.get("SMTP_PASS", "")

    # ──────────────────────────────────────── Public API

    def check_drawdown(self, current_dd: float, context: str = "", threshold: Optional[float] = None) -> bool:
        """
        Fire alert if current_dd exceeds threshold.
        Returns True if alert was fired.
        """
        thresh = threshold if threshold is not None else self.dd_threshold
        if current_dd > thresh:
            msg = (
                f"DRAWDOWN ALERT | {context} | "
                f"Current DD: {current_dd:.1%} > Threshold: {thresh:.1%}"
            )
            self._fire(msg, level="WARNING")
            return True
        return False

    def check_sharpe(self, rolling_sharpe: float, context: str = "") -> bool:
        """Fire alert if rolling Sharpe drops below the configured floor."""
        if rolling_sharpe < self.sharpe_floor:
            msg = (
                f"SHARPE DEGRADATION | {context} | "
                f"Rolling Sharpe: {rolling_sharpe:.3f} < Floor: {self.sharpe_floor:.3f}"
            )
            self._fire(msg, level="WARNING")
            return True
        return False

    def check_regime_shift(self, prev_regime: int, new_regime: int,
                           ticker: str, regime_labels: Optional[dict] = None) -> bool:
        """Fire alert on any regime state change."""
        if prev_regime != new_regime:
            labels = regime_labels or {0: "Bull-Quiet", 1: "Bull-Vol", 2: "Bear-Quiet", 3: "Bear-Vol"}
            msg = (
                f"REGIME SHIFT | {ticker} | "
                f"{labels.get(prev_regime, prev_regime)} -> {labels.get(new_regime, new_regime)}"
            )
            self._fire(msg, level="INFO")
            return True
        return False

    def check_confidence(self, agent_score: float, threshold: float = 0.40, context: str = "") -> bool:
        """Fire alert if LLM agent confidence falls dangerously low."""
        if agent_score < threshold:
            msg = f"LOW CONFIDENCE | {context} | Agent score {agent_score:.2f} < {threshold:.2f}"
            self._fire(msg, level="WARNING")
            return True
        return False
        
    def check_position_limit(self, weight: float, threshold: float = 0.95, tick: str = "") -> bool:
        """Fire alert if model tries to breach hard sizing limits."""
        if abs(weight) > threshold:
            msg = f"POSITION LIMIT BREACH | {tick} | Target weight {weight:.1%} > {threshold:.1%}"
            self._fire(msg, level="WARNING")
            return True
        return False

    def notify(self, message: str, level: str = "INFO") -> None:
        """Send an arbitrary alert message."""
        self._fire(message, level=level)

    # ──────────────────────────────────────── Internal dispatch

    def _fire(self, message: str, level: str = "WARNING") -> None:
        timestamp = datetime.utcnow().isoformat()
        full_msg = f"[{timestamp}] [{level}] {message}"

        # 1. Always log to file
        self._log_to_file(timestamp, level, message)

        # 2. Python logger
        if level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)

        # 3. Slack
        if self.slack_url:
            self._send_slack(full_msg)

        # 4. Email
        if self.alert_email and self.smtp_host:
            self._send_email(subject=f"AegisQuant {level}: {message[:60]}", body=full_msg)

        print(full_msg)

    def _log_to_file(self, timestamp: str, level: str, message: str) -> None:
        record = {"timestamp": timestamp, "level": level, "message": message}
        try:
            with open(ALERT_LOG, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.error("Failed to write alert log: %s", exc)

    def _send_slack(self, message: str) -> None:
        if WebhookClient is None:
            logger.warning("slack_sdk not installed. Alert dropped: %s", message)
            return
            
        try:
            webhook = WebhookClient(self.slack_url)
            response = webhook.send(text=message)
            if response.status_code != 200:
                logger.warning("Slack webhook returned %s", response.status_code)
        except Exception as exc:
            logger.warning("Slack alert failed: %s", exc)

    def _send_email(self, subject: str, body: str) -> None:
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = self.alert_email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.smtp_user, [self.alert_email], msg.as_string())
        except Exception as exc:
            logger.warning("Email alert failed: %s", exc)


# Module-level singleton — import this where needed
alert_engine = AlertEngine()
