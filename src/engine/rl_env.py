import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.engine.cost_model import cost_model


class HedgeFundEnv(gym.Env):
    """
    Custom Environment that follows gym interface for Portfolio Manager Optimization.
    The agent receives state signals from LLM Research Agents and Technicals,
    and outputs a target weight [-1.0, 1.0] for the asset.

    Phase-0 changes:
    - All randomness goes through self.np_random (seeded by gymnasium) — no bare random module.
    - Transaction costs are computed by TransactionCostModel (commission + spread + impact).
    - Reward uses multi-component formula: log_return + drawdown_penalty + cost_penalty + concentration_penalty.
    - Environment accepts seed via reset(seed=...) for reproducible training runs.
    """

    metadata = {"render.modes": ["console"]}

    # Approximate ADV for the simulated asset (used by cost model)
    SIMULATED_ADV = 10_000_000.0
    SIMULATED_PRICE = 100.0

    def __init__(self, initial_balance: float = 1_000_000.0, seed: int | None = None):
        super().__init__()

        self.initial_balance = initial_balance
        self._init_seed = seed

        # Action space: target weight in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # State: [vol, quant_sig, fund_sig, macro_sig, sent_sig, drawdown]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.reset(seed=seed)

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options=None):
        seed = seed if seed is not None else self._init_seed
        super().reset(seed=seed)  # sets self.np_random

        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_step = 0
        self.max_steps = 100
        self.asset_price = self.SIMULATED_PRICE
        self.drawdown = 0.0
        self.current_weight = 0.0  # track current position to compute trade size

        return self._next_observation(), {}

    # ------------------------------------------------------------------
    def _next_observation(self) -> np.ndarray:
        vol = float(self.np_random.uniform(0.01, 0.20))
        q_sig = float(self.np_random.uniform(-1.0, 1.0))
        f_sig = float(self.np_random.uniform(-1.0, 1.0))
        m_sig = float(self.np_random.uniform(-1.0, 1.0))
        s_sig = float(self.np_random.uniform(-1.0, 1.0))

        self.drawdown = (
            (self.peak_balance - self.balance) / self.peak_balance
            if self.peak_balance > 0
            else 0.0
        )

        self._current_consensus = (q_sig + f_sig + m_sig + s_sig) / 4.0
        self._current_vol = vol

        return np.array([vol, q_sig, f_sig, m_sig, s_sig, self.drawdown], dtype=np.float32)

    # ------------------------------------------------------------------
    def step(self, action):
        self.current_step += 1
        target_weight = float(np.clip(action[0], -1.0, 1.0))

        # ---- Simulate market return ----
        noise = float(self.np_random.normal(0.0, self._current_vol))
        actual_return = (self._current_consensus * 0.05) + noise

        # ---- Transaction cost (only on the weight change, not the full notional) ----
        weight_change = abs(target_weight - self.current_weight)
        trade_notional = self.balance * weight_change
        trade_quantity = trade_notional / self.asset_price if self.asset_price > 0 else 0.0
        transaction_cost, _ = cost_model.compute_cost(
            price=self.asset_price,
            quantity=trade_quantity,
            adv=self.SIMULATED_ADV,
            algo="MARKET",
        )
        self.current_weight = target_weight

        # ---- Update balance ----
        prev_balance = self.balance
        trade_pl = self.balance * target_weight * actual_return
        self.balance += trade_pl - transaction_cost

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # ---- Multi-component reward ----
        if prev_balance > 0:
            log_return = float(np.log(max(self.balance, 1e-8) / prev_balance))
        else:
            log_return = -10.0

        drawdown_penalty = -2.0 * max(0.0, self.drawdown - 0.05)
        cost_penalty = -(transaction_cost / prev_balance) if prev_balance > 0 else 0.0
        concentration_penalty = -0.1 * (target_weight ** 2)
        # Turnover penalty: penalises each position change proportionally to its size.
        # Prevents the agent ignoring the tiny cost_penalty signal and over-trading.
        turnover_penalty = -0.003 * weight_change

        reward = log_return + drawdown_penalty + cost_penalty + concentration_penalty + turnover_penalty

        # ---- Terminal conditions ----
        done = self.current_step >= self.max_steps
        truncated = False

        if self.balance <= self.initial_balance * 0.5:
            done = True
            reward -= 100.0

        info = {
            "balance": self.balance,
            "log_return": log_return,
            "drawdown": self.drawdown,
            "transaction_cost": transaction_cost,
        }

        return self._next_observation(), reward, done, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        print(
            f"Step: {self.current_step} | Balance: ${self.balance:,.2f} | "
            f"Drawdown: {self.drawdown*100:.2f}% | Weight: {self.current_weight:.2f}"
        )
