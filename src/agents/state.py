from typing import TypedDict, Annotated, Sequence, Any, Dict, List, NotRequired
import operator
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    """
    Represents the shared state passed between agents in the LangGraph workflow.
    """
    # Current state of the pipeline
    current_asset: str
    timestamp: str

    # Raw Data
    market_data: Dict[str, Any]
    alternative_data: Dict[str, Any]
    technical_indicators: Dict[str, Any]

    # Strategy selection (India multi-strategy layer)
    active_strategies: List[str]
    strategy_scores: Dict[str, float]
    current_strategy: str

    # Trade type and position management (delivery vs intraday)
    trade_type: str                        # "MIS", "CNC", or "SKIP"
    stop_loss_pct: float                   # per-position SL threshold (e.g., 0.08)
    take_profit_pct: float                 # per-position TP threshold (e.g., 0.20)
    intraday_budget: float                 # available intraday capital for this cycle
    delivery_budget: float                 # available delivery capital for this cycle

    # Decisions (annotated with generic list append for accumulation)
    research_signals: Annotated[Sequence[Dict[str, Any]], operator.add]
    committee_decision: Dict[str, Any]
    allocation_request: Dict[str, Any]
    risk_approval: Dict[str, Any]
    execution_result: Dict[str, Any]

    # Global/Portfolio context
    portfolio_state: Dict[str, Any]

    # Persistent agent memory (seeded from memory/*.md at run start).
    # Keys: "strategy", "learnings", "recent_trades". Optional: agents may ignore.
    context_memory: NotRequired[Dict[str, str]]

class GenericAgentDecision(BaseModel):
    """
    Standardized output structure for agent decisions.
    """
    agent_name: str
    action: str = Field(description="The action taken, e.g., 'PROPOSE', 'APPROVE', 'REJECT'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    rationale: str = Field(description="Detailed reasoning for the decision.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom data for specific agents.")
