from typing import TypedDict, Annotated, Sequence, Any, Dict, List
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

    # Decisions (annotated with generic list append for accumulation)
    research_signals: Annotated[Sequence[Dict[str, Any]], operator.add]
    committee_decision: Dict[str, Any]
    allocation_request: Dict[str, Any]
    risk_approval: Dict[str, Any]
    execution_result: Dict[str, Any]

    # Global/Portfolio context
    portfolio_state: Dict[str, Any]

class GenericAgentDecision(BaseModel):
    """
    Standardized output structure for agent decisions.
    """
    agent_name: str
    action: str = Field(description="The action taken, e.g., 'PROPOSE', 'APPROVE', 'REJECT'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    rationale: str = Field(description="Detailed reasoning for the decision.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom data for specific agents.")
