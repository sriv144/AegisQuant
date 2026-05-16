import json
from typing import Dict, Any
from ..base_agent import BaseAgent
from ..state import AgentState

class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Sentiment_Research_Agent",
            role="Alternative Data & Sentiment Analyst tracking news feeds, social media, and retail flow. You capture irrational exuberance or panic before technicals do."
        )

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        print(f"[{self.name}] Analyzing sentiment for {state['current_asset']}...")
        
        # Prepare context
        alt_data = state.get("alternative_data", {})
        
        prompt = f"""
        Analyze the following sentiment aggregation for {state['current_asset']}:
        {json.dumps(alt_data, indent=2)}
        
        Evaluate public sentiment, news momentum, and fear/greed.
        Produce a JSON output matching this schema:
        {{
            "agent_name": "Sentiment_Research_Agent",
            "action": "PROPOSE_LONG" | "HOLD",
            "confidence": float (0.0 to 1.0),
            "rationale": "A concise explanation of the sentiment signal."
        }}
        """
        
        sentiment = alt_data.get("sentiment")
        action = "HOLD"
        confidence = 0.25
        rationale = "Fallback sentiment signal remains neutral."
        if isinstance(sentiment, (int, float)):
            if sentiment >= 0.2:
                action = "PROPOSE_LONG"
                confidence = min(0.75, 0.35 + abs(float(sentiment)))
                rationale = "Positive aggregate sentiment supports a long bias."
            elif sentiment <= -0.2:
                action = "HOLD"
                confidence = 0.25
                rationale = "Negative aggregate sentiment — staying flat (long-only mode)."

        fallback = {
            "agent_name": self.name,
            "action": action,
            "confidence": round(confidence, 4),
            "rationale": rationale,
        }

        decision = self._invoke_llm_json(prompt, fallback)

        return {"research_signals": [decision]}

sentiment_agent = SentimentAgent()
