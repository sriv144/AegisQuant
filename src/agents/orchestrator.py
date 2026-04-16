import json
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from .state import AgentState
from .messaging import broker

# Import Agents
from .research.quant_agent import quant_agent
from .research.fundamental_agent import fundamental_agent
from .research.macro_agent import macro_agent
from .research.sentiment_agent import sentiment_agent

from .executive.strategy_committee_agent import strategy_committee_agent
from .executive.strategy_selector_agent import strategy_selector_agent
from .executive.cio_agent import cio_agent

from .portfolio.pm_agent import pm_agent
from .portfolio.asset_allocation_agent import asset_allocation_agent

from .risk.risk_officer_agent import risk_officer_agent
from .execution.execution_agent import execution_agent

class WorkflowOrchestrator:
    def __init__(self):
        self.workflow = StateGraph(AgentState)
        
        self.workflow.add_node("research", self._research_node)
        self.workflow.add_node("strategy_selector", self._strategy_selector_node)
        self.workflow.add_node("committee", self._committee_node)
        self.workflow.add_node("portfolio", self._portfolio_node)
        self.workflow.add_node("risk", self._risk_node)
        self.workflow.add_node("execution", self._execution_node)

        # Define graph edges
        self.workflow.set_entry_point("research")
        self.workflow.add_edge("research", "strategy_selector")
        self.workflow.add_edge("strategy_selector", "committee")
        self.workflow.add_conditional_edges(
            "committee",
            self._committee_router,
            {"allocate": "portfolio", "reject": END}
        )
        self.workflow.add_edge("portfolio", "risk")
        self.workflow.add_conditional_edges(
            "risk",
            self._risk_router,
            {"execute": "execution", "reject": END}
        )
        self.workflow.add_edge("execution", END)
        
        self.app = self.workflow.compile()

    def _log_state(self, step: str, state: AgentState):
        # We try to remove non-serializable fields if there are any, though they should be dicts.
        broker.publish(f"agent_state.{step}", {"current_asset": state.get('current_asset', 'N/A')})

    def _research_node(self, state: AgentState) -> Dict[str, Any]:
        """Runs all research agents in parallel (conceptually)"""
        print("[Workflow] Research Phase...")
        results = []
        
        # In LangGraph these could be multiple nodes branching out and converging.
        # For simplicity, we invoke sequentially here and accumulate outputs.
        # Each agent returns {"research_signals": [decision]} which get appended.
        q_res = quant_agent.invoke(state)
        f_res = fundamental_agent.invoke(state)
        m_res = macro_agent.invoke(state)
        s_res = sentiment_agent.invoke(state)
        
        signals = q_res['research_signals'] + f_res['research_signals'] + m_res['research_signals'] + s_res['research_signals']
        return {"research_signals": signals}

    def _strategy_selector_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Workflow] Strategy Selection...")
        return strategy_selector_agent.invoke(state)

    def _committee_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Workflow] Committee Review...")
        return strategy_committee_agent.invoke(state)

    def _committee_router(self, state: AgentState) -> str:
        decision = state.get("committee_decision", {}).get("action", "REJECT")
        return "allocate" if decision == "PROPOSE" else "reject"

    def _portfolio_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Workflow] Portfolio Sizing...")
        
        # 1. PM requests allocation
        pm_decision = pm_agent.invoke(state)
        
        # We must manually merge the state change before feeding it to the allocator in this sequential chain
        temp_state = state.copy()
        temp_state.update(pm_decision)
        
        # 2. Allocator adjusts
        aa_decision = asset_allocation_agent.invoke(temp_state)
        
        return aa_decision

    def _risk_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Workflow] Risk Assessment...")
        return risk_officer_agent.invoke(state)

    def _risk_router(self, state: AgentState) -> str:
        decision = state.get("risk_approval", {}).get("action", "REJECT")
        return "execute" if decision == "APPROVE" else "reject"

    def _execution_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Workflow] Execution Phase...")
        return execution_agent.invoke(state)

    def run_cycle(self, initial_state: AgentState):
        print(f"--- Starting cycle for {initial_state.get('current_asset')} ---")
        return self.app.invoke(initial_state)

orchestrator = WorkflowOrchestrator()
