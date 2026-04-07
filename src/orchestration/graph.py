from langgraph.graph import StateGraph, END

from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def ingestion_node(state: WorkflowState) -> WorkflowState:
    logger.info("Running ingestion node")
    state.current_step = "ingestion"
    state.logs.append("Ingestion node executed.")
    state.results["ingestion"] = {
        "status": "success",
        "summary": "Dummy ingestion completed."
    }
    return state


def cleaning_node(state: WorkflowState) -> WorkflowState:
    logger.info("Running cleaning node")
    state.current_step = "cleaning"
    state.logs.append("Cleaning node executed.")
    state.results["cleaning"] = {
        "status": "success",
        "summary": "Dummy cleaning completed."
    }
    return state


def eda_node(state: WorkflowState) -> WorkflowState:
    logger.info("Running EDA node")
    state.current_step = "eda"
    state.logs.append("EDA node executed.")
    state.results["eda"] = {
        "status": "success",
        "summary": "Dummy EDA completed."
    }
    return state


def final_node(state: WorkflowState) -> WorkflowState:
    logger.info("Running final node")
    state.current_step = "final"
    state.status = "completed"
    state.logs.append("Workflow completed successfully.")
    return state


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("ingestion", ingestion_node)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("eda", eda_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "cleaning")
    graph.add_edge("cleaning", "eda")
    graph.add_edge("eda", "final")
    graph.add_edge("final", END)

    return graph.compile()