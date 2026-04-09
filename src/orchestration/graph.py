from langgraph.graph import StateGraph, END

from src.agents.cleaning import run_cleaning
from src.agents.eda import run_eda
from src.agents.ingestion import run_ingestion
from src.agents.sentiment import run_sentiment
from src.agents.reviewer import run_reviewer
from src.agents.segmentation import run_segmentation
from src.agents.strategy import run_strategy
from src.agents.topic import run_topic
from src.agents.writer import run_writer
from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def ingestion_node(state: WorkflowState) -> WorkflowState:
    return run_ingestion(state)


def cleaning_node(state: WorkflowState) -> WorkflowState:
    return run_cleaning(state)


def eda_node(state: WorkflowState) -> WorkflowState:
    return run_eda(state)


def sentiment_node(state: WorkflowState) -> WorkflowState:
    return run_sentiment(state)


def segmentation_node(state: WorkflowState) -> WorkflowState:
    return run_segmentation(state)


def topic_node(state: WorkflowState) -> WorkflowState:
    return run_topic(state)


def strategy_node(state: WorkflowState) -> WorkflowState:
    return run_strategy(state)


def writer_node(state: WorkflowState) -> WorkflowState:
    return run_writer(state)


def reviewer_node(state: WorkflowState) -> WorkflowState:
    return run_reviewer(state)


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
    graph.add_node("segmentation", segmentation_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("topic", topic_node)
    graph.add_node("strategy", strategy_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "cleaning")
    graph.add_edge("cleaning", "eda")
    graph.add_edge("eda", "segmentation")
    graph.add_edge("segmentation", "sentiment")
    graph.add_edge("sentiment", "topic")
    graph.add_edge("topic", "strategy")
    graph.add_edge("strategy", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", "final")
    graph.add_edge("final", END)

    return graph.compile()