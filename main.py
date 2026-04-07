from pprint import pprint

from src.orchestration.graph import build_graph
from src.orchestration.state import WorkflowState


def main():
    app = build_graph()

    initial_state = WorkflowState(
        input_paths=[],
        prior_artifacts=[],
    )

    result = app.invoke(initial_state)

    if hasattr(result, "model_dump"):
        pprint(result.model_dump())
    else:
        pprint(result)


if __name__ == "__main__":
    main()