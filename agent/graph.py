from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.data_understanding import data_understanding_node
from agent.nodes.preprocessing import preprocessing_node
from agent.nodes.modeling import modeling_node
from agent.nodes.evaluation import evaluation_node
from agent.nodes.optimization import optimization_node
from agent.nodes.decision import select_candidate_models, should_optimize
from tools.report_tools import report_node


def build_graph():
    graph = StateGraph(AgentState)

    # === Tambah semua nodes ===
    graph.add_node("data_understanding", data_understanding_node)
    graph.add_node("select_models", select_candidate_models)
    graph.add_node("preprocessing", preprocessing_node)
    graph.add_node("modeling", modeling_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("optimization", optimization_node)
    graph.add_node("report", report_node)

    # === Entry point ===
    graph.set_entry_point("data_understanding")

    # === Linear edges ===
    graph.add_edge("data_understanding", "select_models")
    graph.add_edge("select_models", "preprocessing")
    graph.add_edge("preprocessing", "modeling")
    graph.add_edge("modeling", "evaluation")

    # === Conditional edge: iterasi atau selesai ===
    graph.add_conditional_edges(
        "evaluation",
        should_optimize,
        {
            "optimize": "optimization",
            "finalize": "report",
        }
    )

    # === Loop back setelah optimization ===
    graph.add_edge("optimization", "modeling")

    # === End ===
    graph.add_edge("report", END)

    return graph.compile()
