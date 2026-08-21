from langchain_core.messages import SystemMessage, HumanMessage

from .models import gemini_fast
from .prompts import ROUTER_SYSTEM
from .schemas import RouterDecision
from .state import State

# ============================================================
# 5. ROUTER
# ============================================================
#
# Router trả lời câu hỏi:
#
#     "Topic này có cần web research không?"
#
# Có 3 mode:
#
# closed_book
#     ↓
# Không cần web.
#
# hybrid
#     ↓
# Có kiến thức nền + một phần thông tin mới.
#
# open_book
#     ↓
# Phụ thuộc mạnh vào thông tin mới nhất.
#
# ============================================================

def router_node(state: State) -> dict:

    topic = state["topic"]

    # Structured output giúp LLM trả đúng RouterDecision
    # Router chỉ là task nhẹ nên giao cho Gemini Flash-Lite.
    decider = gemini_fast.with_structured_output(
        RouterDecision
    )

    decision = decider.invoke(
        [
            SystemMessage(
                content=ROUTER_SYSTEM
            ),
            HumanMessage(
                content=(
                    f"Topic: {topic}\n"
                    f"Final output language: "
                    f"{state.get('language', 'English')}"
                )
            ),
        ]
    )

    print("\n========== ROUTER ==========")
    print(decision.model_dump())

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

# ------------------------------------------------------------
# Router conditional edge
# ------------------------------------------------------------

def route_next(state: State) -> str:

    if state["needs_research"]:
        return "research"

    return "orchestrator"