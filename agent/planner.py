from langchain_core.messages import SystemMessage, HumanMessage

from .models import gemini_main
from .prompts import ORCH_SYSTEM
from .schemas import Plan
from .state import State

# ============================================================
# 8. ORCHESTRATOR / PLANNER
# ============================================================
#
# Router:
#
#     "Có cần research không?"
#
# Research:
#
#     "Đây là evidence."
#
# Orchestrator:
#
#     "Dựa vào topic + mode + evidence,
#      hãy lập kế hoạch."
#
# ============================================================

def orchestrator_node(state: State) -> dict:

    # Planner là một trong những node quan trọng nhất,
    # nên dùng Gemini model chính.
    planner = gemini_main.with_structured_output(
        Plan
    )

    evidence = state.get(
        "evidence",
        []
    )

    mode = state.get(
        "mode",
        "closed_book"
    )

    evidence_data = [
        e.model_dump()
        for e in evidence
    ]

    plan = planner.invoke(
        [
            SystemMessage(
                content=ORCH_SYSTEM
            ),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Output language: {state.get('language', 'English')}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence "
                    f"(ONLY use for fresh claims; "
                    f"may be empty):\n"
                    f"{evidence_data[:16]}"
                )
            ),
        ]
    )

    print("\n========== PLAN ==========")
    print(
        plan.model_dump_json(
            indent=2
        )
    )

    return {
        "plan": plan
    }