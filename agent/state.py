import operator

from typing import Annotated, List, Optional, TypedDict

from .schemas import EvidenceItem, Plan


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================
#
# State là "bộ nhớ dùng chung" của workflow.
#
# Có thể hình dung:
#
#                 STATE
#                   │
#       ┌───────────┼────────────┐
#       ▼           ▼            ▼
#    Router      Planner       Worker
#       │           │            │
#       └───────────┼────────────┘
#                   ▼
#                Reducer
#
# ============================================================


class State(TypedDict):

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    topic: str

    # Ngôn ngữ đầu ra được xác định từ topic của người dùng.
    # Mục tiêu là giữ toàn bộ workflow nhất quán về ngôn ngữ.
    language: str

    # --------------------------------------------------------
    # Router / Research
    # --------------------------------------------------------

    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]

    # --------------------------------------------------------
    # Planner
    # --------------------------------------------------------

    plan: Optional[Plan]

    # --------------------------------------------------------
    # Workers
    # --------------------------------------------------------

    sections: Annotated[
        List[tuple[int, str]],
        operator.add
    ]

    # --------------------------------------------------------
    # Reducer / Images
    # --------------------------------------------------------

    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    final: str