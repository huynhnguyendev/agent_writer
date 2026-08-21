from datetime import date
from pathlib import Path
from typing import Optional

import os
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

from .state import State
from .router import router_node, route_next
from .research import research_node
from .planner import orchestrator_node
from .worker import fanout, worker_node
from .reducer import merge_content, decide_images
from .images import generate_and_place_images

# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

def get_database_url() -> str:
    """
    Lấy DATABASE_URL từ file .env.

    DATABASE_URL dùng để kết nối PostgreSQL,
    nơi LangGraph lưu checkpoint / state của workflow.
    """

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    # PostgreSQL trên Render thường cần SSL.
    # Nếu URL chưa có sslmode thì tự động thêm.
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

# ============================================================
# 13. BUILD REDUCER SUBGRAPH
# ============================================================
#
# Subgraph:
#
# START
#   ↓
# merge_content
#   ↓
# decide_images
#   ↓
# generate_and_place_images
#   ↓
# END
#
# ============================================================

reducer_graph = StateGraph(
    State
)

reducer_graph.add_node(
    "merge_content",
    merge_content,
)

reducer_graph.add_node(
    "decide_images",
    decide_images,
)

reducer_graph.add_node(
    "generate_and_place_images",
    generate_and_place_images,
)

reducer_graph.add_edge(
    START,
    "merge_content",
)

reducer_graph.add_edge(
    "merge_content",
    "decide_images",
)

reducer_graph.add_edge(
    "decide_images",
    "generate_and_place_images",
)

reducer_graph.add_edge(
    "generate_and_place_images",
    END,
)

reducer_subgraph = (
    reducer_graph.compile()
)

# ============================================================
# 14. MAIN GRAPH
# ============================================================
#
# Đây là toàn bộ Agentic Writer:
#
#
#                     START
#                       │
#                       ▼
#                    ROUTER
#                  /         \
#                 /           \
#                ▼             ▼
#           RESEARCH       ORCHESTRATOR
#                │             │
#                └──────┬──────┘
#                       ▼
#                  ORCHESTRATOR
#                       │
#                     FANOUT
#                  /    |    \
#                 ▼     ▼     ▼
#              Worker Worker Worker
#                 \     |     /
#                  \    |    /
#                    REDUCER
#                       │
#                ┌──────┴──────┐
#                ▼             ▼
#             Merge      Web Images
#                │             │
#                └──────┬──────┘
#                       ▼
#                      END
#
# ============================================================

g = StateGraph(
    State
)

# ------------------------------------------------------------
# Main nodes
# ------------------------------------------------------------

g.add_node(
    "router",
    router_node,
)

g.add_node(
    "research",
    research_node,
)

g.add_node(
    "orchestrator",
    orchestrator_node,
)

g.add_node(
    "worker",
    worker_node,
)

# Subgraph được gắn như một node
g.add_node(
    "reducer",
    reducer_subgraph,
)


# ------------------------------------------------------------
# Main edges
# ------------------------------------------------------------

g.add_edge(
    START,
    "router",
)

# Router quyết định:
#
# research
#     hoặc
#
# orchestrator
#
g.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator",
    },
)

# Research xong → Planner
g.add_edge(
    "research",
    "orchestrator",
)

# Planner → Fan-out Workers
g.add_conditional_edges(
    "orchestrator",
    fanout,
    ["worker"],
)

# Worker → Reducer Subgraph
g.add_edge(
    "worker",
    "reducer",
)

# Reducer → END
g.add_edge(
    "reducer",
    END,
)

# ============================================================
# 15. POSTGRES CHECKPOINTER
# ============================================================
#
# Đây là nâng cấp lớn so với InMemorySaver.
#
# InMemorySaver:
#
#     RAM
#      ↓
#     restart app
#      ↓
#     mất state
#
#
# PostgreSQL:
#
#     LangGraph
#         ↓
#     PostgreSQL
#         ↓
#     restart app
#         ↓
#     checkpoint vẫn còn
#
# ============================================================

DATABASE_URL = (
    get_database_url()
)

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)

checkpointer = PostgresSaver(
    _conn
)

# Tạo các bảng checkpoint nếu chưa tồn tại
checkpointer.setup()


# ------------------------------------------------------------
# Compile final application
# ------------------------------------------------------------

app = g.compile(
    checkpointer=checkpointer
)

# ============================================================
# 16. CONFIG
# ============================================================
#
# thread_id dùng để LangGraph biết:
#
#     "Đây là conversation / workflow nào?"
#
# PostgreSQL sẽ lưu checkpoint dựa trên thread này.
#
# ============================================================

config = {
    "configurable": {
        "thread_id": "test_thread_id_2"
    }
}

# ============================================================
# 17. RUNNER
# ============================================================

def detect_output_language(text: str) -> str:
    """
    Xác định ngôn ngữ đầu ra dựa trên input của người dùng.

    Với workflow hiện tại, ta ưu tiên tiếng Việt khi input có
    dấu/chữ đặc trưng của tiếng Việt. Nếu không, mặc định tiếng Anh.

    Không gọi thêm LLM cho bước này để tránh tăng latency và quota.
    """

    vietnamese_chars = set(
        "ăâđêôơưáàảãạắằẳẵặấầẩẫậ"
        "éèẻẽẹếềểễệíìỉĩị"
        "óòỏõọốồổỗộớờởỡợ"
        "úùủũụứừửữựýỳỷỹỵ"
    )

    normalized = text.lower()

    if any(char in vietnamese_chars for char in normalized):
        return "Vietnamese"

    return "English"


def run(
    topic: str,
    as_of: Optional[str] = None,
):

    # Nếu caller không truyền ngày,
    # lấy ngày hiện tại.
    if as_of is None:
        as_of = date.today().isoformat()

    # --------------------------------------------------------
    # Giữ folder images/ cho tương lai.
    #
    # Phiên bản hiện tại chỉ nhúng URL ảnh từ Wikimedia vào Markdown,
    # nên folder này chưa dùng để lưu file ảnh.
    # mkdir(exist_ok=True) đảm bảo:
    #     - Mỗi lần run folder được tạo nếu chưa có.
    #     - Nếu folder đã tồn tại thì không làm gì thêm.
    # --------------------------------------------------------
    Path("images").mkdir(
        parents=True,
        exist_ok=True,
    )

    # Xác định ngôn ngữ đầu ra một lần ở đầu workflow.
    # Sau đó truyền xuống Planner/Worker/Image planner để tránh lẫn ngôn ngữ.
    language = detect_output_language(topic)

    # --------------------------------------------------------
    # IMPORTANT
    #
    # State hiện tại không dùng as_of.
    #
    # Vì vậy KHÔNG truyền as_of vào app.invoke()
    # cho tới khi bạn thực sự thêm nó vào State.
    # --------------------------------------------------------

    initial_state = {

        "topic": topic,

        # Ngôn ngữ được giữ xuyên suốt toàn bộ graph.
        "language": language,

        "mode": "",

        "needs_research": False,

        "queries": [],

        "evidence": [],

        "plan": None,

        "sections": [],

        "merged_md": "",

        "md_with_placeholders": "",

        "image_specs": [],

        "final": "",
    }

    out = app.invoke(
        initial_state,
        config=config,
    )

    return out