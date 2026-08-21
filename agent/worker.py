from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send

from .models import groq_worker
from .prompts import WORKER_SYSTEM
from .schemas import EvidenceItem, Plan, Task
from .state import State

# ============================================================
# 9. FAN-OUT
# ============================================================
#
# Đây là một phần rất quan trọng của LangGraph.
#
# Plan:
#
#     Task 1
#     Task 2
#     Task 3
#     Task 4
#
# fanout biến nó thành:
#
#     Worker(Task 1)
#     Worker(Task 2)
#     Worker(Task 3)
#     Worker(Task 4)
#
# Các worker có thể chạy song song.
#
# ============================================================

def fanout(state: State):

    plan = state["plan"]

    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),

                "topic": state["topic"],

                "mode": state["mode"],

                # Truyền ngôn ngữ xuống từng Worker để tránh language drift.
                "language": state.get("language", "English"),

                "plan": plan.model_dump(),

                "evidence": [
                    e.model_dump()
                    for e in state.get(
                        "evidence",
                        []
                    )
                ],
            },
        )

        for task in plan.tasks
    ]


# ============================================================
# 10. WORKER
# ============================================================
#
# Mỗi worker chỉ chịu trách nhiệm:
#
#     1 section
#
# Không viết toàn bộ blog.
#
# Đây là tư tưởng:
#
#     Planner → phân chia công việc
#     Worker  → thực thi từng phần
#
# ============================================================

def worker_node(payload: dict) -> dict:

    # --------------------------------------------------------
    # Deserialize payload
    #
    # Send() truyền dictionary,
    # nên ta convert ngược về Pydantic model.
    # --------------------------------------------------------

    task = Task(
        **payload["task"]
    )

    plan = Plan(
        **payload["plan"]
    )

    evidence = [
        EvidenceItem(**e)
        for e in payload.get(
            "evidence",
            []
        )
    ]

    topic = payload["topic"]

    # Ngôn ngữ được truyền từ State xuống Worker.
    language = payload.get(
        "language",
        "English"
    )

    mode = payload.get(
        "mode",
        "closed_book"
    )

    # --------------------------------------------------------
    # Format bullets
    # --------------------------------------------------------

    bullets_text = (
        "\n- "
        + "\n- ".join(task.bullets)
    )

    # --------------------------------------------------------
    # Format evidence
    # --------------------------------------------------------

    evidence_text = ""

    if evidence:

        evidence_text = "\n".join(
            (
                f"- {e.title} | "
                f"{e.url} | "
                f"{e.published_at or 'date:unknown'}"
            ).strip()

            for e in evidence[:20]
        )

    # --------------------------------------------------------
    # Generate section
    # --------------------------------------------------------

    # Worker có nhiều task nhỏ và chạy fan-out,
    # nên giao cho Groq vì model này có tốc độ token rất cao.
    response = groq_worker.invoke(
        [
            SystemMessage(
                content=WORKER_SYSTEM
            ),

            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Output language: {language}\n"
                    f"Mode: {mode}\n\n"

                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: "
                    f"{task.requires_research}\n"
                    f"requires_citations: "
                    f"{task.requires_citations}\n"
                    f"requires_code: "
                    f"{task.requires_code}\n"

                    f"Bullets:"
                    f"{bullets_text}\n\n"

                    "Evidence "
                    "(ONLY use these URLs when citing):\n"
                    f"{evidence_text}\n"
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # AIMessage.content thường là str,
    # nhưng một số provider có thể trả list.
    #
    # Ta normalize để worker không bị:
    #
    #     AttributeError:
    #     'list' object has no attribute 'strip'
    #
    # --------------------------------------------------------

    content = response.content

    if isinstance(content, list):

        section_md = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    else:

        section_md = content

    section_md = section_md.strip()

    print(
        f"\n✍️ Worker finished: "
        f"Task {task.id} - {task.title}"
    )

    # --------------------------------------------------------
    # QUAN TRỌNG:
    #
    # task.id được trả về cùng section.
    #
    # Reducer sau đó sort theo task.id
    # để đảm bảo thứ tự blog không bị đảo.
    # --------------------------------------------------------

    return {
        "sections": [
            (
                task.id,
                section_md
            )
        ]
    }