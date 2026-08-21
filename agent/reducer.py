from langchain_core.messages import SystemMessage, HumanMessage

from .models import gemini_fast
from .prompts import DECIDE_IMAGES_SYSTEM
from .schemas import GlobalImagePlan
from .state import State

# ============================================================
# 11.1 MERGE CONTENT
# ============================================================

def merge_content(state: State) -> dict:

    plan = state["plan"]

    # --------------------------------------------------------
    # Worker chạy song song nên thứ tự result
    # không nhất thiết giống Task ID.
    #
    # Ví dụ:
    #
    # [(3, "..."), (1, "..."), (2, "...")]
    #
    # Sort lại:
    #
    # [(1, "..."), (2, "..."), (3, "...")]
    # --------------------------------------------------------

    ordered_sections = [
        md
        for _, md in sorted(
            state["sections"],
            key=lambda x: x[0]
        )
    ]

    body = "\n\n".join(
        ordered_sections
    ).strip()

    merged_md = (
        f"# {plan.blog_title}\n\n"
        f"{body}\n"
    )

    return {
        "merged_md": merged_md
    }


# ============================================================
# 11.2 DECIDE IMAGES
# ============================================================

def decide_images(state: State) -> dict:

    # Image planning là task nhẹ,
    # nên dùng Gemini Flash-Lite.
    planner = gemini_fast.with_structured_output(
        GlobalImagePlan
    )

    merged_md = state["merged_md"]

    plan = state["plan"]

    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(
                content=DECIDE_IMAGES_SYSTEM
            ),

            HumanMessage(
                content=(
                    f"Blog kind: "
                    f"{plan.blog_kind}\n"
                    f"Topic: "
                    f"{state['topic']}\n"
                    f"Output language: "
                    f"{state.get('language', 'English')}\n\n"

                    "Insert placeholders + "
                    "propose web image search queries.\n\n"

                    f"{merged_md}"
                )
            ),
        ]
    )

    print("\n========== IMAGE PLAN ==========")

    print(
        image_plan.model_dump_json(
            indent=2
        )
    )

    return {
        "md_with_placeholders": (
            image_plan.md_with_placeholders
        ),

        "image_specs": [
            image.model_dump()
            for image in image_plan.images
        ],
    }