from typing import List, Optional, Literal

from pydantic import BaseModel, Field


# ============================================================
# 2. PYDANTIC SCHEMAS
# ============================================================
#
# Đây là "hợp đồng dữ liệu" giữa LLM và workflow.
#
# Thay vì để LLM trả JSON tự do:
#
#     {"title": "...", "abc": "..."}
#
# ta ép nó phải tuân theo schema:
#
#     Plan
#       └── tasks
#             ├── Task
#             ├── Task
#             └── ...
#
# Đây là nền tảng để dùng:
#
#     llm.with_structured_output(...)
#
# ============================================================


class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description=(
            "One sentence describing what the reader "
            "should be able to do/understand after this section."
        ),
    )

    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description=(
            "3–6 concrete, non-overlapping subpoints "
            "to cover in this section."
        ),
    )

    target_words: int = Field(
        ...,
        description="Target word count for this section (120–550).",
    )

    tags: List[str] = Field(default_factory=list)

    # Section này có cần thông tin từ web không?
    requires_research: bool = False

    # Section này có bắt buộc citation không?
    requires_citations: bool = False

    # Section này có cần code example không?
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str

    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design",
    ] = "explainer"

    constraints: List[str] = Field(default_factory=list)

    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str

    # Có thể Tavily không trả published date.
    published_at: Optional[str] = None

    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    # Có cần research không?
    needs_research: bool

    # closed_book / hybrid / open_book
    mode: Literal[
        "closed_book",
        "hybrid",
        "open_book",
    ]

    # Các query gửi cho Tavily
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(
        default_factory=list
    )


class ImageSpec(BaseModel):
    # Placeholder xuất hiện trong Markdown
    #
    # Ví dụ:
    #     [[IMAGE_1]]
    placeholder: str = Field(
        ...,
        description="e.g. [[IMAGE_1]]",
    )

    # Tên file image local nếu muốn lưu metadata/cache.
    filename: str = Field(
        ...,
        description="Save under images/, e.g. self_attention.jpg",
    )

    alt: str
    caption: str

    # Thay vì prompt gửi Image Model,
    # bây giờ đây là từ khóa dùng để tìm ảnh trên web.
    image_query: str = Field(
        ...,
        description="Specific web image search query.",
    )

    # Các field dưới đây được điền sau khi search.
    image_url: Optional[str] = None
    source_url: Optional[str] = None

    # Nguồn ảnh / license nếu API tìm được.
    credit: Optional[str] = None


class GlobalImagePlan(BaseModel):
    # Markdown sau khi LLM chèn [[IMAGE_1]], ...
    md_with_placeholders: str

    # Danh sách image cần generate
    images: List[ImageSpec] = Field(
        default_factory=list
    )