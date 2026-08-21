from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from .models import gemini_fast
from .prompts import RESEARCH_SYSTEM
from .schemas import EvidenceItem, EvidencePack
from .state import State

# ============================================================
# 6. TAVILY RESEARCH
# ============================================================

def _tavily_search(
    query: str,
    max_results: int = 5,
) -> List[dict]:
    """
    Chạy một Tavily search và normalize kết quả.

    Normalize rất quan trọng vì:
    raw result của tool có thể có format khác nhau.

    Ta chuyển tất cả về format thống nhất.
    """

    tool = TavilySearchResults(
        max_results=max_results
    )

    results = tool.invoke(
        {
            "query": query
        }
    )

    normalized: List[dict] = []

    for r in results or []:

        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": (
                    r.get("content")
                    or r.get("snippet")
                    or ""
                ),
                "published_at": (
                    r.get("published_date")
                    or r.get("published_at")
                ),
                "source": r.get("source"),
            }
        )

    return normalized


# ============================================================
# 7. RESEARCH SYNTHESIZER
# ============================================================

def research_node(state: State) -> dict:

    queries = (
        state.get("queries", [])
        or []
    )

    max_results = 6

    raw_results: List[dict] = []

    # --------------------------------------------------------
    # Chạy các query song song.
    #
    # Trước đây:
    #
    #     query 1 → search
    #     query 2 → search
    #     query 3 → search
    #
    # Bây giờ dùng ThreadPoolExecutor để các request I/O
    # tới Tavily có thể chạy đồng thời.
    #
    # Đây là một tối ưu latency quan trọng vì research
    # không cần phải chờ query trước hoàn thành.
    # --------------------------------------------------------

    if queries:

        with ThreadPoolExecutor(
            max_workers=min(5, len(queries))
        ) as executor:

            futures = {
                executor.submit(
                    _tavily_search,
                    query,
                    max_results,
                ): query
                for query in queries
            }

            for future in as_completed(futures):

                query = futures[future]

                try:
                    raw_results.extend(
                        future.result()
                    )

                except Exception as e:

                    print(
                        f"⚠️ Research failed for query "
                        f"'{query}': {e}"
                    )

    # Không có kết quả
    if not raw_results:

        print("\n⚠️ No research results.")

        return {
            "evidence": []
        }

    print(
        f"\n🔎 Raw research results: "
        f"{len(raw_results)}"
    )

    # --------------------------------------------------------
    # LLM tổng hợp evidence
    #
    # Đây là task structured-output tương đối nhẹ,
    # nên giao cho Gemini Flash-Lite thay vì model chính.
    # --------------------------------------------------------

    extractor = gemini_fast.with_structured_output(
        EvidencePack
    )

    pack = extractor.invoke(
        [
            SystemMessage(
                content=RESEARCH_SYSTEM
            ),

            HumanMessage(
                content=(
                    f"Raw results:\n"
                    f"{raw_results}"
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # Deduplicate theo URL
    # --------------------------------------------------------

    dedup = {}

    for evidence in pack.evidence:

        if evidence.url:
            dedup[evidence.url] = evidence

    evidence = list(
        dedup.values()
    )

    print(
        f"✅ Evidence after dedup: "
        f"{len(evidence)}"
    )

    return {
        "evidence": evidence
    }