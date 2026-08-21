import re
from pathlib import Path
from typing import Optional

import requests

from .state import State

# ============================================================
# 11.3 WEB IMAGE SEARCH
# ============================================================
#
# Trước đây project dùng:
#
#     Gemini Image Model
#             ↓
#        generate image
#
# Nhưng image generation có thể ăn quota khá nhanh.
#
# Phiên bản này:
#
#     Gemini
#        ↓
#     image_query
#        ↓
#     Wikimedia Commons API
#        ↓
#     image URL
#        ↓
#     Markdown
#
# Ưu điểm:
#     - Không tốn image-generation quota.
#     - Không cần thêm image API key.
#     - Có source URL để người đọc kiểm tra nguồn.
#
# Lưu ý:
#     Wikimedia Commons có nhiều license khác nhau.
#     Vì vậy ta giữ lại source/credit trong Markdown
#     thay vì giả định mọi ảnh đều tự do sử dụng.
#
# ============================================================


def _wikimedia_search_image(
    query: str,
) -> Optional[dict]:
    """
    Tìm một hình ảnh trên Wikimedia Commons.

    Wikimedia API không yêu cầu API key cho kiểu search này.

    Return:
        {
            "image_url": "...",
            "source_url": "...",
            "title": "...",
            "credit": "..."
        }

    hoặc None nếu không tìm thấy.
    """

    api_url = (
        "https://commons.wikimedia.org/w/api.php"
    )

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 5,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 1400,
        "format": "json",
    }

    response = requests.get(
        api_url,
        params=params,
        timeout=20,
        headers={
            "User-Agent":
                "AgenticWriter/1.0"
        },
    )

    response.raise_for_status()

    data = response.json()

    pages = (
        data.get("query", {})
        .get("pages", {})
    )

    if not pages:
        return None

    # Chọn kết quả đầu tiên có thumbnail/image URL.
    for page in pages.values():

        imageinfo = (
            page.get("imageinfo") or []
        )

        if not imageinfo:
            continue

        info = imageinfo[0]

        image_url = (
            info.get("thumburl")
            or info.get("url")
        )

        if not image_url:
            continue

        title = page.get(
            "title",
            "Wikimedia Commons",
        )

        source_url = (
            "https://commons.wikimedia.org/wiki/"
            + title.replace(" ", "_")
        )

        metadata = info.get(
            "extmetadata",
            {},
        )

        artist = (
            metadata.get("Artist", {})
            .get("value")
        )

        license_name = (
            metadata.get("LicenseShortName", {})
            .get("value")
        )

        credit_parts = []

        if artist:
            # Metadata đôi khi chứa HTML.
            artist_clean = re.sub(
                r"<[^>]+>",
                "",
                artist,
            ).strip()

            if artist_clean:
                credit_parts.append(
                    artist_clean
                )

        if license_name:
            credit_parts.append(
                license_name
            )

        credit = " — ".join(
            credit_parts
        ) or "Wikimedia Commons"

        return {
            "image_url": image_url,
            "source_url": source_url,
            "title": title,
            "credit": credit,
        }

    return None


# ============================================================
# 11.4 FIND + PLACE WEB IMAGES
# ============================================================

def generate_and_place_images(
    state: State,
) -> dict:

    plan = state["plan"]

    assert plan is not None

    md = (
        state.get(
            "md_with_placeholders"
        )
        or state["merged_md"]
    )

    image_specs = (
        state.get(
            "image_specs",
            []
        )
        or []
    )

    # --------------------------------------------------------
    # Nếu LLM quyết định không cần ảnh
    # --------------------------------------------------------

    if not image_specs:

        filename = (
            safe_filename(
                plan.blog_title
            )
            + ".md"
        )

        Path(filename).write_text(
            md,
            encoding="utf-8",
        )

        print(
            f"\n📄 Markdown saved: "
            f"{Path(filename).resolve()}"
        )

        return {
            "final": md
        }

    # --------------------------------------------------------
    # Tạo thư mục images/
    #
    # Hiện tại ta không bắt buộc download ảnh.
    # Folder này chỉ được giữ lại để tương thích
    # với cấu trúc project cũ.
    # --------------------------------------------------------

    images_dir = Path(
        "images"
    )

    images_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Search từng image trên web.
    #
    # Tối đa 3 ảnh nên chạy tuần tự là đủ.
    # Nếu sau này tăng số lượng ảnh,
    # có thể fan-out phần này bằng ThreadPoolExecutor.
    # --------------------------------------------------------

    for spec in image_specs:

        placeholder = (
            spec["placeholder"]
        )

        query = (
            spec.get("image_query")
            or spec.get("alt")
            or plan.blog_title
        )

        try:

            result = (
                _wikimedia_search_image(
                    query
                )
            )

        except Exception as e:

            result = None

            print(
                f"⚠️ Image search failed "
                f"for '{query}': {e}"
            )

        # ----------------------------------------------------
        # Nếu không tìm thấy ảnh:
        #
        # Blog vẫn được tạo.
        # Không để image search làm chết workflow.
        # ----------------------------------------------------

        if not result:

            fallback = (
                "> **[IMAGE NOT FOUND]** "
                f"{spec.get('caption', '')}\n"
                ">\n"
                f"> **Search:** {query}\n"
            )

            md = md.replace(
                placeholder,
                fallback,
            )

            continue

        image_url = result["image_url"]
        source_url = result["source_url"]
        credit = result["credit"]

        # ----------------------------------------------------
        # Lưu thông tin vào spec để debug / inspect state.
        # ----------------------------------------------------

        spec["image_url"] = image_url
        spec["source_url"] = source_url
        spec["credit"] = credit

        # ----------------------------------------------------
        # Thay placeholder bằng Markdown image.
        #
        # Ảnh được load trực tiếp từ URL trên web.
        # Không cần Gemini image generation.
        # ----------------------------------------------------

        img_md = (
            f"![{spec['alt']}]"
            f"({image_url})\n"
            f"*{spec['caption']}*\n"
            f"\n"
            f"*Source: [{credit}]({source_url})*"
        )

        md = md.replace(
            placeholder,
            img_md,
        )

        print(
            f"🖼️ Image found: "
            f"{spec.get('filename', query)}"
        )

    # --------------------------------------------------------
    # Save final Markdown vào folder blog/
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parent.parent
    )

    blog_dir = (
        project_root / "blog"
    )

    blog_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        safe_filename(
            plan.blog_title
        )
        + ".md"
    )

    output_path = (
        blog_dir / filename
    )

    output_path.write_text(
        md,
        encoding="utf-8",
    )

    print(
        f"\n📄 Final Markdown saved:"
        f"\n{output_path.resolve()}"
    )

    return {
        "final": md
    }


# ============================================================
# 12. SAFE FILENAME
# ============================================================
#
# Đây là phần mình cố tình thêm.
#
# Trước đây title có thể là:
#
#     Understanding Machine Learning:
#     A Beginner's Roadmap
#
# Windows không cho phép ':' trong filename.
#
# Vì vậy phải sanitize title trước khi tạo file.
#
# ============================================================

def safe_filename(
    title: str,
) -> str:

    # Xóa ký tự không hợp lệ trên Windows
    filename = re.sub(
        r'[<>:"/\\|?*]',
        "",
        title,
    )

    # Thay nhiều khoảng trắng bằng _
    filename = re.sub(
        r"\s+",
        "_",
        filename,
    )

    return filename.lower()