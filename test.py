from agent import run

result = run(
    "Giải thích kiến trúc Transformer và cho biết những cải tiến đáng chú ý của các phiên bản Transformer gần đây."
)

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(
    result.get(
        "final",
        ""
    )
)