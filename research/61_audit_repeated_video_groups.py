
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "datasets" / "engagement" / "processed"
    / "engagement_all_330.csv"
)

OUTPUT_PATH = (
    ROOT / "reports" / "engagement"
    / "repeated_video_groups.csv"
)


def main():
    df = pd.read_csv(DATA_PATH)

    real = df[df["dataset_type"] == "real_world"].copy()

    real["video_title"] = (
        real["video_title"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    grouped = (
        real.groupby("video_title")
        .agg(
            comment_count=("comment_id", "count"),
            intents=("intent", lambda x: ", ".join(sorted(set(x)))),
            intent_count=("intent", "nunique"),
            comment_ids=("comment_id", lambda x: ", ".join(x)),
        )
        .reset_index()
    )

    repeated = (
        grouped[grouped["comment_count"] > 1]
        .sort_values(
            ["comment_count", "video_title"],
            ascending=[False, True],
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    repeated.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nMODERAI — REPEATED VIDEO GROUP AUDIT")
    print("=" * 60)

    print("Unique video titles:", real["video_title"].nunique())
    print("Repeated video titles:", len(repeated))
    print(
        "Comments belonging to repeated titles:",
        repeated["comment_count"].sum(),
    )

    print("\nRepeated video groups:")
    if repeated.empty:
        print("No repeated titles found.")
    else:
        print(
            repeated[
                ["video_title", "comment_count", "intents", "intent_count"]
            ].to_string(index=False)
        )

    print("\nSaved report:")
    print(OUTPUT_PATH)

    print("\nAudit completed.")


if __name__ == "__main__":
    main()
