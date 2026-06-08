from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
STATIC_DIR = APP_DIR / "static"
RESPONSES_DIR = APP_DIR / "responses"
RESPONSES_FILE = RESPONSES_DIR / "responses.jsonl"


app = FastAPI(title="Cleanliness User Test")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/dataset", StaticFiles(directory=DATASET_DIR), name="dataset")


class ResponsePayload(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    participant_id: str | None = Field(default=None, max_length=128)
    image_id: str = Field(min_length=1, max_length=128)
    group_id: int = Field(ge=0)
    dirty_variant: int = Field(ge=2, le=4)
    clean_image: str = Field(min_length=1)
    dirty_image: str = Field(min_length=1)
    cleanliness_score: int = Field(ge=1, le=5)
    cleaning_needed: bool
    response_ms: int | None = Field(default=None, ge=0)
    trial_index: int | None = Field(default=None, ge=0)
    trial_count: int | None = Field(default=None, ge=1)


def parse_dataset() -> list[dict[str, Any]]:
    if not DATASET_DIR.exists():
        raise HTTPException(status_code=500, detail=f"Dataset directory not found: {DATASET_DIR}")

    groups: dict[int, dict[str, Any]] = {}

    for path in DATASET_DIR.glob("*.jpg"):
        stem_parts = path.stem.split("_")
        if len(stem_parts) != 3:
            continue

        number, image_type, variant = stem_parts
        if not number.isdigit() or image_type not in {"clean", "dirty"} or not variant.isdigit():
            continue

        group_id = int(number)
        variant_id = int(variant)
        groups.setdefault(group_id, {"clean": None, "dirty": {}})

        if image_type == "clean" and variant_id == 1:
            groups[group_id]["clean"] = path.name
        elif image_type == "dirty":
            groups[group_id]["dirty"][variant_id] = path.name

    trials: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        clean_image = groups[group_id]["clean"]
        if not clean_image:
            continue

        for dirty_variant in sorted(groups[group_id]["dirty"]):
            dirty_image = groups[group_id]["dirty"][dirty_variant]
            trials.append(
                {
                    "image_id": f"{group_id}_dirty_{dirty_variant:02d}",
                    "group_id": group_id,
                    "dirty_variant": dirty_variant,
                    "clean_image": clean_image,
                    "dirty_image": dirty_image,
                    "clean_url": f"/dataset/{clean_image}",
                    "dirty_url": f"/dataset/{dirty_image}",
                }
            )

    return trials


def read_responses() -> list[dict[str, Any]]:
    if not RESPONSES_FILE.exists():
        return []

    rows: list[dict[str, Any]] = []
    with RESPONSES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sample_stddev(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def calculate_threshold(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    labelled = [
        (int(row["cleanliness_score"]), bool(row["cleaning_needed"]))
        for row in rows
        if "cleanliness_score" in row and "cleaning_needed" in row
    ]

    if not labelled:
        return None

    candidates: list[dict[str, Any]] = []
    for threshold in range(1, 6):
        correct = 0
        for score, needed in labelled:
            predicted_needed = score <= threshold
            if predicted_needed == needed:
                correct += 1

        candidates.append(
            {
                "threshold": threshold,
                "accuracy": correct / len(labelled),
                "correct": correct,
                "total": len(labelled),
            }
        )

    # Lower cleanliness scores mean "needs cleaning"; tie-break toward simpler stricter thresholds.
    return max(candidates, key=lambda item: (item["accuracy"], -item["threshold"]))


def summarize_responses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image_id"]), []).append(row)

    images: list[dict[str, Any]] = []
    for image_id in sorted(grouped):
        image_rows = grouped[image_id]
        scores = [int(row["cleanliness_score"]) for row in image_rows]
        cleaning_needed_count = sum(1 for row in image_rows if row["cleaning_needed"])
        images.append(
            {
                "image_id": image_id,
                "group_id": image_rows[0].get("group_id"),
                "dirty_variant": image_rows[0].get("dirty_variant"),
                "dirty_image": image_rows[0].get("dirty_image"),
                "responses": len(image_rows),
                "median_cleanliness": statistics.median(scores),
                "stddev_cleanliness": sample_stddev(scores),
                "cleaning_needed_count": cleaning_needed_count,
                "cleaning_not_needed_count": len(image_rows) - cleaning_needed_count,
                "cleaning_needed_ratio": cleaning_needed_count / len(image_rows),
            }
        )

    return {
        "total_responses": len(rows),
        "image_count": len(images),
        "images": images,
        "threshold": calculate_threshold(rows),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/trials")
def get_trials() -> dict[str, Any]:
    trials = parse_dataset()
    return {"trials": trials, "count": len(trials)}


@app.post("/api/responses")
def save_response(payload: ResponsePayload) -> dict[str, Any]:
    known_image_ids = {trial["image_id"] for trial in parse_dataset()}
    if payload.image_id not in known_image_ids:
        raise HTTPException(status_code=400, detail=f"Unknown image_id: {payload.image_id}")

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    row = payload.dict()
    row["created_at"] = datetime.now(timezone.utc).isoformat()

    with RESPONSES_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"ok": True}


@app.get("/api/summary")
def get_summary() -> dict[str, Any]:
    return summarize_responses(read_responses())


@app.get("/api/export.csv")
def export_csv() -> PlainTextResponse:
    rows = read_responses()
    if not rows:
        return PlainTextResponse("", media_type="text/csv")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    lines: list[str] = []
    writer_buffer = _CsvStringWriter(lines)
    writer = csv.DictWriter(writer_buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    return PlainTextResponse("".join(lines), media_type="text/csv")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "dataset_dir": str(DATASET_DIR),
        "trial_count": len(parse_dataset()),
        "response_file": str(RESPONSES_FILE),
    }


class _CsvStringWriter:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def write(self, value: str) -> int:
        self.lines.append(value)
        return len(value)
