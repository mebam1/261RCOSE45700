import csv
import os
import time
from collections import defaultdict

from classify_frame import classify_frame
from state_tracker import RestaurantStateTracker


CROP_DIR = "table_crops"
RESULT_CSV = "results/multi_table_results.csv"

# CCTV 캡처 간격
FRAME_INTERVAL_SECONDS = 10

# 사람이 없는 상태가 몇 초 이상 지속되면 AFTER_MEAL로 볼지
# 600초 = 10분
EMPTY_THRESHOLD_SECONDS = 600

# API 호출 사이 대기 시간
API_WAIT_SECONDS = 10

# API 오류 발생 시 재시도 횟수
MAX_RETRIES = 3

# 재시도 전 대기 시간
RETRY_WAIT_SECONDS = 30


def parse_table_crop_filename(filename: str):
    """
    T01_0000.jpg -> ("T01", "0000")
    T02_0010.jpg -> ("T02", "0010")
    """
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split("_")

    if len(parts) < 2:
        return None, None

    table_id = parts[0]
    timestamp_text = parts[1]

    return table_id, timestamp_text


def timestamp_to_seconds(timestamp_text: str) -> int:
    try:
        return int(timestamp_text)
    except ValueError:
        return 0


def classify_with_retry(image_path: str):
    """
    Gemini API 호출 중 429, 503 등의 오류가 발생하면
    일정 시간 기다렸다가 다시 시도한다.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return classify_frame(image_path)

        except Exception as e:
            print(f"API 오류 발생: {e}")
            print(f"재시도 {attempt}/{MAX_RETRIES}")

            if attempt < MAX_RETRIES:
                print(f"{RETRY_WAIT_SECONDS}초 대기 후 재시도합니다.")
                time.sleep(RETRY_WAIT_SECONDS)
            else:
                print("최대 재시도 횟수를 초과하여 UNCERTAIN으로 처리합니다.")

                return {
                    "visible_people": False,
                    "active_dining": False,
                    "cleaning_action": False,
                    "used_dishes_or_leftovers": False,
                    "personal_items_or_food_remaining": False,
                    "frame_state": "UNCERTAIN",
                    "confidence": 0.0,
                    "reason": f"API 오류로 인해 판정하지 못함: {e}"
                }


def main():
    os.makedirs("results", exist_ok=True)

    crop_files = [
        file for file in os.listdir(CROP_DIR)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not crop_files:
        print("table_crops 폴더 안에 crop 이미지가 없습니다.")
        print("먼저 python crop_tables.py 를 실행하세요.")
        return

    parsed_items = []

    for filename in crop_files:
        table_id, timestamp_text = parse_table_crop_filename(filename)

        if table_id is None:
            print(f"파일명 형식이 맞지 않아 건너뜀: {filename}")
            continue

        timestamp_seconds = timestamp_to_seconds(timestamp_text)

        parsed_items.append({
            "filename": filename,
            "table_id": table_id,
            "timestamp_seconds": timestamp_seconds
        })

    # 시간 순서 우선, 같은 시간 안에서는 table_id 순서
    parsed_items.sort(key=lambda item: (item["timestamp_seconds"], item["table_id"]))

    # table_id별 상태 추적기
    trackers = defaultdict(
        lambda: RestaurantStateTracker(
            empty_threshold_seconds=EMPTY_THRESHOLD_SECONDS
        )
    )

    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "timestamp_seconds",
            "table_id",
            "filename",
            "frame_state",
            "final_state",
            "visible_people",
            "active_dining",
            "cleaning_action",
            "used_dishes_or_leftovers",
            "personal_items_or_food_remaining",
            "confidence",
            "frame_reason",
            "temporal_reason"
        ])

        for index, item in enumerate(parsed_items):
            filename = item["filename"]
            table_id = item["table_id"]
            timestamp_seconds = item["timestamp_seconds"]

            image_path = os.path.join(CROP_DIR, filename)

            print(f"분석 중: table={table_id}, file={filename}, t={timestamp_seconds}s")

            frame_result = classify_with_retry(image_path)

            tracker = trackers[table_id]
            temporal_result = tracker.update(timestamp_seconds, frame_result)

            writer.writerow([
                timestamp_seconds,
                table_id,
                filename,
                frame_result.get("frame_state", "UNCERTAIN"),
                temporal_result.get("final_state", "UNCERTAIN"),
                frame_result.get("visible_people", False),
                frame_result.get("active_dining", False),
                frame_result.get("cleaning_action", False),
                frame_result.get("used_dishes_or_leftovers", False),
                frame_result.get("personal_items_or_food_remaining", False),
                frame_result.get("confidence", 0.0),
                frame_result.get("reason", ""),
                temporal_result.get("temporal_reason", "")
            ])

            if index < len(parsed_items) - 1:
                print(f"{API_WAIT_SECONDS}초 대기 후 다음 이미지 분석...")
                time.sleep(API_WAIT_SECONDS)

    print(f"결과 저장 완료: {RESULT_CSV}")


if __name__ == "__main__":
    main()