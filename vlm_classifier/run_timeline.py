import csv
import os
import time

from classify_frame import classify_frame
from state_tracker import RestaurantStateTracker


IMAGE_DIR = "images"
RESULT_CSV = "results/timeline_results.csv"

# CCTV 캡처 간격.
# 예: 10초마다 캡처했다면 10
FRAME_INTERVAL_SECONDS = 10

# 사람이 없는 상태가 몇 초 이상 지속되면 AFTER_MEAL로 볼지.
# 600초 = 10분
EMPTY_THRESHOLD_SECONDS = 600

# API 호출 사이 대기 시간
API_WAIT_SECONDS = 10

# API 오류 발생 시 재시도 횟수
MAX_RETRIES = 3

# 재시도 전 대기 시간
RETRY_WAIT_SECONDS = 30


def classify_with_retry(image_path):
    """
    Gemini API 호출 중 503, 429 등의 일시적 오류가 발생하면
    일정 시간 기다린 뒤 다시 시도한다.
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

    image_files = [
        file for file in os.listdir(IMAGE_DIR)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_files.sort()

    if not image_files:
        print("images 폴더 안에 이미지 파일이 없습니다.")
        return

    tracker = RestaurantStateTracker(
        empty_threshold_seconds=EMPTY_THRESHOLD_SECONDS
    )

    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "index",
            "timestamp_seconds",
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

        for index, filename in enumerate(image_files):
            timestamp_seconds = index * FRAME_INTERVAL_SECONDS
            image_path = os.path.join(IMAGE_DIR, filename)

            print(f"분석 중: {filename} / t={timestamp_seconds}s")

            frame_result = classify_with_retry(image_path)
            temporal_result = tracker.update(timestamp_seconds, frame_result)

            writer.writerow([
                index,
                timestamp_seconds,
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

            if index < len(image_files) - 1:
                print(f"{API_WAIT_SECONDS}초 대기 후 다음 이미지 분석...")
                time.sleep(API_WAIT_SECONDS)

    print(f"결과 저장 완료: {RESULT_CSV}")


if __name__ == "__main__":
    main()