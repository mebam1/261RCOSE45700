import os
from PIL import Image

from table_config import TABLE_AREAS


FRAME_DIR = "frames"
CROP_DIR = "table_crops"


def extract_timestamp_from_filename(filename: str) -> str:
    """
    frame_0000.jpg -> 0000
    frame_0010.jpg -> 0010
    """
    name_without_ext = os.path.splitext(filename)[0]

    if "_" in name_without_ext:
        return name_without_ext.split("_")[-1]

    return name_without_ext


def crop_tables_from_frames():
    os.makedirs(CROP_DIR, exist_ok=True)

    frame_files = [
        file for file in os.listdir(FRAME_DIR)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    frame_files.sort()

    if not frame_files:
        print("frames 폴더 안에 CCTV 이미지가 없습니다.")
        return

    for frame_file in frame_files:
        frame_path = os.path.join(FRAME_DIR, frame_file)
        timestamp = extract_timestamp_from_filename(frame_file)

        image = Image.open(frame_path)

        print(f"프레임 처리 중: {frame_file}")

        for table_id, box in TABLE_AREAS.items():
            x1, y1, x2, y2 = box

            cropped = image.crop((x1, y1, x2, y2))

            crop_filename = f"{table_id}_{timestamp}.jpg"
            crop_path = os.path.join(CROP_DIR, crop_filename)

            cropped.save(crop_path)

            print(f"  저장: {crop_path}")

    print("테이블별 crop 완료")


if __name__ == "__main__":
    crop_tables_from_frames()