# make_dirty_images.py

import argparse
import base64
import getpass
import os
import random
import tempfile
from pathlib import Path
from typing import Sequence

from PIL import Image
from openai import OpenAI


DEFAULT_DIRECTORY = r"D:\SW\mvp1\data\dataset"
IMAGE_VARIANTS = 4
CLEAN_VARIANT_INDEX = 1
CLEAN_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DIRTY_VARIANT_GENERATION_RULES = {
    2: {"object_types": 1, "total_count": 1},
    3: {"object_types": 2, "total_count": 2},
    4: {"object_types": 3, "total_count": 3},
}


# 물체 종류는 API 호출 직전에 랜덤으로 선택하고, 개수는 variant 규칙을 따릅니다.
DIRTY_OBJECT_CATALOG = {
    "식기류": [
        {"name": "빈 그릇"},
        {"name": "공기밥 그릇"},
        {"name": "소스 종지"},
        {"name": "컵"},
        {"name": "맥주잔"},
        {"name": "수저"},
        {"name": "젓가락"},
        {"name": "수저 포장지"},
    ],
    "음식 잔여물": [
        {"name": "밥알"},
        {"name": "채소 조각"},
        {"name": "생선 가시"},
        {"name": "달걀 껍데기"},
    ],
    "종이류": [
        {"name": "구겨진 휴지"},
        {"name": "펼쳐진 휴지"},
        {"name": "영수증"},
        {"name": "쿠폰 종이"},
    ],
    "얼룩 계열": [
        {"name": "간장 자국"},
        {"name": "고추장 자국"},
        {"name": "커피 얼룩"},
    ],
    "식당 세팅 계열": [
        {"name": "깨끗한 물병"},
        {"name": "수저통"},
        {"name": "소스 통"},
        {"name": "휴지 통"},
    ],
}

PROMPT_OBJECT_NAMES = {
    "빈 그릇": ("empty bowl", "empty bowls"),
    "공기밥 그릇": ("rice bowl", "rice bowls"),
    "소스 종지": ("small sauce dish", "small sauce dishes"),
    "컵": ("cup", "cups"),
    "맥주잔": ("beer glass", "beer glasses"),
    "수저": ("spoon", "spoons"),
    "젓가락": ("chopstick", "chopsticks"),
    "수저 포장지": (
        "spoon and chopstick paper wrapper",
        "spoon and chopstick paper wrappers",
    ),
    "밥알": ("grain of rice", "grains of rice"),
    "채소 조각": ("small vegetable piece", "small vegetable pieces"),
    "생선 가시": ("fish bone", "fish bones"),
    "달걀 껍데기": ("eggshell piece", "eggshell pieces"),
    "구겨진 휴지": ("crumpled tissue", "crumpled tissues"),
    "펼쳐진 휴지": ("unfolded tissue", "unfolded tissues"),
    "영수증": ("receipt", "receipts"),
    "쿠폰 종이": ("coupon paper", "coupon papers"),
    "간장 자국": ("soy sauce stain", "soy sauce stains"),
    "고추장 자국": ("gochujang sauce stain", "gochujang sauce stains"),
    "커피 얼룩": ("coffee stain", "coffee stains"),
    "깨끗한 물병": ("clean water bottle", "clean water bottles"),
    "수저통": ("cutlery holder", "cutlery holders"),
    "소스 통": ("sauce container", "sauce containers"),
    "휴지 통": ("tissue dispenser", "tissue dispensers"),
}

CLEAN_IMAGE_PROMPT = "Create a clean version of this image without adding any objects."


def flatten_object_catalog() -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []

    for category, entries in DIRTY_OBJECT_CATALOG.items():
        for entry in entries:
            objects.append(
                {
                    "category": category,
                    "name": entry["name"],
                }
            )

    return objects


def choose_dirty_objects(
    rng: random.Random,
    object_type_count: int,
    total_count: int,
) -> list[dict[str, object]]:
    """
    API 호출 직전에 생성할 물체 종류를 랜덤으로 고르고 총 개수를 배분합니다.
    """
    all_objects = flatten_object_catalog()

    if object_type_count < 1:
        raise ValueError("object_type_count는 1 이상이어야 합니다.")

    if total_count < object_type_count:
        raise ValueError("total_count는 object_type_count 이상이어야 합니다.")

    if object_type_count > len(all_objects):
        raise ValueError("요청한 물체 종류 수가 카탈로그의 물체 수보다 많습니다.")

    selected_objects = rng.sample(all_objects, k=object_type_count)
    counts = [1] * object_type_count

    for _ in range(total_count - object_type_count):
        counts[rng.randrange(object_type_count)] += 1

    plan: list[dict[str, object]] = []
    for item, count in zip(selected_objects, counts):
        plan.append(
            {
                "category": item["category"],
                "name": item["name"],
                "count": count,
            }
        )

    return plan


def prompt_object_name(name: str, count: int) -> str:
    object_names = PROMPT_OBJECT_NAMES.get(name)

    if not object_names:
        return name

    singular, plural = object_names
    return singular if count == 1 else plural


def build_edit_prompt(object_plan: Sequence[dict[str, object]]) -> str:
    return " ".join(
        f"Add {item['count']} "
        f"{prompt_object_name(str(item['name']), int(item['count']))} "
        "on the table."
        for item in object_plan
    )


def get_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


def save_as_png_temp(jpg_path: Path) -> Path:
    """
    OpenAI 이미지 편집 입력 안정성을 위해 JPG를 임시 PNG로 변환.
    """
    img = Image.open(jpg_path).convert("RGBA")

    tmp = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )
    tmp_path = Path(tmp.name)
    tmp.close()

    img.save(tmp_path, format="PNG")
    return tmp_path


def edit_image_with_prompt(
    client: OpenAI,
    input_image_path: Path,
    prompt: str,
    model: str = "gpt-image-1",
) -> bytes:
    """
    프롬프트에 따라 원본 이미지를 편집하고 PNG bytes로 반환합니다.
    """
    temp_png_path = save_as_png_temp(input_image_path)

    try:
        with open(temp_png_path, "rb") as image_file:
            result = client.images.edit(
                model=model,
                image=image_file,
                prompt=prompt,
                size="auto",
                n=1,
            )

        image_base64 = result.data[0].b64_json
        return base64.b64decode(image_base64)

    finally:
        temp_png_path.unlink(missing_ok=True)


def edit_clean_image_with_openai(
    client: OpenAI,
    input_image_path: Path,
    model: str = "gpt-image-1",
) -> bytes:
    """
    아무 물체도 추가하지 않는 clean 재생성본을 반환합니다.
    """
    return edit_image_with_prompt(
        client=client,
        input_image_path=input_image_path,
        prompt=CLEAN_IMAGE_PROMPT,
        model=model,
    )


def edit_dirty_image_with_openai(
    client: OpenAI,
    input_image_path: Path,
    object_plan: Sequence[dict[str, object]],
    model: str = "gpt-image-1",
) -> bytes:
    """
    랜덤으로 선택한 물체 계획을 반영한 dirty 이미지를 반환합니다.
    """
    return edit_image_with_prompt(
        client=client,
        input_image_path=input_image_path,
        prompt=build_edit_prompt(object_plan),
        model=model,
    )


def resize_and_save_jpg(
    image_bytes: bytes,
    output_path: Path,
    target_size: tuple[int, int],
    quality: int = 95,
) -> None:
    """
    생성된 이미지를 원본 해상도에 맞추고 JPG로 저장합니다.
    """
    temp_output = output_path.with_suffix(".tmp.png")

    with open(temp_output, "wb") as f:
        f.write(image_bytes)

    try:
        with Image.open(temp_output) as img:
            img = img.convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img.save(output_path, format="JPEG", quality=quality)

    finally:
        temp_output.unlink(missing_ok=True)


def find_clean_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in CLEAN_IMAGE_SUFFIXES
        and path.stem.endswith("_clean")
    )


def clean_to_output_name(clean_path: Path, variant_index: int) -> Path:
    """
    예:
    1_clean.jpg -> 1_clean_01.jpg
    1_clean.jpg -> 1_dirty_02.jpg
    store_clean.png -> store_dirty_04.jpg
    """
    stem = clean_path.stem

    if not stem.endswith("_clean"):
        raise ValueError(f"파일명이 *_clean 형식이 아닙니다: {clean_path}")

    output_stem = stem.removesuffix("_clean")
    if variant_index == CLEAN_VARIANT_INDEX:
        output_name = f"{output_stem}_clean_{variant_index:02d}.jpg"
    else:
        output_name = f"{output_stem}_dirty_{variant_index:02d}.jpg"

    return clean_path.with_name(output_name)


def process_directory(
    directory: Path,
    api_key: str,
    model: str = "gpt-image-1",
    seed: int | None = None,
    overwrite: bool = False,
) -> None:
    client = OpenAI(api_key=api_key)
    rng = random.Random(seed)

    clean_files = find_clean_images(directory)

    if not clean_files:
        print(f"*_clean 이미지 파일을 찾지 못했습니다: {directory}")
        return

    print(
        f"총 {len(clean_files)}개 clean image 처리 시작 "
        f"({IMAGE_VARIANTS}장씩 생성: 1번 clean, 2번 1종/1개, "
        "3번 2종/총2개, 4번 3종/총3개)"
    )

    for clean_path in clean_files:
        original_size = get_image_size(clean_path)

        for variant_index in range(1, IMAGE_VARIANTS + 1):
            output_path = clean_to_output_name(clean_path, variant_index)

            if output_path.exists() and not overwrite:
                print(f"[SKIP] 이미 존재함: {output_path.name}")
                continue

            try:
                if variant_index == CLEAN_VARIANT_INDEX:
                    print(f"[RUN] {clean_path.name} -> {output_path.name}")
                    print("      생성 계획: clean regeneration, no added objects")
                    edited_image_bytes = edit_clean_image_with_openai(
                        client=client,
                        input_image_path=clean_path,
                        model=model,
                    )
                else:
                    generation_rule = DIRTY_VARIANT_GENERATION_RULES[variant_index]
                    object_plan = choose_dirty_objects(
                        rng=rng,
                        object_type_count=generation_rule["object_types"],
                        total_count=generation_rule["total_count"],
                    )
                    object_summary = ", ".join(
                        f"{item['name']} {item['count']}개" for item in object_plan
                    )

                    print(f"[RUN] {clean_path.name} -> {output_path.name}")
                    print(f"      생성 계획: {object_summary}")
                    edited_image_bytes = edit_dirty_image_with_openai(
                        client=client,
                        input_image_path=clean_path,
                        object_plan=object_plan,
                        model=model,
                    )

                resize_and_save_jpg(
                    image_bytes=edited_image_bytes,
                    output_path=output_path,
                    target_size=original_size,
                )

                print(f"[OK] 저장 완료: {output_path}")

            except Exception as e:
                print(f"[ERROR] {clean_path.name} variant {variant_index}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        nargs="?",
        type=str,
        default=DEFAULT_DIRECTORY,
        help="*_clean.jpg/jpeg/png 파일들이 들어 있는 디렉터리 경로",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. 생략하면 입력창에서 받거나 OPENAI_API_KEY 환경변수를 사용합니다.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-image-1",
        help="이미지 편집 모델 이름",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="랜덤 선택 재현이 필요할 때 사용할 seed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 *_clean_01.jpg 또는 *_dirty_XX.jpg가 있어도 덮어쓰기",
    )

    args = parser.parse_args()

    directory = Path(args.directory).expanduser().resolve()

    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"유효한 디렉터리가 아닙니다: {directory}")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        api_key = getpass.getpass("OpenAI API key 입력: ")

    process_directory(
        directory=directory,
        api_key=api_key,
        model=args.model,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
