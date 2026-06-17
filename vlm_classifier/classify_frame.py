import base64
import json
import os
import sys
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)


PROMPT = """
너는 식당 CCTV 테이블 이미지를 분석하는 상황 판정 모델이다.

이미지 한 장만 보고 식사가 완전히 종료되었다고 성급하게 판단하지 말아야 한다.
사람들이 담배, 화장실, 전화 등으로 잠시 자리를 비운 상황이 있을 수 있다.

주어진 이미지를 보고 아래 정보를 JSON으로 판단하라.

1. visible_people:
- 이미지 안에 사람이 보이면 true
- 사람이 전혀 보이지 않으면 false

2. active_dining:
- 사람이 앉아 식사 중이거나 음식/식기를 사용 중이면 true
- 단순히 지나가는 사람만 있거나 식사 행동이 명확하지 않으면 false

3. cleaning_action:
- 사람이 식기를 수거하거나 테이블을 닦거나 쓰레기를 치우는 행동이 보이면 true
- 단순히 서 있거나 지나가는 것만으로는 false

4. used_dishes_or_leftovers:
- 사용한 접시, 컵, 수저, 잔반, 휴지, 쓰레기 등이 테이블 위에 보이면 true
- 깨끗한 빈 테이블이면 false

5. personal_items_or_food_remaining:
- 음식, 술잔, 개인 물품, 가방, 외투 등 손님이 다시 돌아올 가능성을 보여주는 물건이 있으면 true
- 그런 단서가 없으면 false

6. frame_state:
아래 중 하나로 분류하라.

DINING:
- 손님이 앉아 식사 중이거나 식사가 진행 중인 상태

CLEANING:
- 직원 또는 사람이 식기 수거, 테이블 닦기, 쓰레기 정리 등을 하는 상태

POSSIBLY_EMPTY:
- 사람은 보이지 않지만 음식, 식기, 개인 물품 등이 남아 있어
  식사 종료인지 일시적 자리 비움인지 단일 이미지로 확정하기 어려운 상태

AFTER_MEAL_CANDIDATE:
- 사람이 없고 사용한 식기/잔반/쓰레기가 남아 있으며
  식사가 끝난 것으로 보이지만 시간 흐름 확인이 필요한 상태

UNCERTAIN:
- 흐림, 가림, 빈 테이블, 근거 부족 등으로 판단하기 어려운 상태

중요한 규칙:
- 사람이 없고 음식, 술잔, 개인 물품, 사용 중인 식기가 남아 있으면
  AFTER_MEAL_CANDIDATE보다 POSSIBLY_EMPTY를 우선 선택하라.
- 빈 테이블은 AFTER_MEAL로 판단하지 말고 UNCERTAIN으로 판단하라.
- 청소 행동이 명확하면 CLEANING을 최우선으로 판단하라.
- 단순히 사람이 테이블 근처를 지나가는 것만으로는 CLEANING이 아니다.

반드시 아래 JSON 형식으로만 답하라.

{
  "visible_people": true 또는 false,
  "active_dining": true 또는 false,
  "cleaning_action": true 또는 false,
  "used_dishes_or_leftovers": true 또는 false,
  "personal_items_or_food_remaining": true 또는 false,
  "frame_state": "DINING 또는 CLEANING 또는 POSSIBLY_EMPTY 또는 AFTER_MEAL_CANDIDATE 또는 UNCERTAIN",
  "confidence": 0.0에서 1.0 사이 숫자,
  "reason": "판단 근거를 한국어 한 문장으로 설명"
}
"""


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def classify_frame(image_path: str) -> Dict[str, Any]:
    base64_image = encode_image_to_base64(image_path)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": PROMPT
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]
    )

    result_text = response.output_text.strip()

    try:
        return json.loads(result_text)

    except json.JSONDecodeError:
        return {
            "visible_people": False,
            "active_dining": False,
            "cleaning_action": False,
            "used_dishes_or_leftovers": False,
            "personal_items_or_food_remaining": False,
            "frame_state": "UNCERTAIN",
            "confidence": 0.0,
            "reason": f"JSON 파싱 실패: {result_text}"
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python classify_frame.py table_crops\\T01_0000.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    result = classify_frame(image_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))