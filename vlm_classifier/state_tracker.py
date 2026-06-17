from typing import Dict, Any


class RestaurantStateTracker:
    """
    여러 CCTV 프레임을 순서대로 받아 시간 흐름 기반 최종 상태를 판단한다.
    """

    def __init__(self, empty_threshold_seconds: int = 600):
        """
        empty_threshold_seconds:
        사람이 없는 상태가 몇 초 이상 지속되면 식사 후로 볼 것인지에 대한 기준.
        기본값은 600초, 즉 10분.
        """
        self.empty_threshold_seconds = empty_threshold_seconds

        # 현재 최종 상태
        self.current_state = "UNCERTAIN"

        # 사람이 없는 상태가 시작된 시각
        self.empty_start_time = None

        # 마지막으로 식사 중 상태가 확인된 시각
        self.last_seen_dining_time = None

    def update(self, timestamp_seconds: int, frame_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        한 프레임의 Gemini 분석 결과를 받아 시간 흐름까지 고려한 최종 상태를 반환한다.

        timestamp_seconds:
        현재 프레임의 영상 내 시각. 예: 0초, 10초, 20초 ...

        frame_result:
        classify_frame.py에서 반환한 한 장 이미지 분석 결과.
        """

        frame_state = frame_result.get("frame_state", "UNCERTAIN")

        visible_people = frame_result.get("visible_people", False)
        active_dining = frame_result.get("active_dining", False)
        cleaning_action = frame_result.get("cleaning_action", False)
        used_dishes = frame_result.get("used_dishes_or_leftovers", False)
        personal_items = frame_result.get("personal_items_or_food_remaining", False)

        reason = frame_result.get("reason", "")

        # 1순위: 청소 행동이 명확하면 CLEANING
        if cleaning_action or frame_state == "CLEANING":
            self.current_state = "CLEANING"
            self.empty_start_time = None

            return {
                "final_state": "CLEANING",
                "temporal_reason": "식기 수거, 테이블 닦기, 정리 행동이 감지되어 청소 중으로 판단함.",
                "frame_reason": reason
            }

        # 2순위: 사람이 보이고 식사 행동이 명확하면 DINING
        if visible_people and active_dining:
            self.current_state = "DINING"
            self.empty_start_time = None
            self.last_seen_dining_time = timestamp_seconds

            return {
                "final_state": "DINING",
                "temporal_reason": "사람이 보이고 식사 행동이 감지되어 식사 중으로 판단함.",
                "frame_reason": reason
            }

        # 3순위: 사람이 없고 음식/식기/개인 물품이 남아 있는 경우
        # 담배, 화장실, 전화 등으로 잠시 자리를 비운 상황일 수 있으므로
        # 바로 AFTER_MEAL로 확정하지 않는다.
        if not visible_people and (used_dishes or personal_items):
            if self.empty_start_time is None:
                self.empty_start_time = timestamp_seconds

            empty_duration = timestamp_seconds - self.empty_start_time

            # 사람이 없는 상태가 10분 이상 지속되면 식사 후로 판단
            if empty_duration >= self.empty_threshold_seconds:
                self.current_state = "AFTER_MEAL"

                return {
                    "final_state": "AFTER_MEAL",
                    "temporal_reason": f"사람이 없는 상태가 {empty_duration}초 이상 지속되어 식사 후로 판단함.",
                    "frame_reason": reason
                }

            # 10분이 지나기 전까지는 식사 후로 확정하지 않고 보류
            else:
                self.current_state = "UNCERTAIN"

                return {
                    "final_state": "UNCERTAIN",
                    "temporal_reason": f"사람은 없지만 음식/식기/개인 물품이 남아 있어 일시적 자리 비움 가능성이 있음. 현재 공백 지속 시간: {empty_duration}초.",
                    "frame_reason": reason
                }

        # 4순위: 사람이 보이지만 식사 행동이 명확하지 않은 경우
        if visible_people and not active_dining:
            self.current_state = "UNCERTAIN"
            self.empty_start_time = None

            return {
                "final_state": "UNCERTAIN",
                "temporal_reason": "사람은 보이지만 식사 중인지 청소 중인지 명확하지 않아 판정 보류.",
                "frame_reason": reason
            }

        # 5순위: 빈 테이블 또는 근거 부족
        self.current_state = "UNCERTAIN"

        return {
            "final_state": "UNCERTAIN",
            "temporal_reason": "식사 중, 식사 후, 청소 중으로 판단할 충분한 근거가 없음.",
            "frame_reason": reason
        }