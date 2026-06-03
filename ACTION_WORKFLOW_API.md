# Action Workflow API

이 문서는 현재 구현된 Action 기반 청소 판단 API, Visual Metric 연동 방식, DB 저장 전략을 정리한다.

범위:

- `POST /api/action-cleanliness/workflow`
- Action workflow 실행 흐름
- Visual Metric 반영 방식
- `cleanliness_results` 저장 전략
- 예시 요청/응답

구현 코드는 이 문서에서 변경하지 않으며, 현재 동작 기준만 설명한다.

## 1. Endpoint

`POST /api/action-cleanliness/workflow`

역할:

1. 요청 JSON에서 `store_id`, `table_id`, `zone_id`, `frames`, `staff_zone_visits`를 받는다.
2. 각 frame의 raw YOLO/Vision payload를 기존 adapter로 정규화한다.
3. 기존 Action workflow를 실행한다.
4. 기존 Hybrid final score/cap rule을 적용한다.
5. 결과 JSON을 반환한다.
6. `save_result=true`일 때만 DB에 저장한다.

## 2. Request Schema

최소 요청 구조:

```json
{
  "store_id": "store_001",
  "table_id": "T06",
  "zone_id": "zone_B",
  "save_result": false,
  "frames": [
    {
      "captured_at": "2026-06-03T14:10:20",
      "person_present": true,
      "person_count": 2,
      "payload": {}
    },
    {
      "captured_at": "2026-06-03T14:42:35",
      "person_present": false,
      "person_count": 0,
      "payload": {}
    }
  ],
  "staff_zone_visits": []
}
```

필드 설명:

- `store_id`: 매장 식별자
- `table_id`: 테이블 식별자
- `zone_id`: 직원 체류 판단에 사용할 zone 식별자
- `save_result`: `true`면 DB 저장, 없거나 `false`면 계산만 수행
- `frames`: frame 목록
- `staff_zone_visits`: 직원 zone 체류 정보

`frames[].payload`에는 기존 adapter가 지원하는 raw 입력을 넣을 수 있다.

지원 예:

- `objects`
- `detected_objects`
- `yolo_objects`
- `flags`
- `scene_flags`
- `vision_confidence`
- `contamination_area_ratio`
- `person_exposed`

### 2.1 `captured_at` Sorting and Validation

`frames[].captured_at`는 ISO datetime 문자열이어야 한다.

현재 문서 기준 규칙:

- frame는 입력 순서와 무관하게 내부에서 `captured_at` 기준 오름차순 정렬된다.
- `frames`는 최소 2개 이상 필요하다.
- `captured_at`이 파싱 불가능하면 validation error가 발생한다.
- 정렬 후에도 식사 흐름을 만들 수 없으면 workflow는 `400`으로 거절될 수 있다.
- 정렬 후에도 `MEAL_ENDED`까지 도달하지 못하면 workflow는 `400`으로 거절될 수 있다.

권장:

- 가능하면 호출 측에서 이미 시간순으로 보내는 것이 가장 안전하다.
- 동일 시각 frame를 여러 개 보내는 것은 피하는 것이 좋다.

## 3. Frame Payload Normalization

frame payload는 기존 Visual Metric adapter를 통해 `YoloTableState` 입력으로 정규화된다.

지원 형태 예:

```json
{
  "objects": [
    { "class": "cup", "confidence": 0.91 },
    { "class": "trash", "confidence": 0.71 }
  ],
  "scene_flags": {
    "messy_arrangement": false
  },
  "vision_confidence": 0.84
}
```

alias 처리 예:

- `trash`, `litter` -> `trash`
- `paper`, `wrapper` -> `wrapper`
- `tissue`, `napkin` -> `napkin`
- `plastic_cup`, `cup` -> `cup`
- `plate`, `dish` -> `dish`
- `tray` -> `tray`
- `leftover`, `food_residue` -> `food_waste`

보조 규칙:

- `person`은 `person_exposed=true`로 반영되며 점수에 직접 감점되지 않는다.
- unknown class는 감점 대상 객체 수에 포함되지 않는다.
- scene-level `vision_confidence`가 없으면 missing 상태로 취급되고 관련 reason code가 추가될 수 있다.

## 4. Action Workflow Execution

이 API는 기존 Action workflow를 그대로 호출한다.

핵심 흐름:

1. `frames`로부터 `TableOccupancySample` 생성
2. `MealSession` 판단
3. `CleaningTask` 생성
4. `StaffZoneVisit` 후보 판정
5. 전후 frame의 `YoloTableState` 비교
6. Action 결과 계산
7. Visual Metric 계산
8. Hybrid final score/cap rule 적용

이 문서의 범위에서는 내부 상태 머신 규칙을 재정의하지 않는다.

### 4.1 Minimum Status Criteria

아래는 구현 이해를 위한 최소 기준 요약이다.

`CUSTOMER_IN_USE`

- 테이블이 아직 손님 사용 중으로 판단된 상태다.
- 일반적으로 고객 존재가 충분한 시간 동안 이어질 때 사용 중 세션으로 간주된다.
- 현재 구현 테스트 기준에서는 `CUSTOMER_IN_USE < 60초`인 경우 meal session이 생성되지 않는다.

`MEAL_ENDED`

- 고객 부재가 일정 시간 이상 이어져 식사가 종료된 것으로 판단된 상태다.
- 현재 구현 테스트 기준에서는 고객 부재가 `90초` 이상 이어지면 `MEAL_ENDED`가 가능하다.

`CLEANING_CANDIDATE`

- 식사 종료 후 같은 zone에서 직원 체류가 청소 후보로 볼 만큼 확인된 상태다.
- 현재 구현 테스트 기준에서는 동일 zone 체류가 `10초 이상`이면 cleaning candidate가 될 수 있다.

`CLEANED_LIKELY`

- 직원 체류만으로는 부족하고, 식사 종료 이후 전후 frame에서 상태 개선 증거가 함께 있어야 한다.
- 즉, `StaffZoneVisit`만으로는 안 되고 YOLO/Visual 변화 조건을 함께 만족해야 한다.
- 또한 `CLEANED_LIKELY`는 `NEED_CLEANING` 이후에만 가능하다.

`MISSED_SUSPECTED`

- 식사 종료 후 SLA 안에 청소 체류 또는 충분한 청소 증거가 확인되지 않아 청소 누락 의심으로 판단된 상태다.
- 현재 구현 기준 SLA는 `300초`다.

## 5. Response Schema

응답 필드 예:

```json
{
  "store_id": "store_001",
  "table_id": "T06",
  "zone_id": "zone_B",
  "saved": false,
  "result_id": null,
  "meal_status": "MEAL_ENDED",
  "cleaning_status": "CLEANED_LIKELY",
  "action_score": 0.85,
  "visual_score": 100,
  "visual_clean_score": 1.0,
  "visual_mess_score": 0.0,
  "final_cleanliness_score": 92,
  "final_grade": "정상",
  "decision": "CLEAN_OK",
  "applied_caps": [],
  "reason_codes": [
    "CUSTOMER_LEFT_CONFIRMED",
    "STAFF_ZONE_DWELL_CONFIRMED",
    "TABLE_STATE_CHANGED",
    "OBJECT_COUNT_REDUCED",
    "CLUTTER_OBJECTS_REDUCED"
  ],
  "explanation": "식사 종료 이후 해당 구역에 직원 체류가 확인되었고, 전후 프레임에서 테이블 상태 개선이 관찰되어 청소 완료 추정으로 판단했습니다."
}
```

필드 설명:

- `saved`: DB 저장 여부
- `result_id`: 저장된 경우 `cleanliness_results.id`
- `meal_status`: 식사 상태
- `cleaning_status`: Action workflow 상태
- `action_score`: Action 기반 점수
- `visual_score`: Visual Metric 0~100 점수
- `visual_clean_score`: `visual_score / 100`
- `visual_mess_score`: `1 - visual_clean_score`
- `final_cleanliness_score`: 최종 0~100 점수
- `final_grade`: 최종 등급 문자열
- `decision`: 최종 의사결정
- `applied_caps`: cap rule 적용 내역
- `reason_codes`: Action + Visual reason code 병합 결과
- `explanation`: 최종 설명

### 5.1 `visual_metric_grade`

`visual_metric_grade`는 Visual Metric 내부 진단 필드다.

등급:

- `A`: `90~100`
- `B`: `70~89`
- `C`: `50~69`
- `D`: `30~49`
- `E`: `0~29`

주의:

- 현재 top-level API 예시 응답에는 이 필드를 직접 노출하지 않아도 된다.
- 다만 내부 Visual Metric 결과나 확장 응답에서 사용할 수 있는 진단 값으로 이해하면 된다.

## 6. Visual Metric

현재 final score에 들어가는 visual component는 기존 Visual Metric 결과를 사용한다.

공식:

```text
visual_score = 100 - penalties + bonuses
visual_score = clamp(0, 100)
visual_clean_score = visual_score / 100
visual_mess_score = 1 - visual_clean_score
```

대표 출력:

- `visual_score`
- `visual_clean_score`
- `visual_mess_score`
- penalty breakdown
- bonus breakdown
- `reason_codes`
- `visual_metric_grade`

최종 점수 계산에서는 기존 Hybrid rule을 유지한다.

## 7. Final Score and Cap Rules

최종 점수는 기존 Hybrid final score 로직을 그대로 사용한다.

특징:

- Action score와 Visual clean score를 함께 사용
- high mess cap 유지
- missed suspected cap 유지
- need cleaning cap 유지
- cleaning unverified cap 유지
- low visual confidence cap 유지

이 API는 cap rule을 재정의하지 않고 기존 구현 결과를 그대로 반환한다.

## 8. DB Save Strategy

`save_result=true`일 때만 기존 `cleanliness_results` 테이블에 저장한다.

새 테이블은 사용하지 않는다.

저장 매핑:

- `store_name` <- `store_id`
- `cctv_id` <- `{store_id}::{zone_id}`
- `cctv_nickname` <- `zone_id`
- `roi_name` <- `table_id`
- `mode` <- `action_workflow`
- `decision` <- 리포트 호환용 decision
- `score` <- 리포트 호환용 1~5 점수
- `confidence` <- `action_confidence`
- `final_stage` <- `workflow_api`
- `summary` <- 최종 explanation
- `findings` <- `reason_codes`
- `action_features` <- 요청/응답/세부 breakdown JSON

중요:

- `store_name <- store_id` 매핑은 기존 `cleanliness_results` 테이블과 reports 화면 호환을 위한 저장 전략이다.
- 즉, 여기서의 `store_name`은 별도 정규화된 매장 마스터 이름이 아니라 API 요청의 `store_id`를 그대로 담는 호환용 값이다.

### 8.1 Report-Compatible Decision Mapping

기존 reports 집계는 `decision` 값을 기준으로 요약한다.

저장 시 매핑:

- `CLEANED_LIKELY` -> `cleaned_likely`
- `CLEANING_UNVERIFIED` -> `needs_check`
- `MISSED_SUSPECTED` -> `needs_check`
- `NEED_CLEANING` -> `needs_check`
- 그 외 -> `unknown`

### 8.2 Report-Compatible Score Mapping

기존 reports 화면과 `fetch_cleanliness_store_summary()`는 `score`를 `1~5` 척도로 사용한다.

그래서 action workflow 저장 시:

- `action_features.final_cleanliness_score`에는 실제 `0~100` 점수를 그대로 저장
- `cleanliness_results.score`에는 아래 규칙으로 변환한 `1~5` 점수를 저장

변환 규칙:

```text
90~100 -> 5
70~89  -> 4
50~69  -> 3
30~49  -> 2
0~29   -> 1
```

이 전략으로 reports의 평균 점수와 개별 점수 표시가 기존 방식과 호환된다.

## 9. Example Request

```json
{
  "store_id": "store_001",
  "table_id": "T06",
  "zone_id": "zone_B",
  "save_result": true,
  "frames": [
    {
      "captured_at": "2026-06-03T14:10:20",
      "person_present": true,
      "person_count": 2,
      "payload": {
        "objects": [
          { "class": "cup", "confidence": 0.91 },
          { "class": "tray", "confidence": 0.84 }
        ]
      }
    },
    {
      "captured_at": "2026-06-03T14:42:35",
      "person_present": false,
      "person_count": 0,
      "payload": {
        "objects": [
          { "class": "cup", "confidence": 0.91 },
          { "class": "tray", "confidence": 0.84 }
        ]
      }
    },
    {
      "captured_at": "2026-06-03T14:44:40",
      "person_present": false,
      "person_count": 0,
      "payload": {
        "objects": [],
        "scene_flags": {
          "messy_arrangement": false
        },
        "vision_confidence": 0.84
      }
    }
  ],
  "staff_zone_visits": [
    {
      "visit_id": "visit_000341",
      "staff_id": "staff_03",
      "zone_id": "zone_B",
      "entered_at": "2026-06-03T14:44:10",
      "left_at": "2026-06-03T14:44:36",
      "dwell_seconds": 26,
      "mean_confidence": 0.76,
      "sample_count": 8
    }
  ]
}
```

설명:

- 이 예시는 청소 전 frame에 컵/트레이가 있고 청소 후 frame에서 clutter가 줄어든 상황을 단순화해서 보여준다.
- Example Response의 `visual_score=100`과 모순되지 않도록 마지막 frame에서는 trash-like object를 넣지 않았다.

## 10. Example Response

```json
{
  "store_id": "store_001",
  "table_id": "T06",
  "zone_id": "zone_B",
  "saved": true,
  "result_id": 123,
  "meal_status": "MEAL_ENDED",
  "cleaning_status": "CLEANED_LIKELY",
  "action_score": 0.85,
  "visual_score": 100,
  "visual_clean_score": 1.0,
  "visual_mess_score": 0.0,
  "final_cleanliness_score": 92,
  "final_grade": "정상",
  "decision": "CLEAN_OK",
  "applied_caps": [],
  "reason_codes": [
    "CUSTOMER_LEFT_CONFIRMED",
    "STAFF_ZONE_DWELL_CONFIRMED",
    "TABLE_STATE_CHANGED",
    "OBJECT_COUNT_REDUCED",
    "CLUTTER_OBJECTS_REDUCED"
  ],
  "explanation": "식사 종료 이후 해당 구역에 직원 체류가 확인되었고, 전후 프레임에서 테이블 상태 개선이 관찰되어 청소 완료 추정으로 판단했습니다."
}
```

주의:

- 위 예시는 설명용 예시다.
- 실제 응답에서는 Visual Metric 결과에 따라 `reason_codes`에 object 관련 code가 더 들어갈 수 있다.
- `TRASH_LIKE_OBJECTS_DETECTED` 같은 code가 포함될 수 있는 경우에는 `visual_score`가 더 낮거나, 다른 보너스/감점 조합이 함께 반영될 수 있다.

## 11. Notes

- `save_result=false`면 계산만 하고 DB에는 저장하지 않는다.
- `save_result=true`면 결과를 `cleanliness_results`에 저장하고 `result_id`를 반환한다.
- 기존 Action workflow, Visual Metric, Hybrid cap rule은 이 API에서 재작성하지 않고 그대로 재사용한다.
