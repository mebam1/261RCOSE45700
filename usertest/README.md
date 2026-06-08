# Table Cleanliness User Test

웹에서 테이블 청결도 유저 테스트를 수행하는 로컬 앱입니다.

## 실행

```powershell
uvicorn usertest.server:app --reload --host 127.0.0.1 --port 8010
```

브라우저에서 접속:

```text
http://127.0.0.1:8010
```

## 테스트 방식

- 왼쪽: 청결한 기준 사진
- 오른쪽: 평가 대상 generated dirty 사진
- 참가자는 오른쪽 사진에 대해 청결도 `1~5점`과 `청소 필요/불필요`를 선택합니다.

`data/dataset`의 파일명 규칙을 사용합니다.

```text
{번호}_clean_01.jpg
{번호}_dirty_02.jpg
{번호}_dirty_03.jpg
{번호}_dirty_04.jpg
```

## 저장 위치

응답은 JSONL로 누적 저장됩니다.

```text
usertest/responses/responses.jsonl
```

## 분석 API

이미지별 median, stddev, 청소 필요 비율과 threshold 후보를 계산합니다.

```text
http://127.0.0.1:8010/api/summary
```

CSV 다운로드:

```text
http://127.0.0.1:8010/api/export.csv
```

threshold는 `청결도 점수 <= threshold`이면 청소 필요로 예측하는 규칙 중,
현재 응답에 대해 accuracy가 가장 높은 값을 선택합니다.
