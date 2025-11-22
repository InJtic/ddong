# Project DDong

제작중입니다.

테스트 실행 코드:

- 데이터 생성 예시:
    `PYTHONPATH=. python3 scripts/data.py --speed 1 --direction DOWN`
    a-zA-Z0-9의 모든 데이터를 만듭니다.

    그 외 자동 설정:
      - fps = 30
      - 영상 길이 = 2s
      - 영상 크기 = 224×224
      - 글자 폰트 = 맑은 고딕(`resources/malgun.ttf`) 100pt
      - 글자 위치 = 잘리지 않는 영역에서 랜덤
      - 글자 변환법 = 선형 이동(방향: `direction`)
      - 배경 변환법 = 고정

- 영상화 예시:
    `PYTHONPATH=. python3 scripts/combine.py --speed 1 --direction DOWN --label c --sample_id 25`
    특정 샘플을 영상화합니다.

- 답변 추출 예시:
    `PYTHONPATH=. python3 src/execute.py --speed 1 --direction DOWN --label c --sample_id 25`
    특정 샘플을 Qwen-VL에게 넘겨 읽도록 명령합니다. 프롬프트는 다음을 사용했습니다:

    ```python
    PROMPT = (
      "Step 1: Describe the shape and movement of the object hiding in the noise. "
      "(e.g. Does it have curves? Straight lines? Is it closed or open?) "
      "\n"
      "Step 2: Based on the description, identify the specific English alphabet. "
      "\n"
      "Final Answer Format: The letter is [X]."
    )
    ```

  - 사용된 모델 = `Qwen/Qwen2-VL-7B-Instruct`
