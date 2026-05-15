# 스마트폰 중독 위험도 예측 및 AI 관리 시스템

스마트폰 사용 패턴 데이터를 기반으로 사용자의 스마트폰 중독 위험도를 예측하고, Gemini LLM을 활용하여 사용자 맞춤형 분석과 관리 리포트를 제공하는 딥러닝 + 생성형 AI 융합 프로젝트입니다.

기존 스마트폰 중독 위험도 예측 딥러닝 프로젝트를 확장하여, 단순 예측 기능뿐만 아니라 LLM 기반 분석, 상담 기록 저장, 변화 추이 시각화, AI 관리 리포트 생성, PDF 다운로드 기능까지 구현했습니다.

---

## 프로젝트 소개

본 프로젝트는 스마트폰 사용 패턴 데이터를 기반으로 사용자의 스마트폰 중독 위험도를 예측하는 딥러닝 프로젝트를 고도화한 확장 프로젝트입니다.

기존 프로젝트에서는 일일 스마트폰 사용시간, SNS 사용시간, 게임 사용시간, 수면 시간, 알림 수, 앱 실행 횟수 등의 데이터를 활용하여 스마트폰 중독 여부를 예측하는 딥러닝 이진 분류 모델을 구현했습니다.

이번 고도화 프로젝트에서는 예측 결과를 단순히 확률과 위험도로 보여주는 것에서 나아가, Gemini LLM을 활용하여 사용자의 행동 패턴을 자연어로 분석하고 개선 방향을 제안하는 AI 관리 기능을 추가했습니다.

또한 사용자의 예측 결과를 기록으로 저장하고, 저장된 기록을 기반으로 변화 추이를 시각화하며, 7일, 14일, 30일 단위의 AI 관리 리포트를 생성할 수 있도록 구현했습니다.

이를 통해 딥러닝 예측 모델과 생성형 AI를 결합한 사용자 맞춤형 스마트폰 사용 습관 관리 시스템을 구축했습니다.

---

## 주요 기능

- 스마트폰 사용 패턴 기반 중독 위험도 예측
- 딥러닝 기반 이진 분류 모델 구현
- 예측 확률 기반 위험도 4단계 분류
  - 정상
  - 관심군
  - 주의군
  - 고위험군
- Gemini LLM 기반 사용자 맞춤형 분석 제공
- 사용자의 스마트폰 사용 습관에 대한 문제점 및 개선 방법 제안
- 목표 사용시간, 목표 수면시간, 목표 알림 수 설정
- 현재 사용 패턴과 목표값 비교 기능
- 예측 결과 및 LLM 분석 결과 저장
- 저장된 기록 조회 기능
- 중독 확률, 일일 사용시간, 수면 시간 변화 추이 시각화
- 7일, 14일, 30일 단위 AI 관리 리포트 생성
- 예측 결과 PDF 저장 기능
- AI 관리 리포트 PDF 저장 기능
- 실제값과 예측값 비교 기능
- 정확 / 오분류 결과 확인 기능

---

## 사용 기술

### Language

- Python

### Data Processing / Machine Learning

- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Joblib

### Deep Learning

- Dense Layer
- Dropout
- Sigmoid
- Binary Classification

### LLM / Generative AI

- Google Generative AI
- Gemini
- Prompt Engineering

### Web Application

- Streamlit

### Report / Storage

- JSON
- ReportLab
- PDF 생성

---

## 데이터셋

본 프로젝트에서는 스마트폰 사용 및 중독 관련 데이터를 사용했습니다.

사용한 데이터 파일은 다음과 같습니다.

```text
Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
```

데이터는 총 7,500개의 행으로 구성되어 있으며, 스마트폰 사용 습관과 관련된 다양한 컬럼을 포함합니다.

주요 컬럼은 다음과 같습니다.

| 컬럼명 | 설명 |
|---|---|
| daily_screen_time_hours | 일일 스마트폰 사용시간 |
| social_media_hours | SNS 사용시간 |
| gaming_hours | 게임 사용시간 |
| sleep_hours | 수면 시간 |
| notifications_per_day | 하루 알림 수 |
| app_opens_per_day | 하루 앱 실행 횟수 |
| addicted_label | 스마트폰 중독 여부 |

---

## 데이터 전처리

모델 학습에 사용한 입력 변수는 다음과 같습니다.

```python
FEATURES = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
]
```

타깃 변수는 다음과 같습니다.

```python
TARGET = "addicted_label"
```

입력 데이터는 학습 데이터와 테스트 데이터로 분리했습니다.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

딥러닝 모델의 안정적인 학습을 위해 `StandardScaler`를 사용하여 입력 데이터를 표준화했습니다.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

학습된 scaler는 웹 애플리케이션에서도 동일한 전처리를 적용할 수 있도록 저장하여 재사용했습니다.

```python
joblib.dump(scaler, "model/scaler.pkl")
```

---

## 모델 구조

본 프로젝트에서는 Keras 기반 딥러닝 이진 분류 모델을 사용했습니다.

모델 구조는 다음과 같습니다.

```python
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])
```

은닉층에는 `ReLU` 활성화 함수를 사용했고, 과적합을 완화하기 위해 `Dropout` 계층을 추가했습니다.

출력층에서는 `sigmoid` 활성화 함수를 사용하여 0과 1 사이의 확률값을 출력하도록 구성했습니다.

---

## 모델 학습

모델은 이진 분류 문제에 적합한 `binary_crossentropy` 손실 함수를 사용하여 학습했습니다.

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

학습 설정은 다음과 같습니다.

| 항목 | 값 |
|---|---:|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 50 |
| Batch Size | 16 |
| Validation Split | 0.2 |

학습 완료 후 모델은 Streamlit 웹 애플리케이션에서 재사용할 수 있도록 저장했습니다.

```python
model.save("model/addiction_model.h5")
```

---

## 모델 성능

학습 로그 기준 주요 성능은 다음과 같습니다.

| 지표 | 값 |
|---|---:|
| Train Accuracy | 0.9404 |
| Train Loss | 0.0999 |
| Validation Accuracy | 0.9317 |
| Validation Loss | 0.1169 |

모델은 테스트 데이터에 대해 실제값과 예측값을 비교할 수 있도록 구성했으며, 예측 결과를 기준으로 정확 / 오분류 여부를 확인할 수 있습니다.

---

## 위험도 분류 기준

모델은 0과 1 사이의 확률값을 출력합니다.

해당 확률값을 기준으로 스마트폰 중독 위험도를 4단계로 분류했습니다.

| 확률 구간 | 위험도 |
|---|---|
| 0.00 ~ 0.24 | 정상 |
| 0.25 ~ 0.49 | 관심군 |
| 0.50 ~ 0.74 | 주의군 |
| 0.75 ~ 1.00 | 고위험군 |

위험도 분류 함수는 다음과 같이 구성했습니다.

```python
def classify_risk(prob: float) -> str:
    if prob < 0.25:
        return "정상"
    if prob < 0.50:
        return "관심군"
    if prob < 0.75:
        return "주의군"
    return "고위험군"
```

---

## LLM 기반 AI 분석 기능

본 프로젝트에서는 Gemini LLM을 활용하여 사용자의 스마트폰 사용 패턴을 자연어로 분석하는 기능을 구현했습니다.

딥러닝 모델이 중독 확률과 위험도를 예측하면, 해당 결과와 사용자의 입력 데이터를 LLM 프롬프트에 포함하여 맞춤형 분석 결과를 생성합니다.

LLM 분석에는 다음 정보가 사용됩니다.

- 일일 스마트폰 사용시간
- SNS 사용시간
- 게임 사용시간
- 수면 시간
- 알림 수
- 앱 실행 횟수
- 딥러닝 모델의 중독 확률
- 위험도 단계

LLM은 사용자의 행동 패턴을 바탕으로 다음 내용을 제공합니다.

- 현재 사용 습관 분석
- 위험 요인 설명
- 문제 원인 제시
- 실천 가능한 개선 방법 제안
- 스마트폰 사용 습관 관리 방향 제안

LLM 호출에는 `google-generativeai` 라이브러리를 사용했습니다.

```python
import google.generativeai as genai

model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content(prompt)
```

---

## 웹 애플리케이션 구성

Streamlit을 사용하여 딥러닝 모델과 LLM 분석 기능을 웹 애플리케이션 형태로 구현했습니다.

웹 애플리케이션은 총 5개의 탭으로 구성되어 있습니다.

```text
예측하기
기록 조회
변화 추이
AI 관리 리포트
실제값 vs 예측값
```

---

### 예측하기 탭

사용자가 직접 스마트폰 사용 패턴을 입력하면 딥러닝 모델이 중독 확률을 예측합니다.

입력 항목은 다음과 같습니다.

- 일일 사용시간
- SNS 사용시간
- 게임 사용시간
- 수면 시간
- 알림 수
- 앱 실행 횟수

또한 사용자는 관리 목표를 직접 설정할 수 있습니다.

- 목표 일일 사용시간
- 목표 수면 시간
- 목표 알림 수

예측 결과로는 다음 정보를 제공합니다.

- 입력값 요약
- 중독 확률
- 위험도 단계
- 목표 비교 결과
- Gemini LLM 분석 결과
- 프롬프트 확인 기능
- 예측 결과 PDF 저장 기능

---

### 기록 조회 탭

예측하기 탭에서 저장된 결과를 조회할 수 있는 기능입니다.

저장되는 주요 정보는 다음과 같습니다.

- 생성 시각
- 입력값
- 중독 확률
- 위험도
- LLM 분석 결과

기록은 JSON 파일 형태로 저장됩니다.

```text
storage/history.json
```

---

### 변화 추이 탭

저장된 기록을 기반으로 사용자의 스마트폰 사용 습관 변화를 시각화합니다.

제공하는 변화 추이는 다음과 같습니다.

- 중독 확률 추이
- 일일 사용시간 추이
- 수면 시간 추이

Streamlit의 `line_chart`를 활용하여 시간 흐름에 따른 변화를 확인할 수 있도록 구성했습니다.

---

### AI 관리 리포트 탭

저장된 기록을 기반으로 Gemini LLM이 주기적인 관리 리포트를 생성합니다.

분석 기간은 다음 중 선택할 수 있습니다.

- 7일
- 14일
- 30일

리포트 생성 시 사용되는 정보는 다음과 같습니다.

- 평균 일일 사용시간
- 평균 수면 시간
- 평균 중독 확률
- 위험도 분포

AI 관리 리포트는 다음 내용을 포함합니다.

- 최근 사용 패턴 요약
- 위험 요인 설명
- 다음 기간의 실천 목표 제안
- 관리 방향 제시

생성된 리포트는 PDF 파일로 저장할 수 있습니다.

---

### 실제값 vs 예측값 탭

테스트 데이터에 대한 실제값과 모델 예측값을 비교할 수 있는 기능입니다.

제공하는 정보는 다음과 같습니다.

- 전체 데이터 수
- 정확히 예측한 개수
- 오분류 개수
- 실제값
- 예측값
- 예측 확률
- 판정 결과

또한 전체 / 정확 / 오분류 필터를 제공하여 모델이 어떤 데이터를 잘 예측했고, 어떤 데이터를 오분류했는지 확인할 수 있도록 구성했습니다.

---

## PDF 저장 기능

본 프로젝트에서는 `ReportLab`을 활용하여 PDF 리포트 저장 기능을 구현했습니다.

PDF로 저장할 수 있는 항목은 다음과 같습니다.

- 단일 예측 결과 리포트
- AI 관리 리포트

단일 예측 결과 리포트에는 다음 정보가 포함됩니다.

- 생성 시각
- 중독 확률
- 위험도
- 사용자 입력값
- 목표 비교 결과
- LLM 분석 결과
- 사용된 프롬프트

AI 관리 리포트에는 다음 정보가 포함됩니다.

- 분석 기간
- 기록 수
- 평균 일일 사용시간
- 평균 수면 시간
- 평균 중독 확률
- LLM 관리 리포트

---

## 프로젝트 구조

```text
JMJ_DL_LLM_proj/
├── app.py
├── dataset/
│   └── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
├── llm/
│   └── llm_service.py
├── model/
│   ├── addiction_model.h5
│   └── scaler.pkl
├── storage/
│   └── history.json
├── utils/
│   └── pdf_service.py
├── requirements.txt
└── README.md
```

---

## 실행 방법

### 1. 프로젝트 클론

```bash
git clone https://github.com/David35798/DL_LLM_proj_01.git
cd DL_LLM_proj_01
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. Gemini API Key 설정

Gemini API를 사용하기 위해 환경변수에 API Key를 설정합니다.

Windows PowerShell 기준:

```powershell
$env:GOOGLE_API_KEY="your_api_key"
```

또는 `.env` 파일을 사용하는 경우 다음과 같이 설정할 수 있습니다.

```env
GOOGLE_API_KEY=your_api_key
GOOGLE_MODEL=gemini-2.5-flash
```

보안을 위해 API Key는 코드에 직접 작성하지 않고 환경변수로 관리하는 것을 권장합니다.

### 4. Streamlit 실행

```bash
streamlit run app.py
```

---

## 본 프로젝트를 통해 배운 점

- 기존 딥러닝 분류 프로젝트를 LLM 기반 AI 서비스로 확장하는 과정을 경험했습니다.
- 스마트폰 사용 패턴 데이터를 기반으로 중독 위험도를 예측하는 딥러닝 모델을 구현했습니다.
- 딥러닝 모델의 예측 결과를 LLM 프롬프트에 반영하여 사용자 맞춤형 분석 결과를 생성했습니다.
- Gemini API를 활용하여 자연어 기반 AI 분석 기능을 구현했습니다.
- Prompt Engineering을 통해 LLM이 사용자의 행동 패턴, 위험 요인, 개선 방법을 구체적으로 제안하도록 구성했습니다.
- 예측 결과를 JSON 파일로 저장하고, 저장된 기록을 기반으로 변화 추이를 시각화했습니다.
- 누적 기록을 바탕으로 7일, 14일, 30일 단위의 AI 관리 리포트를 생성하는 기능을 구현했습니다.
- ReportLab을 활용하여 예측 결과와 관리 리포트를 PDF로 저장하는 기능을 구현했습니다.
- Streamlit을 활용하여 딥러닝 모델, LLM, 기록 저장, 시각화, PDF 생성 기능을 하나의 웹 서비스로 통합했습니다.

---

## 향후 개선 방향

- API Key 보안 관리 강화
  - 환경변수 사용
  - `.env` 파일 관리
  - `.gitignore`에 민감 정보 제외
- 사용자별 계정 기능 추가
- 상담 기록을 JSON 파일이 아닌 DB에 저장
  - SQLite
  - MySQL
  - PostgreSQL
- LangChain 기반 LLM 호출 구조 고도화
- RAG 기반 스마트폰 사용 습관 개선 가이드 검색 기능 추가
- 사용자 기록 기반 맞춤형 프롬프트 고도화
- 더 다양한 시각화 기능 추가
  - 위험도 분포 그래프
  - 주간 평균 사용시간 그래프
  - 수면 시간과 중독 확률 상관 분석
- 모델 성능 평가 지표 추가
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- 다양한 모델과 성능 비교
  - Logistic Regression
  - Random Forest
  - XGBoost
  - DNN 구조 변경
- Docker 기반 배포
- Streamlit Cloud 또는 클라우드 서버 배포

---

## 프로젝트 한 줄 소개

스마트폰 사용 패턴 데이터를 기반으로 딥러닝 모델이 중독 위험도를 예측하고, Gemini LLM이 사용자 맞춤형 분석과 관리 리포트를 제공하는 AI 기반 스마트폰 사용 습관 관리 시스템입니다.