import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

from llm.llm_service import generate_single_analysis, generate_weekly_report
from utils.pdf_service import build_single_result_pdf, build_weekly_report_pdf

st.set_page_config(page_title="스마트폰 중독 관리 시스템", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_FILE = BASE_DIR / "dataset" / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
STORAGE_DIR = BASE_DIR / "storage"
HISTORY_FILE = STORAGE_DIR / "history.json"

STORAGE_DIR.mkdir(exist_ok=True)

FEATURES = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
]
TARGET = "addicted_label"


@st.cache_resource
def load_assets():
    model = load_model(MODEL_DIR / "addiction_model.h5")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    return model, scaler


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


def classify_risk(prob: float) -> str:
    if prob < 0.25:
        return "정상"
    if prob < 0.50:
        return "관심군"
    if prob < 0.75:
        return "주의군"
    return "고위험군"


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def append_history(record: dict) -> None:
    history = load_history()
    history.append(record)
    save_history(history)


def history_to_df() -> pd.DataFrame:
    history = load_history()
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df.sort_values("created_at", ascending=False).reset_index(drop=True)


def build_record(
    daily: float,
    sns: float,
    game: float,
    sleep: float,
    notifications: int,
    app_opens: int,
    prob: float,
    risk: str,
    llm_text: str,
) -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "daily_screen_time_hours": daily,
        "social_media_hours": sns,
        "gaming_hours": game,
        "sleep_hours": sleep,
        "notifications_per_day": notifications,
        "app_opens_per_day": app_opens,
        "probability": round(prob, 4),
        "risk_level": risk,
        "llm_analysis": llm_text,
    }


model, scaler = load_assets()
df = load_data()

st.title("스마트폰 중독 관리 시스템")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["예측하기", "기록 조회", "변화 추이", "AI 관리 리포트", "실제값 vs 예측값"]
)

with tab1:
    left, right = st.columns(2)

    with left:
        daily = st.slider("일일 사용시간", 0.0, 15.0, 5.0, 0.1)
        sns_time = st.slider("SNS 사용시간", 0.0, 10.0, 2.0, 0.1)
        game = st.slider("게임 사용시간", 0.0, 10.0, 1.0, 0.1)

    with right:
        sleep = st.slider("수면 시간", 0.0, 12.0, 6.0, 0.1)
        notifications = st.slider("알림 수", 0, 300, 80, 1)
        app_opens = st.slider("앱 실행 횟수", 0, 200, 40, 1)

    st.subheader("입력값 요약")
    summary_df = pd.DataFrame(
        {
            "항목": [
                "일일 사용시간",
                "SNS 사용시간",
                "게임 사용시간",
                "수면 시간",
                "알림 수",
                "앱 실행 횟수",
            ],
            "값": [
                f"{daily:.1f}시간",
                f"{sns_time:.1f}시간",
                f"{game:.1f}시간",
                f"{sleep:.1f}시간",
                f"{notifications}회",
                f"{app_opens}회",
            ],
        }
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.subheader("목표 설정")
    goal_col1, goal_col2, goal_col3 = st.columns(3)
    with goal_col1:
        target_daily = st.number_input("목표 일일 사용시간", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
    with goal_col2:
        target_sleep = st.number_input("목표 수면 시간", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
    with goal_col3:
        target_notifications = st.number_input("목표 알림 수", min_value=0, max_value=300, value=80, step=5)

    if st.button("예측 및 저장", use_container_width=True):
        data = np.array([[daily, sns_time, game, sleep, notifications, app_opens]])
        data_scaled = scaler.transform(data)

        prob = float(model.predict(data_scaled, verbose=0)[0][0])
        risk = classify_risk(prob)

        llm_text, prompt_text = generate_single_analysis(
            daily_time=daily,
            sns_time=sns_time,
            game_time=game,
            sleep_time=sleep,
            notifications=notifications,
            app_opens=app_opens,
            probability=prob,
            risk_level=risk,
        )

        st.subheader("예측 결과")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("중독 확률", f"{prob * 100:.2f}%")
            st.progress(max(0.0, min(1.0, prob)))
        with metric_col2:
            st.metric("위험도", risk)

        st.subheader("목표 비교")
        compare_goal_df = pd.DataFrame(
            {
                "항목": ["일일 사용시간", "수면 시간", "알림 수"],
                "현재값": [daily, sleep, notifications],
                "목표값": [target_daily, target_sleep, target_notifications],
                "판정": [
                    "달성" if daily <= target_daily else "초과",
                    "달성" if sleep >= target_sleep else "미달",
                    "달성" if notifications <= target_notifications else "초과",
                ],
            }
        )
        st.dataframe(compare_goal_df, hide_index=True, use_container_width=True)

        st.subheader("LLM 분석 결과")
        st.write(llm_text)

        with st.expander("프롬프트 보기"):
            st.code(prompt_text, language="text")

        record = build_record(
            daily=daily,
            sns=sns_time,
            game=game,
            sleep=sleep,
            notifications=notifications,
            app_opens=app_opens,
            prob=prob,
            risk=risk,
            llm_text=llm_text,
        )
        append_history(record)

        pdf_bytes = build_single_result_pdf(record, compare_goal_df, prompt_text)
        st.download_button(
            "예측 결과 PDF 저장",
            data=pdf_bytes,
            file_name=f"single_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )

with tab2:
    st.subheader("기록 조회")
    hist_df = history_to_df()

    if hist_df.empty:
        st.info("저장된 기록이 없습니다.")
    else:
        view_df = hist_df.copy()
        view_df["created_at"] = view_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(view_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("변화 추이")
    hist_df = history_to_df()

    if hist_df.empty:
        st.info("추이를 표시할 기록이 없습니다.")
    else:
        trend_df = hist_df.sort_values("created_at").copy()
        trend_df["날짜"] = trend_df["created_at"].dt.strftime("%m-%d %H:%M")

        st.write("중독 확률 추이")
        st.line_chart(trend_df.set_index("날짜")["probability"])

        st.write("일일 사용시간 추이")
        st.line_chart(trend_df.set_index("날짜")["daily_screen_time_hours"])

        st.write("수면 시간 추이")
        st.line_chart(trend_df.set_index("날짜")["sleep_hours"])

with tab4:
    st.subheader("AI 관리 리포트")
    hist_df = history_to_df()

    if hist_df.empty:
        st.info("관리 리포트를 생성할 기록이 없습니다.")
    else:
        days = st.selectbox("분석 기간", [7, 14, 30], index=0)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        report_df = hist_df[hist_df["created_at"] >= cutoff].sort_values("created_at")

        if report_df.empty:
            st.info("선택한 기간에 해당하는 기록이 없습니다.")
        else:
            report_text, report_prompt = generate_weekly_report(report_df)
            st.write(report_text)

            with st.expander("리포트 생성 프롬프트 보기"):
                st.code(report_prompt, language="text")

            report_pdf = build_weekly_report_pdf(report_df, report_text, days)
            st.download_button(
                "관리 리포트 PDF 저장",
                data=report_pdf,
                file_name=f"weekly_report_{days}days.pdf",
                mime="application/pdf",
            )

with tab5:
    st.subheader("실제값 vs 예측값")
    X = df[FEATURES]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X_test)
    y_prob = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    compare_df = pd.DataFrame(
        {
            "실제값": y_test.values,
            "예측값": y_pred,
            "확률": y_prob,
        }
    )
    compare_df["판정"] = np.where(compare_df["실제값"] == compare_df["예측값"], "정확", "오분류")

    total = len(compare_df)
    correct = int((compare_df["판정"] == "정확").sum())
    wrong = total - correct

    c1, c2, c3 = st.columns(3)
    c1.metric("전체", total)
    c2.metric("정확", correct)
    c3.metric("오분류", wrong)

    filter_opt = st.radio("보기", ["전체", "정확", "오분류"], horizontal=True)

    if filter_opt == "정확":
        show_df = compare_df[compare_df["판정"] == "정확"]
    elif filter_opt == "오분류":
        show_df = compare_df[compare_df["판정"] == "오분류"]
    else:
        show_df = compare_df

    st.dataframe(show_df.head(50), use_container_width=True, hide_index=True)