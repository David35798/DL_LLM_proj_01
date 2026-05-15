import os
from typing import Tuple

import pandas as pd
import google.generativeai as genai


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAkOnOmZbjrN3isUbmN9PPXItmYmlZWJRE")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


def _get_model():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
    if not GOOGLE_MODEL:
        raise RuntimeError("GOOGLE_MODEL 환경변수가 설정되지 않았습니다.")
    return genai.GenerativeModel(GOOGLE_MODEL)


def generate_single_analysis(
    daily_time: float,
    sns_time: float,
    game_time: float,
    sleep_time: float,
    notifications: int,
    app_opens: int,
    probability: float,
    risk_level: str,
) -> Tuple[str, str]:
    prompt = f"""
당신은 스마트폰 사용 습관 관리 코치입니다.
아래 데이터를 바탕으로 사용자의 행동 패턴을 분석하고,
문제점과 개선 방법을 한국어로 3~5문장 정도로 설명하세요.

[사용자 행동 데이터]
- 일일 사용시간: {daily_time:.1f}시간
- SNS 사용시간: {sns_time:.1f}시간
- 게임 사용시간: {game_time:.1f}시간
- 수면 시간: {sleep_time:.1f}시간
- 알림 수: {notifications}회
- 앱 실행 횟수: {app_opens}회

[딥러닝 예측 결과]
- 중독 확률: {probability * 100:.2f}%
- 위험도: {risk_level}

조건:
- 과장된 표현 없이 구체적으로 작성
- 문제 원인 1~2개 포함
- 실천 가능한 개선 방법 2개 이상 포함
"""
    model = _get_model()
    response = model.generate_content(prompt)
    return response.text.strip(), prompt.strip()


def generate_weekly_report(report_df: pd.DataFrame) -> Tuple[str, str]:
    avg_daily = report_df["daily_screen_time_hours"].mean()
    avg_sleep = report_df["sleep_hours"].mean()
    avg_prob = report_df["probability"].mean() * 100
    risk_counts = report_df["risk_level"].value_counts().to_dict()

    prompt = f"""
당신은 스마트폰 사용 습관 관리 코치입니다.
아래 최근 기록 요약을 바탕으로 주기적 관리 리포트를 한국어로 작성하세요.

[기록 요약]
- 평균 일일 사용시간: {avg_daily:.2f}시간
- 평균 수면 시간: {avg_sleep:.2f}시간
- 평균 중독 확률: {avg_prob:.2f}%
- 위험도 분포: {risk_counts}

조건:
- 최근 사용 패턴의 특징 요약
- 위험 요인 설명
- 다음 주 실천 목표 2~3개 제안
- 자연스럽고 간결한 문장으로 작성
"""
    model = _get_model()
    response = model.generate_content(prompt)
    return response.text.strip(), prompt.strip()