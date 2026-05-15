from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


def _register_font():
    try:
        pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
        return "Malgun"
    except Exception:
        return "Helvetica"


def build_single_result_pdf(record: dict, compare_goal_df, prompt_text: str) -> bytes:
    font_name = _register_font()
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 40
    c.setFont(font_name, 16)
    c.drawString(40, y, "스마트폰 중독 예측 결과 리포트")

    y -= 30
    c.setFont(font_name, 10)
    c.drawString(40, y, f"생성시각: {record['created_at']}")

    y -= 30
    c.setFont(font_name, 12)
    c.drawString(40, y, f"중독 확률: {record['probability'] * 100:.2f}%")
    y -= 20
    c.drawString(40, y, f"위험도: {record['risk_level']}")

    y -= 30
    c.setFont(font_name, 11)
    fields = [
        ("일일 사용시간", record["daily_screen_time_hours"]),
        ("SNS 사용시간", record["social_media_hours"]),
        ("게임 사용시간", record["gaming_hours"]),
        ("수면 시간", record["sleep_hours"]),
        ("알림 수", record["notifications_per_day"]),
        ("앱 실행 횟수", record["app_opens_per_day"]),
    ]
    for label, value in fields:
        c.drawString(40, y, f"{label}: {value}")
        y -= 18

    y -= 10
    c.setFont(font_name, 12)
    c.drawString(40, y, "목표 비교")
    y -= 20
    c.setFont(font_name, 10)
    for _, row in compare_goal_df.iterrows():
        c.drawString(40, y, f"{row['항목']} | 현재값: {row['현재값']} | 목표값: {row['목표값']} | 판정: {row['판정']}")
        y -= 16

    y -= 10
    c.setFont(font_name, 12)
    c.drawString(40, y, "LLM 분석 결과")
    y -= 20
    c.setFont(font_name, 10)
    for line in record["llm_analysis"].split("\n"):
        if y < 60:
            c.showPage()
            c.setFont(font_name, 10)
            y = height - 40
        c.drawString(40, y, line[:100])
        y -= 15

    y -= 10
    c.setFont(font_name, 12)
    c.drawString(40, y, "프롬프트")
    y -= 20
    c.setFont(font_name, 9)
    for line in prompt_text.split("\n"):
        if y < 60:
            c.showPage()
            c.setFont(font_name, 9)
            y = height - 40
        c.drawString(40, y, line[:110])
        y -= 13

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


def build_weekly_report_pdf(report_df, report_text: str, days: int) -> bytes:
    font_name = _register_font()
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 40
    c.setFont(font_name, 16)
    c.drawString(40, y, f"{days}일 관리 리포트")

    y -= 30
    c.setFont(font_name, 10)
    c.drawString(40, y, f"기록 수: {len(report_df)}건")

    y -= 30
    c.setFont(font_name, 12)
    c.drawString(40, y, "요약 통계")
    y -= 20
    c.setFont(font_name, 10)
    c.drawString(40, y, f"평균 일일 사용시간: {report_df['daily_screen_time_hours'].mean():.2f}시간")
    y -= 16
    c.drawString(40, y, f"평균 수면 시간: {report_df['sleep_hours'].mean():.2f}시간")
    y -= 16
    c.drawString(40, y, f"평균 중독 확률: {report_df['probability'].mean() * 100:.2f}%")

    y -= 30
    c.setFont(font_name, 12)
    c.drawString(40, y, "LLM 관리 리포트")
    y -= 20
    c.setFont(font_name, 10)
    for line in report_text.split("\n"):
        if y < 60:
            c.showPage()
            c.setFont(font_name, 10)
            y = height - 40
        c.drawString(40, y, line[:100])
        y -= 15

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf