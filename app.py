import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------
# 기본 설정
# -----------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write("서울의 과거 연평균기온을 바탕으로 회귀 직선을 만들어 선택한 연도의 예상 평균기온을 확인합니다.")

# -----------------------------------
# 데이터 불러오기
# -----------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# -----------------------------------
# 기준 기간 적용
# 2025년까지의 자료만 사용
# -----------------------------------
df = df[df["연도"] <= 2025].copy()

# 평균기온이 존재하는 관측만 사용
valid_temp = df.dropna(subset=["평균기온"]).copy()

# 연도별 관측일 수
year_count = valid_temp.groupby("연도").size()

# 관측일이 300일 이상인 연도만 사용
valid_years = year_count[year_count >= 300].index

filtered_df = valid_temp[valid_temp["연도"].isin(valid_years)].copy()

# -----------------------------------
# 연도별 평균기온 계산
# -----------------------------------
annual = (
    filtered_df
    .groupby("연도")["평균기온"]
    .mean()
    .reset_index()
    .rename(columns={"평균기온": "연평균기온"})
)

annual = annual.sort_values("연도").reset_index(drop=True)

# -----------------------------------
# 회귀 직선 계산
# -----------------------------------
x = annual["연도"].to_numpy()
y = annual["연평균기온"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# -----------------------------------
# 슬라이더
# -----------------------------------
selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 선택 연도의 예상 기온
predicted_temp = slope * selected_year + intercept

# -----------------------------------
# 주요 정보 표시
# -----------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "선택 연도",
        f"{selected_year}년"
    )

with col2:
    st.metric(
        "예상 연평균기온",
        f"{predicted_temp:.2f} °C"
    )

with col3:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )

with col4:
    st.metric(
        "사용한 연도 수",
        f"{len(annual)}년"
    )

# -----------------------------------
# 회귀 직선 정보
# -----------------------------------
st.info(
    f"회귀 직선을 만든 기간: **{annual['연도'].min()}년 ~ {annual['연도'].max()}년**  |  "
    f"사용한 연도: **{len(annual)}개**  |  "
    f"회귀식: **연평균기온 = {slope:.4f} × 연도 + {intercept:.2f}**"
)

# -----------------------------------
# 그래프용 회귀 직선
# -----------------------------------
line_start = annual["연도"].min()
line_end = max(annual["연도"].max(), selected_year)

line_years = np.linspace(line_start, line_end, 300)
line_temps = slope * line_years + intercept

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
        marker=dict(size=7)
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
        hovertemplate="%{x:.0f}년<br>%{y:.2f} °C<extra></extra>"
    )
)

# 선택한 연도의 예상값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=14, symbol="star"),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상 평균기온: {predicted_temp:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연도별 평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# 설명
# -----------------------------------
st.subheader("📌 데이터 처리 기준")

st.write(
    f"""
- **기준 기간:** 2025년까지
- **제외 기준:** 한 해의 평균기온 관측일이 300일 미만인 연도
- **연평균기온:** 해당 연도의 유효한 평균기온 관측값의 평균
- **회귀:** 남은 연도별 연평균기온을 이용한 단순 선형회귀
- **상관계수:** 연도와 연평균기온 사이의 피어슨 상관계수
- **예측 범위:** 1900년 ~ 2100년
"""
)

st.caption(
    "※ 2026년 이후 값은 실제 관측값이 아니라 과거 자료로 만든 선형 회귀식을 단순 외삽한 예상값입니다."
)
