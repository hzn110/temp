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
st.write(
    "서울의 과거 연평균기온을 바탕으로 회귀 직선을 만들어 "
    "선택한 연도의 예상 평균기온을 확인합니다."
)

# -----------------------------------
# 데이터 불러오기
# -----------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# -----------------------------------
# 2025년까지의 자료만 사용
# -----------------------------------
df = df[df["연도"] <= 2025].copy()

# 평균기온이 존재하는 자료만 사용
valid_temp = df.dropna(subset=["평균기온"]).copy()

# 연도별 관측일 수 계산
year_count = valid_temp.groupby("연도").size()

# 관측일이 300일 이상인 연도만 사용
valid_years = year_count[year_count >= 300].index

filtered_df = valid_temp[
    valid_temp["연도"].isin(valid_years)
].copy()

# -----------------------------------
# 연도별 평균기온
# -----------------------------------
annual = (
    filtered_df
    .groupby("연도")["평균기온"]
    .mean()
    .reset_index()
    .rename(columns={"평균기온": "연평균기온"})
    .sort_values("연도")
    .reset_index(drop=True)
)

# -----------------------------------
# 전체 기간 회귀
# -----------------------------------
x_all = annual["연도"].to_numpy()
y_all = annual["연평균기온"].to_numpy()

slope_all, intercept_all = np.polyfit(x_all, y_all, 1)

# 100년당 상승 기온
rise_100_all = slope_all * 100

# 전체 기간 상관계수
correlation = np.corrcoef(x_all, y_all)[0, 1]

# -----------------------------------
# 최근 20년 회귀
# 2006~2025년
# -----------------------------------
recent_start = 2025 - 19
recent_end = 2025

recent = annual[
    (annual["연도"] >= recent_start) &
    (annual["연도"] <= recent_end)
].copy()

x_recent = recent["연도"].to_numpy()
y_recent = recent["연평균기온"].to_numpy()

slope_recent, intercept_recent = np.polyfit(
    x_recent,
    y_recent,
    1
)

rise_100_recent = slope_recent * 100

# -----------------------------------
# 연도 슬라이더
# -----------------------------------
selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 전체 기간 회귀를 이용한 예측
predicted_temp = slope_all * selected_year + intercept_all

# -----------------------------------
# 주요 정보
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
# 100년당 기온 상승 비교
# -----------------------------------
st.subheader("📈 100년당 기온 상승 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 기간",
        f"{rise_100_all:.2f} °C / 100년"
    )
    st.caption(
        f"{annual['연도'].min()}~{annual['연도'].max()}년 자료"
    )

with col2:
    st.metric(
        "최근 20년",
        f"{rise_100_recent:.2f} °C / 100년"
    )
    st.caption(
        f"{recent_start}~{recent_end}년 자료"
    )

# -----------------------------------
# 회귀 직선 정보
# -----------------------------------
st.info(
    f"""
**전체 기간:** {annual['연도'].min()}~{annual['연도'].max()}년
→ {len(annual)}개 연도 사용

**최근 20년:** {recent_start}~{recent_end}년
→ {len(recent)}개 연도 사용

**전체 기간 100년당 변화:** {rise_100_all:.2f} °C

**최근 20년 100년당 변화:** {rise_100_recent:.2f} °C
"""
)

# -----------------------------------
# 그래프용 전체 회귀 직선
# -----------------------------------
line_start = annual["연도"].min()
line_end = max(annual["연도"].max(), selected_year)

line_years = np.linspace(
    line_start,
    line_end,
    300
)

line_temps = (
    slope_all * line_years
    + intercept_all
)

# -----------------------------------
# 그래프
# -----------------------------------
fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        hovertemplate=(
            "%{x}년<br>"
            "연평균기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
        marker=dict(size=7)
    )
)

# 전체 기간 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "%{x:.0f}년<br>"
            "회귀값: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 선택 연도의 예상값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(
            size=14,
            symbol="star"
        ),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상 평균기온: {predicted_temp:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연도별 평균기온과 전체 기간 회귀 직선",
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

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------
# 데이터 처리 기준
# -----------------------------------
st.subheader("📌 데이터 처리 기준")

st.write(
    f"""
- **기준 기간:** 2025년까지
- **제외 기준:** 연간 관측일이 300일 미만인 연도
- **연평균기온:** 해당 연도의 유효한 평균기온 관측값의 평균
- **전체 기간 회귀:** {annual['연도'].min()}~{annual['연도'].max()}년
- **최근 20년 회귀:** {recent_start}~{recent_end}년
- **예측 범위:** 1900~2100년
"""
)

st.caption(
    "※ 2026년 이후 값은 실제 관측값이 아니라 "
    "과거 자료를 이용한 선형 회귀를 단순 외삽한 예상값입니다."
)
