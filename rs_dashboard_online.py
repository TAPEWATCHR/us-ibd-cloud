# rs_dashboard_online.py
#
# 온라인용 US IBD 스타일 대시보드 (RS + 산업군 RS + SMR)
# - GitHub에 올라간 latest_rs_smr.csv / latest_industry_rs.csv 읽기
# - 비밀번호(secrets.APP_PASSWORD) 잠금
# - 개별 RS + 산업군 RS + SMR + TradingView + 분기 재무제표
#
# *** 중요 변경점 ***
# 1) RS/산업군 CSV 로딩에서 @st.cache_data 제거 → 항상 최신 파일을 다시 읽도록
# 2) 실제로 읽힌 컬럼들을 DEBUG 영역에 표시 → SMR 컬럼 포함 여부 눈으로 확인 가능

from __future__ import annotations

from typing import Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# === 1) GitHub raw URL 설정 (반드시 latest_*.csv를 가리키도록!) ===
# 아래 두 줄을 네 GitHub 주소에 맞게 정확히 바꿔줘야 한다.
RS_URL = "https://raw.githubusercontent.com/mbkk7ch6kh-hub/us-ibd-cloud/main/data/latest_rs_smr.csv"
IND_URL = "https://raw.githubusercontent.com/mbkk7ch6kh-hub/us-ibd-cloud/main/data/latest_industry_rs.csv"


# === 2) 비밀번호 잠금 로직 ===
def check_password() -> bool:
    """간단한 1인용 비밀번호 보호."""
    # 시크릿 체크 (없으면 친절히 에러)
    if "APP_PASSWORD" not in st.secrets:
        st.error(
            "서버에 APP_PASSWORD 시크릿이 설정되어 있지 않습니다.\n"
            "Streamlit Cloud → Settings → Secrets에서 APP_PASSWORD 를 지정해 주세요."
        )
        return False

    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_ok"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_ok"] = False

    if "password_ok" not in st.session_state:
        st.text_input(
            "비밀번호를 입력하세요",
            type="password",
            key="password",
            on_change=password_entered,
        )
        return False
    elif not st.session_state["password_ok"]:
        st.text_input(
            "비밀번호가 틀렸습니다. 다시 입력하세요",
            type="password",
            key="password",
            on_change=password_entered,
        )
        st.error("비밀번호가 틀렸습니다.")
        return False
    else:
        return True


# === 3) 데이터 로딩 함수 (캐시 제거 → 항상 최신 파일 사용) ===
def load_rs_from_cloud() -> pd.DataFrame:
    df = pd.read_csv(RS_URL)
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def load_industry_from_cloud() -> pd.DataFrame | None:
    try:
        df = pd.read_csv(IND_URL)
    except Exception:
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    if "group_name" not in df.columns and "group_key" in df.columns:
        df = df.rename(columns={"group_key": "group_name"})
    return df


@st.cache_data(show_spinner=False)
def load_quarterly_financials(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    yfinance에서 분기 재무제표를 가져와 정리.
    fin_q: 분기 손익계산서
    bs_q : 분기 재무상태표
    cf_q : 분기 현금흐름표
    """
    def tidy(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df = df.transpose()
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        df = df.sort_index(ascending=False)
        return df

    try:
        t = yf.Ticker(ticker)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        fin_q_raw = t.quarterly_financials
    except Exception:
        fin_q_raw = pd.DataFrame()

    try:
        bs_q_raw = t.quarterly_balance_sheet
    except Exception:
        bs_q_raw = pd.DataFrame()

    try:
        cf_q_raw = t.quarterly_cashflow
    except Exception:
        cf_q_raw = pd.DataFrame()

    fin_q = tidy(fin_q_raw)
    bs_q = tidy(bs_q_raw)
    cf_q = tidy(cf_q_raw)

    return fin_q, bs_q, cf_q


# === 4) 메인 앱 ===
def main():
    st.set_page_config(
        page_title="US IBD RS Online",
        layout="wide",
    )

    # 비밀번호 체크
    if not check_password():
        st.stop()

    st.title("US IBD RS Online Dashboard 🔐")

    rs_df = load_rs_from_cloud()
    industry_df = load_industry_from_cloud()

    if rs_df is None or rs_df.empty:
        st.error("RS 데이터(latest_rs_smr.csv)를 불러오지 못했습니다. GitHub data 폴더를 확인해 주세요.")
        return

    st.caption(
        "데이터 출처: GitHub latest_rs_smr.csv / latest_industry_rs.csv\n"
        f"RS_URL = {RS_URL}"
    )

    # 🔍 DEBUG: 실제 컬럼들 표시 (SMR 들어왔는지 눈으로 확인)
    with st.expander("DEBUG: RS 데이터 컬럼 보기", expanded=False):
        st.write(rs_df.columns.tolist())

    total_count = len(rs_df)

    # 필수 컬럼 확인
    required_cols = {"ticker", "rs_onil"}
    missing = required_cols - set(rs_df.columns)
    if missing:
        st.error(
            f"필수 컬럼 {missing} 이(가) 없습니다. calc_rs_onil.py + enrich_smr.py 결과를 확인해 주세요.\n"
            f"현재 컬럼: {rs_df.columns.tolist()}"
        )
        return

    # SMR/산업군 관련 컬럼이 없어도 코드가 죽지 않도록 기본 생성
    optional_cols = [
        "sector",
        "industry",
        "group_rank",
        "group_rs_99",
        "group_rs_100",
        "group_grade",
        "onil_weighted_ret",
        "ret_3m",
        "ret_6m",
        "rs_onil_99",
        "last_close",
        "avg_dollar_vol_50",
        "s_raw",
        "m_raw",
        "r_raw",
        "s_pct",
        "m_pct",
        "r_pct",
        "smr_score",
        "smr_grade",
    ]
    for col in optional_cols:
        if col not in rs_df.columns:
            rs_df[col] = None

    # ---------- 사이드바: 필터 ----------
    st.sidebar.header("가격 / 거래대금 필터")

    min_price = st.sidebar.slider(
        "최소 주가 (USD, 이 값 미만 제외)",
        0.0,
        100.0,
        15.0,
        step=0.5,
    )

    min_dollar_vol_m = st.sidebar.slider(
        "최소 평균 거래대금 (최근 50일, 백만 달러)",
        0.0,
        500.0,
        25.0,
        step=5.0,
    )
    min_dollar_vol = min_dollar_vol_m * 1_000_000

    st.sidebar.header("RS / 추세 필터")
    min_rs = st.sidebar.slider("개별 RS 최소값 (rs_onil, 0~100)", 0.0, 100.0, 0.0)

    min_ret_3m_pct = st.sidebar.slider(
        "최근 3개월 최소 수익률 (%)",
        -100.0,
        200.0,
        -100.0,
        step=5.0,
    )
    min_ret_6m_pct = st.sidebar.slider(
        "최근 6개월 최소 수익률 (%)",
        -100.0,
        200.0,
        -100.0,
        step=5.0,
    )

    st.sidebar.header("SMR 필터")
    if rs_df["smr_score"].notna().any():
        min_smr_score = st.sidebar.slider(
            "SMR 최소 점수 (0~100)",
            0.0,
            100.0,
            0.0,
            step=1.0,
        )
    else:
        min_smr_score = 0.0

    smr_grade_choices = ["A", "B", "C", "D", "E"]
    selected_smr_grades = st.sidebar.multiselect(
        "허용 SMR 등급 (비선택 시 전체)",
        smr_grade_choices,
        default=[],
    )

    st.sidebar.header("산업군 필터")
    has_group_rs = rs_df["group_rs_99"].notna().any()
    has_group_rank = rs_df["group_rank"].notna().any()
    has_group_grade = rs_df["group_grade"].notna().any()

    if has_group_rs:
        min_group_rs = st.sidebar.slider(
            "산업군 RS 최소값 (group_rs_99, 1~99)",
            1,
            99,
            1,
        )
    else:
        min_group_rs = 1

    if has_group_rank:
        max_rank_val = int(rs_df["group_rank"].dropna().max())
        max_group_rank_sel = st.sidebar.slider(
            "허용 최대 산업군 순위 (1이 최상)",
            1,
            max_rank_val,
            max_rank_val,
            step=1,
        )
    else:
        max_group_rank_sel = None

    if has_group_grade:
        grade_choices = ["A", "B", "C", "D", "E"]
        selected_grades = st.sidebar.multiselect(
            "허용 산업군 등급 (비선택 시 전체)",
            grade_choices,
            default=[],
        )
    else:
        selected_grades = []

    st.sidebar.header("섹터 / 산업군 검색")
    sector_list = sorted(rs_df["sector"].dropna().unique()) if "sector" in rs_df.columns else []
    if sector_list:
        selected_sectors = st.sidebar.multiselect("섹터 선택", sector_list, default=[])
    else:
        selected_sectors = []

    industry_query = st.sidebar.text_input("산업군 이름 검색 (부분 일치, industry)", "")

    st.sidebar.header("표시 개수")
    show_all = st.sidebar.checkbox("필터 후 전체 보기", value=True)
    max_n = int(len(rs_df))
    top_n = st.sidebar.slider(
        "상위 N개까지 보기 (RS 기준)",
        10,
        max(10, max_n),
        min(200, max_n),
        step=10,
    )

    # ---------- 필터 적용 ----------
    df = rs_df.copy()

    if "last_close" in df.columns:
        df = df[df["last_close"].fillna(0) >= min_price]

    if "avg_dollar_vol_50" in df.columns:
        df = df[df["avg_dollar_vol_50"].fillna(0) >= min_dollar_vol]

    df = df[df["rs_onil"] >= min_rs]

    if "ret_3m" in df.columns and min_ret_3m_pct > -100.0:
        df = df[df["ret_3m"].fillna(-999) >= (min_ret_3m_pct / 100.0)]
    if "ret_6m" in df.columns and min_ret_6m_pct > -100.0:
        df = df[df["ret_6m"].fillna(-999) >= (min_ret_6m_pct / 100.0)]

    if rs_df["smr_score"].notna().any():
        df = df[df["smr_score"].fillna(-1) >= min_smr_score]

    if selected_smr_grades:
        df = df[df["smr_grade"].isin(selected_smr_grades)]

    if has_group_rs:
        df = df[df["group_rs_99"].fillna(1) >= min_group_rs]

    if has_group_rank and max_group_rank_sel is not None:
        df = df[df["group_rank"].fillna(max_group_rank_sel) <= max_group_rank_sel]

    if selected_grades and has_group_grade:
        df = df[df["group_grade"].isin(selected_grades)]

    if selected_sectors and "sector" in df.columns:
        df = df[df["sector"].isin(selected_sectors)]

    if industry_query and "industry" in df.columns:
        q = industry_query.strip().lower()
        df = df[df["industry"].fillna("").str.lower().str.contains(q)]

    df = df.sort_values("rs_onil", ascending=False).reset_index(drop=True)

    if show_all:
        filtered_top = df
    else:
        filtered_top = df.head(top_n)

    st.caption(
        f"필터 적용 전 종목 수: {total_count}개 / "
        f"필터 후: {len(filtered_top)}개"
    )

    if filtered_top.empty:
        st.info("현재 필터 조건에 해당하는 종목이 없습니다.")
        return

    # ---------- 공통: 종목 선택 ----------
    st.subheader("분석할 종목 선택")
    selected_ticker = st.selectbox(
        "필터된 리스트 중에서 종목 선택",
        filtered_top["ticker"].tolist(),
    )

    tab_rs, tab_chart, tab_fund = st.tabs(["RS · 산업군 · SMR", "차트 (TradingView)", "재무/지표"])

    # ---------- 탭 1: RS / 산업군 / SMR ----------
    with tab_rs:
        st.subheader("필터링된 종목 리스트")

        show_df = filtered_top.copy()

        if "ret_3m" in show_df.columns:
            show_df["ret_3m(%)"] = show_df["ret_3m"] * 100
        if "ret_6m" in show_df.columns:
            show_df["ret_6m(%)"] = show_df["ret_6m"] * 100
        if "onil_weighted_ret" in show_df.columns:
            show_df["onil_weighted_ret(%)"] = show_df["onil_weighted_ret"] * 100
        if "avg_dollar_vol_50" in show_df.columns:
            show_df["avg_dollar_vol_50(M$)"] = show_df["avg_dollar_vol_50"] / 1_000_000

        if "s_raw" in show_df.columns:
            show_df["S(매출성장,%)"] = show_df["s_raw"] * 100
        if "m_raw" in show_df.columns:
            show_df["M(이익률,%)"] = show_df["m_raw"] * 100
        if "r_raw" in show_df.columns:
            show_df["R(ROE,%)"] = show_df["r_raw"] * 100

        display_cols = [
            c
            for c in [
                "ticker",
                "last_close",
                "avg_dollar_vol_50(M$)",
                "sector",
                "industry",
                "rs_onil",
                "rs_onil_99",
                "ret_3m(%)",
                "ret_6m(%)",
                "smr_score",
                "smr_grade",
                "S(매출성장,%)",
                "M(이익률,%)",
                "R(ROE,%)",
                "group_rank",
                "group_rs_99",
                "group_grade",
                "onil_weighted_ret(%)",
            ]
            if c in show_df.columns
        ]

        st.dataframe(
            show_df[display_cols],
            use_container_width=True,
            height=350,
        )

        if industry_df is not None and not industry_df.empty:
            st.subheader("산업군 RS / 랭크 / 등급 목록")

            ind_df = industry_df.copy()
            if "group_rs_99" not in ind_df.columns:
                if "group_rs_100" in ind_df.columns:
                    ind_df["group_rs_99"] = ind_df["group_rs_100"]
                elif "group_rs_6m" in ind_df.columns:
                    ind_df["group_rs_99"] = ind_df["group_rs_6m"]

            sort_by = st.selectbox(
                "산업군 정렬 기준",
                ["group_rank", "group_rs_99", "group_rs_100", "avg_ret_6m"],
                index=0,
            )

            ind_display_cols = [
                c
                for c in [
                    "group_rank",
                    "group_name",
                    "group_grade",
                    "group_rs_99",
                    "group_rs_100",
                    "avg_ret_6m",
                    "n_members",
                ]
                if c in ind_df.columns
            ]

            if sort_by == "group_rank":
                ind_df = ind_df.sort_values(sort_by, ascending=True)
            else:
                ind_df = ind_df.sort_values(sort_by, ascending=False)

            max_ind = len(ind_df)
            n_ind = st.slider(
                "표시할 산업군 개수",
                5,
                max(10, max_ind),
                min(30, max_ind),
                step=5,
                key="industry_n_online",
            )

            st.dataframe(
                ind_df[ind_display_cols].head(n_ind),
                use_container_width=True,
                height=400,
            )

    # ---------- 탭 2: 차트 (TradingView) ----------
    with tab_chart:
        st.subheader("TradingView 차트")

        default_symbol = f"NASDAQ:{selected_ticker}"
        tv_symbol = st.text_input(
            "TradingView 심볼 (예: NASDAQ:AAPL, NYSE:MS 등)",
            value=default_symbol,
            key="tv_symbol_online",
        )

        widget_id = f"tradingview_{selected_ticker}".replace(".", "_")
        tv_html = """
        <div class="tradingview-widget-container">
          <div id="{widget_id}"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
              "width": "100%",
              "height": 650,
              "symbol": "{symbol}",
              "interval": "D",
              "timezone": "Etc/UTC",
              "theme": "light",
              "style": "1",
              "locale": "kr",
              "toolbar_bg": "#f1f3f6",
              "enable_publishing": false,
              "allow_symbol_change": true,
              "save_image": false,
              "container_id": "{widget_id}"
          }});
          </script>
        </div>
        """.format(symbol=tv_symbol, widget_id=widget_id)

        components.html(tv_html, height=670)

    # ---------- 탭 3: 재무/지표 ----------
    with tab_fund:
        st.subheader("지표 요약 & 분기 재무제표")

        row = filtered_top[filtered_top["ticker"] == selected_ticker].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("개별 RS (rs_onil)", f"{row['rs_onil']:.1f}")
            if pd.notna(row.get("rs_onil_99", None)):
                st.metric("RS 점수 (1~99)", f"{int(row['rs_onil_99'])}")
            if pd.notna(row.get("onil_weighted_ret", None)):
                st.metric("12M 가중 수익률", f"{row['onil_weighted_ret']*100:.2f}%")
            if pd.notna(row.get("ret_3m", None)):
                st.metric("3M 수익률", f"{row['ret_3m']*100:.2f}%")
            if pd.notna(row.get("ret_6m", None)):
                st.metric("6M 수익률", f"{row['ret_6m']*100:.2f}%")

        with col2:
            if pd.notna(row.get("last_close", None)):
                st.metric("현재 주가", f"${row['last_close']:.2f}")
            if pd.notna(row.get("avg_dollar_vol_50", None)):
                st.metric(
                    "평균 거래대금(50일)",
                    f"{row['avg_dollar_vol_50']/1_000_000:.1f}M USD/일",
                )
            if pd.notna(row.get("group_rank", None)):
                st.metric("산업군 순위", f"{int(row['group_rank'])}")
            if pd.notna(row.get("group_rs_99", None)):
                st.metric("산업군 RS (1~99)", f"{int(row['group_rs_99'])}")
            if pd.notna(row.get("group_rs_100", None)):
                st.metric("산업군 RS (0~100)", f"{row['group_rs_100']:.1f}")
            st.write(f"산업군 등급: {row.get('group_grade', 'N/A')}")

        with col3:
            if pd.notna(row.get("smr_score", None)):
                st.metric("SMR 점수", f"{row['smr_score']:.1f}")
                st.write(f"SMR 등급: {row.get('smr_grade', 'N/A')}")
            if pd.notna(row.get("s_raw", None)):
                st.write(f"S · 매출 성장률: {row['s_raw']*100:.1f}%")
            if pd.notna(row.get("m_raw", None)):
                st.write(f"M · 이익률: {row['m_raw']*100:.1f}%")
            if pd.notna(row.get("r_raw", None)):
                st.write(f"R · ROE: {row['r_raw']*100:.1f}%")

            st.write(f"섹터: {row.get('sector', 'N/A')}")
            st.write(f"산업군: {row.get('industry', 'N/A')}")
            st.write(f"티커: {row.get('ticker', 'N/A')}")

            st.info("아래에 분기 손익·재무상태·현금흐름표를 표시합니다.")

        st.markdown("---")
        st.subheader(f"{selected_ticker} 분기 재무제표")

        with st.spinner("분기 재무제표 불러오는 중..."):
            fin_q, bs_q, cf_q = load_quarterly_financials(selected_ticker)

        st.markdown("#### 분기 손익계산서 (최근 12분기)")

        if fin_q is None or fin_q.empty:
            st.info("손익계산서 분기 데이터가 없습니다.")
        else:
            fin_show = fin_q.head(12).copy()
            if not fin_show.empty and isinstance(fin_show.index[0], pd.Timestamp):
                fin_show.index = fin_show.index.strftime("%Y-%m")
            fin_show.index.name = "Quarter"
            st.dataframe(fin_show, use_container_width=True, height=300)

        st.markdown("#### 분기 재무상태표 (최근 12분기)")

        if bs_q is None or bs_q.empty:
            st.info("재무상태표 분기 데이터가 없습니다.")
        else:
            bs_show = bs_q.head(12).copy()
            if not bs_show.empty and isinstance(bs_show.index[0], pd.Timestamp):
                bs_show.index = bs_show.index.strftime("%Y-%m")
            bs_show.index.name = "Quarter"
            st.dataframe(bs_show, use_container_width=True, height=300)

        st.markdown("#### 분기 현금흐름표 (최근 12분기)")

        if cf_q is None or cf_q.empty:
            st.info("현금흐름표 분기 데이터가 없습니다.")
        else:
            cf_show = cf_q.head(12).copy()
            if not cf_show.empty and isinstance(cf_show.index[0], pd.Timestamp):
                cf_show.index = cf_show.index.strftime("%Y-%m")
            cf_show.index.name = "Quarter"
            st.dataframe(cf_show, use_container_width=True, height=300)


if __name__ == "__main__":
    main()
