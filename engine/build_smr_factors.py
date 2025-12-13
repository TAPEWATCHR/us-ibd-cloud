"""
build_smr_factors.py

- us_universe.csv 에 있는 전체 미국 종목을 대상으로
- yfinance.Ticker 를 이용해 최근 3~5년 재무 데이터를 가져온 뒤
    * 매출 성장률 (대략 3~5년간 연평균 성장률)
    * 최근 연도 이익률 (Net Margin)
    * 최근 연도 ROE
  를 계산해서 smr_factors.csv 로 저장한다.

※ 주의사항
- 야후 재무 데이터가 없는 종목도 많으므로, 그런 종목은 NaN 으로 남는다.
- 전체 6,000~7,000 종목에 대해 재무를 받으면 시간이 상당히 오래 걸 수 있다.
  (처음 구조 잡을 때는 괜찮지만, 매일 돌리기엔 무거울 수 있음)
"""

import os
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_FILE = os.path.join(BASE_DIR, "us_universe.csv")
OUT_FILE = os.path.join(BASE_DIR, "smr_factors.csv")

# 속도/제한 관련 설정
SLEEP_EACH_TICKER = 0.2   # 티커 하나 처리 후 잠깐 쉼 (rate limit 완화용)
LOG_EVERY = 100           # 몇 종목마다 진행 상황 로그 찍을지
MAX_TICKERS = None        # None 이면 전체, 숫자를 넣으면 상위 N개만 처리 (테스트 용)


def load_universe(path: str):
    """
    us_universe.csv에서 티커 목록을 읽어온다.
    - symbol / ticker / Symbol / Ticker / 종목코드 등 여러 이름을 허용
    - 최종적으로는 'symbol' 컬럼으로 통일해서 반환
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"유니버스 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)

    # 티커 컬럼 후보들 중 하나를 찾기
    symbol_col = None
    for c in ["symbol", "ticker", "Symbol", "Ticker", "종목코드"]:
        if c in df.columns:
            symbol_col = c
            break

    if symbol_col is None:
        raise ValueError(
            f"유니버스 파일에서 티커 컬럼(symbol/ticker)을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}"
        )

    # 내부적으로는 항상 'symbol'로 통일
    if symbol_col != "symbol":
        df = df.rename(columns={symbol_col: "symbol"})

    tickers = (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    if not tickers:
        raise ValueError("유니버스에서 티커를 한 개도 찾지 못했습니다.")

    return tickers


def _extract_annual_series(fin_df: pd.DataFrame, row_name_candidates) -> Optional[pd.Series]:
    """
    야후 annual financials 에서 특정 row (예: 'Total Revenue', 'Net Income') 를 찾아
    column 을 오래된 순서(좌측이 과거) -> 최근 순으로 정렬한 Series 를 반환.
    """
    if fin_df is None or fin_df.empty:
        return None

    index_lower = [str(idx).lower() for idx in fin_df.index]

    target_idx = None
    for cand in row_name_candidates:
        cand_lower = cand.lower()
        for idx, idx_lower_name in zip(fin_df.index, index_lower):
            if cand_lower == idx_lower_name:
                target_idx = idx
                break
        if target_idx is not None:
            break

    if target_idx is None:
        return None

    row = fin_df.loc[target_idx]

    # column(시기)을 오래된 순서 -> 최신 순으로 정렬
    # columns 는 DatetimeIndex 또는 str(YYYY-MM-DD) 형태일 수 있음
    cols = list(row.index)

    def _parse_col(c):
        try:
            return pd.to_datetime(c)
        except Exception:
            return c

    parsed = [(c, _parse_col(c)) for c in cols]

    # 날짜형만 있는 경우/섞여있는 경우에 대비해, 날짜형 우선 정렬
    # 날짜형이 아닌 경우는 그대로 뒤쪽에 둔다.
    date_part = [(c, d) for c, d in parsed if isinstance(d, pd.Timestamp)]
    other_part = [(c, d) for c, d in parsed if not isinstance(d, pd.Timestamp)]

    date_part_sorted = sorted(date_part, key=lambda x: x[1])  # 오래된 -> 최신

    sorted_cols = [c for c, _ in date_part_sorted] + [c for c, _ in other_part]

    series_sorted = row[sorted_cols]
    # 최신 값이 NaN 일 경우 뒤에서부터 유효값 찾기 등 추가 정제도 가능하지만,
    # 여기서는 그대로 둔다.
    return series_sorted


def compute_growth_from_series(series: pd.Series, min_years: int = 3) -> Optional[float]:
    """
    annual revenue series 를 받아서 대략 3~5년 성장률(CAGR 비슷한 것)을 계산.
    - min_years 이상 데이터가 없으면 None.
    """
    # NaN 제거
    ser = series.dropna()
    if len(ser) < min_years:
        return None

    # 가장 과거값과 최신값 선택 (최대 5년 정도 범위에서)
    # 예를 들어 5개 이상 있으면 마지막과 -5번째, 아니면 처음/마지막
    if len(ser) >= 5:
        base = ser.iloc[-5]
        latest = ser.iloc[-1]
        years = 5
    else:
        base = ser.iloc[0]
        latest = ser.iloc[-1]
        # 대략 연도 수 추정 (컬럼 간 1년씩 차이로 가정)
        years = len(ser) - 1
        if years <= 0:
            return None

    if base <= 0 or latest <= 0:
        return None

    try:
        cagr = (latest / base) ** (1 / years) - 1
        return float(cagr)
    except Exception:
        return None


def compute_smr_factors_for_ticker(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    단일 티커에 대해
    - sales_growth (최근 3~5년 매출 CAGR 비슷하게)
    - profit_margin (최근 연도 Net Income / Revenue)
    - roe (최근 연도 Net Income / Equity)
    를 계산.
    """
    try:
        tk = yf.Ticker(ticker)
        fin = tk.financials        # annual
        bs = tk.balance_sheet      # annual balance sheet
    except Exception as e:
        print(f"  [SMR SKIP] {ticker}: 재무 데이터 로딩 오류: {e}")
        return None, None, None

    if fin is None or fin.empty:
        print(f"  [SMR SKIP] {ticker}: financials 가 비어 있음")
        return None, None, None

    # 매출 시리즈
    revenue_series = _extract_annual_series(
        fin,
        ["Total Revenue", "Revenue", "Total Revenue From Operations"]
    )

    # 순이익 시리즈
    net_income_series = _extract_annual_series(
        fin,
        ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"]
    )

    # 기본값
    sales_growth = None
    profit_margin = None
    roe = None

    # 1) 매출 성장률
    if revenue_series is not None:
        sales_growth = compute_growth_from_series(revenue_series, min_years=3)

    # 최신 연도 기준 값 추출
    latest_revenue = None
    latest_net_income = None

    if revenue_series is not None:
        # 뒤에서부터 유효값 찾기
        for v in reversed(revenue_series.values):
            if pd.notna(v):
                latest_revenue = float(v)
                break

    if net_income_series is not None:
        for v in reversed(net_income_series.values):
            if pd.notna(v):
                latest_net_income = float(v)
                break

    # 2) 이익률 (Net Margin)
    if latest_revenue and latest_net_income is not None and latest_revenue != 0:
        profit_margin = latest_net_income / latest_revenue

    # 3) ROE
    if bs is not None and not bs.empty and latest_net_income is not None:
        equity_series = _extract_annual_series(
            bs,
            ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Total Equity"]
        )
        if equity_series is not None:
            latest_equity = None
            for v in reversed(equity_series.values):
                if pd.notna(v):
                    latest_equity = float(v)
                    break
            if latest_equity and latest_equity != 0:
                roe = latest_net_income / latest_equity

    return sales_growth, profit_margin, roe


def main():
    print("=== SMR 팩터 계산 시작 (build_smr_factors.py) ===")

    tickers = load_universe(UNIVERSE_FILE)
    if MAX_TICKERS is not None:
        tickers = tickers[:MAX_TICKERS]
        print(f"[INFO] 테스트용으로 상위 {MAX_TICKERS}개 티커만 처리합니다.")

    rows = []
    total = len(tickers)

    for i, tkr in enumerate(tickers, start=1):
        print(f"[{i}/{total}] {tkr} SMR 팩터 계산 중...")
        try:
            sg, pm, r = compute_smr_factors_for_ticker(tkr)
        except Exception as e:
            print(f"  [SMR SKIP] {tkr}: 예외 발생: {e}")
            sg, pm, r = None, None, None

        rows.append(
            {
                "symbol": tkr,
                "sales_growth": sg,
                "profit_margin": pm,
                "roe": r,
            }
        )

        if i % LOG_EVERY == 0:
            print(f"[INFO] {i}/{total} 종목 처리 완료...")

        time.sleep(SLEEP_EACH_TICKER)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[INFO] SMR 팩터 파일 저장 완료: {OUT_FILE}")
    print("=== SMR 팩터 계산 종료 ===")


if __name__ == "__main__":
    main()
