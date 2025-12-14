import os
import time
import datetime as dt
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_FILE = "us_universe.csv"
OUT_DIR = "."
PRICE_PERIOD = "500d"  # ~ 1.5년
CHUNK_SIZE = 200       # 한번에 요청할 티커 수
SLEEP_SEC = 1.0        # 청크 사이 딜레이


def load_universe(path: str) -> pd.DataFrame:
    """
    us_universe.csv에서 symbol, group_key를 읽어온다.
    - symbol / ticker / Symbol / Ticker / 종목코드 등 여러 이름을 허용
    - group_key가 없으면 sector/industry 조합으로 생성
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"유니버스 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)

    # 티커 컬럼 찾기
    symbol_col = None
    for c in ["symbol", "ticker", "Symbol", "Ticker", "종목코드"]:
        if c in df.columns:
            symbol_col = c
            break

    if symbol_col is None:
        raise ValueError(
            f"유니버스 파일에서 티커 컬럼(symbol/ticker)을 찾지 못했습니다. 현재 컬럼: {list(df.columns)}"
        )

    # 내부적으로 'symbol'로 통일
    if symbol_col != "symbol":
        df = df.rename(columns={symbol_col: "symbol"})

    # group_key 없으면 sector/industry 조합으로 생성
    if "group_key" not in df.columns:
        sector_col = None
        industry_col = None
        for c in ["sector", "Sector", "섹터"]:
            if c in df.columns:
                sector_col = c
                break
        for c in ["industry", "Industry", "산업", "industry_group"]:
            if c in df.columns:
                industry_col = c
                break

        if sector_col is not None and industry_col is not None:
            df["group_key"] = (
                df[sector_col].fillna("Unknown") + " | " + df[industry_col].fillna("Unknown")
            )
        elif sector_col is not None:
            df["group_key"] = df[sector_col].fillna("Unknown")
        elif industry_col is not None:
            df["group_key"] = df[industry_col].fillna("Unknown")
        else:
            df["group_key"] = "Unknown"

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.dropna(subset=["symbol"])
    df = df[df["symbol"] != ""]

    return df[["symbol", "group_key"]].drop_duplicates()


def download_prices_chunk(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    yfinance로 여러 티커를 한 번에 다운로드.
    반환값: {티커: DataFrame(OHLCV)}
    """
    result: Dict[str, pd.DataFrame] = {}

    if not tickers:
        return result

    tickers_str = " ".join(tickers)
    print(f"[DEBUG] yfinance.download 호출: {tickers_str[:80]}... (총 {len(tickers)}개)")

    try:
        data = yf.download(
            tickers=tickers_str,
            period=PRICE_PERIOD,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
        )
    except Exception as e:
        print(f"[ERROR] yfinance.download 예외 발생: {e}")
        return result

    if data is None or len(data) == 0:
        print("[WARN] yfinance에서 빈 데이터 프레임 반환")
        return result

    # MultiIndex 인지 여부 체크
    if isinstance(data.columns, pd.MultiIndex):
        # data[ticker] 가 각 티커별 서브프레임
        top_level = list(dict.fromkeys(data.columns.get_level_values(0)))
        print(f"[DEBUG] MultiIndex 컬럼 티커 수: {len(top_level)}")

        for t in tickers:
            if t not in top_level:
                print(f"  [SKIP] {t}: MultiIndex 컬럼에 없음")
                continue
            try:
                px = data[t].copy()
                px = px.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "close",
                        "Volume": "volume",
                    }
                )
                px = px.reset_index().rename(columns={"Date": "date"})
                result[t] = px
            except Exception as e:
                print(f"  [SKIP] {t}: 개별 티커 데이터 처리 중 예외 {e}")
                continue
    else:
        # 단일 티커일 때
        print("[DEBUG] 단일 티커 데이터 구조 감지")
        if len(tickers) != 1:
            print("[WARN] tickers는 여러 개인데 data는 단일 구조입니다.")
        t = tickers[0]
        try:
            px = data.copy()
            px = px.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "close",
                    "Volume": "volume",
                }
            )
            px = px.reset_index().rename(columns={"Date": "date"})
            result[t] = px
        except Exception as e:
            print(f"  [SKIP] {t}: 단일 구조 처리 중 예외 {e}")

    return result


def calc_returns_from_prices(px: pd.DataFrame):
    """
    px: columns 에서 'close' 역할을 하는 컬럼을 찾아
    3, 6, 9, 12개월 수익률과 보조 지표를 계산한다.
    - 'close' / 'adj close' 중복 컬럼도 안전하게 처리
    - 가격 데이터가 아주 적은 종목만 제외하고, 가능한 한 많이 살린다
    """
    # 날짜 컬럼 정리
    if "date" not in px.columns:
        if "Date" in px.columns:
            px = px.rename(columns={"Date": "date"})
        else:
            return None

    # 1) 종가 역할을 하는 컬럼 찾기
    close_candidate = None

    if "close" in px.columns:
        close_candidate = px["close"]
    else:
        # 이름에 'close' 가 들어가는 첫 번째 컬럼을 후보로 사용
        for c in px.columns:
            if isinstance(c, str) and "close" in c.lower():
                close_candidate = px[c]
                break

    if close_candidate is None:
        # 종가 역할을 하는 컬럼을 찾지 못하면 해당 티커는 스킵
        return None

    # 2) close_candidate 가 DataFrame 이면 첫 번째 컬럼만 사용
    if isinstance(close_candidate, pd.DataFrame):
        close_series = close_candidate.iloc[:, 0]
    else:
        close_series = close_candidate

    # 숫자로 강제 변환
    close_series = pd.to_numeric(close_series, errors="coerce")

    # 원본 프레임 복사 후, 사용할 종가를 별도 컬럼으로 고정
    px = px.copy()
    px["close_used"] = close_series

    # 3) 날짜 정렬
    px = px.sort_values("date").reset_index(drop=True)

    valid_close = px["close_used"].notna()
    n = int(valid_close.sum())

    # 🔎 최소 데이터 길이 조건 완화:
    #  - 30일 미만: 너무 짧아서 RS 의미 없다고 보고 제외
    #  - 30일 이상이면 "있는 범위 안에서" 3/6/9/12M 수익률 계산 (없으면 NaN)
    if n < 30:
        return None

    last_idx = px.index[valid_close].max()
    last_close = float(px.loc[last_idx, "close_used"])
    last_date = pd.to_datetime(px.loc[last_idx, "date"]).date()

    def ret_n(days: int):
        idx = last_idx - days
        if idx < 0:
            # 과거 데이터가 모자라면 해당 기간 수익률은 NaN 으로
            return np.nan
        base = px.loc[idx, "close_used"]
        if pd.isna(base) or base == 0:
            return np.nan
        return float(last_close / base - 1.0)

    # 대략적 거래일 기준 (3, 6, 9, 12개월)
    ret_3m = ret_n(63)
    ret_6m = ret_n(126)
    ret_9m = ret_n(189)
    ret_12m = ret_n(252)

    # 거래량 지표
    vol = pd.to_numeric(px.get("volume"), errors="coerce")
    avg_vol_50 = float(vol.tail(50).mean()) if vol.notna().sum() > 0 else np.nan
    avg_dollar_vol_50 = float(avg_vol_50 * last_close) if not np.isnan(avg_vol_50) else np.nan

    # 오닐식 가중 수익률 (12m*3 + 9m*2 + 6m + 3m) / 7
    # → 이용 가능한 기간만 사용 (예: 새 종목은 3M만 들어갈 수도 있음)
    weights = []
    vals = []
    for r, w in [(ret_12m, 3), (ret_9m, 2), (ret_6m, 1), (ret_3m, 1)]:
        if not np.isnan(r):
            vals.append(r * w)
            weights.append(w)

    if len(weights) == 0:
        weighted_ret = np.nan
    else:
        weighted_ret = float(sum(vals) / sum(weights))

    return {
        "last_date": last_date,
        "last_close": last_close,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "ret_9m": ret_9m,
        "ret_12m": ret_12m,
        "onil_weighted_ret": weighted_ret,
        "avg_vol_50": avg_vol_50,
        "avg_dollar_vol_50": avg_dollar_vol_50,
    }



def rs_scale(series: pd.Series, max_score: int = 99) -> pd.Series:
    s = series.copy()
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.rank(pct=True) * max_score


def build_industry_rs(df: pd.DataFrame) -> pd.DataFrame:
    if "group_key" not in df.columns:
        df["group_key"] = "Unknown"

    grp = df.groupby("group_key", dropna=False)

    ind = grp.agg(
        n_members=("symbol", "count"),
        avg_ret_6m=("ret_6m", "mean"),
        avg_weighted_ret=("onil_weighted_ret", "mean"),
    ).reset_index()

    ind["group_rs_100"] = ind["avg_weighted_ret"].rank(pct=True) * 100
    ind["group_rs_99"] = ind["avg_weighted_ret"].rank(pct=True) * 99

    ind = ind.sort_values("group_rs_100", ascending=False)
    ind["group_rank"] = range(1, len(ind) + 1)

    def grade_func(score):
        if np.isnan(score):
            return "E"
        if score >= 90:
            return "A"
        if score >= 70:
            return "B"
        if score >= 40:
            return "C"
        if score >= 20:
            return "D"
        return "E"

    ind["group_grade"] = ind["group_rs_100"].apply(grade_func)
    ind = ind.rename(columns={"avg_weighted_ret": "group_rs_6m"})

    return ind


def main():
    today = dt.date.today().strftime("%Y%m%d")
    out_rs = os.path.join(OUT_DIR, f"rs_onil_all_{today}.csv")
    out_ind = os.path.join(OUT_DIR, f"industry_rs_6m_{today}.csv")

    print("=== US IBD RS (O'Neil style) 계산 시작 ===")
    print(f"[INFO] 유니버스 파일: {UNIVERSE_FILE}")

    uni = load_universe(UNIVERSE_FILE)
    tickers = uni["symbol"].tolist()
    print(f"[INFO] 유니버스 티커 수: {len(tickers)}")

    records = []
    total = len(tickers)

    for i in range(0, total, CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        print(f"[CHUNK] {i+1} ~ {min(i+CHUNK_SIZE, total)} 티커 다운로드 중...")

        prices_dict = download_prices_chunk(chunk)

        for t in chunk:
            px = prices_dict.get(t)
            if px is None or px.empty:
                print(f"  [SKIP] {t}: 가격 데이터 없음 또는 형식 오류")
                continue

            metrics = calc_returns_from_prices(px)
            if metrics is None:
                print(f"  [SKIP] {t}: 수익률 계산 불가 (데이터 부족)")
                continue

            rec = {"symbol": t}
            rec.update(metrics)
            g = uni.loc[uni["symbol"] == t, "group_key"]
            rec["group_key"] = g.iloc[0] if not g.empty else "Unknown"
            records.append(rec)

        time.sleep(SLEEP_SEC)

    if not records:
        print("[ERROR] 유효한 RS 계산 결과가 없습니다. 가격/데이터를 확인하세요.")
        return

    df = pd.DataFrame(records)
    print(f"[INFO] RS 계산 완료 티커 수: {len(df)}")

    df["rs_onil_99"] = rs_scale(df["onil_weighted_ret"], max_score=99)
    df["rs_onil"] = df["rs_onil_99"]

    ind_df = build_industry_rs(df)

    df = df.merge(
        ind_df[
            [
                "group_key",
                "n_members",
                "avg_ret_6m",
                "group_rs_6m",
                "group_rs_99",
                "group_rs_100",
                "group_rank",
                "group_grade",
            ]
        ],
        on="group_key",
        how="left",
    )

    cols = [
        "symbol",
        "last_date",
        "last_close",
        "ret_3m",
        "ret_6m",
        "ret_9m",
        "ret_12m",
        "onil_weighted_ret",
        "avg_vol_50",
        "avg_dollar_vol_50",
        "rs_onil",
        "rs_onil_99",
        "group_key",
        "n_members",
        "avg_ret_6m",
        "group_rs_6m",
        "group_rs_99",
        "group_rs_100",
        "group_rank",
        "group_grade",
    ]
    df = df[cols]

    df.to_csv(out_rs, index=False, encoding="utf-8-sig")
    print(f"[INFO] 종목 RS 파일 저장 완료: {out_rs}")

    ind_df.to_csv(out_ind, index=False, encoding="utf-8-sig")
    print(f"[INFO] 산업군 RS 파일 저장 완료: {out_ind}")
    print("=== US IBD RS (O'Neil style) 계산 종료 ===")


if __name__ == "__main__":
    main()
