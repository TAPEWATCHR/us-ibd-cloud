"""
enrich_smr.py

- 최신 RS 원본 파일(rs_onil_all_YYYYMMDD.csv)을 찾는다.
- 티커 컬럼 이름이 무엇이든(symbol, ticker, TICKER, code 등) 자동으로 인식해서 'symbol'로 통일한다.
- smr_factors.csv가 있으면, 거기 있는 매출성장/이익률/ROE로 SMR 점수/등급을 계산해서 병합.
- smr_factors.csv가 없으면, 경고만 찍고
  RS 원본에 SMR 관련 컬럼(전부 None)을 추가한 뒤 저장하고 정상 종료한다.
"""

import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_latest_base_rs() -> str:
    """
    rs_onil_all_*.csv 중에서 '_smr'가 붙지 않은 가장 최신 파일을 찾는다.
    예: rs_onil_all_20251208.csv
    """
    pattern = os.path.join(BASE_DIR, "rs_onil_all_*.csv")
    candidates = []

    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        # 이미 SMR가 붙은 파일은 제외 (예: rs_onil_all_20251208_smr.csv)
        if "_smr" in fname:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError("rs_onil_all_*.csv (원본 RS 파일)을 찾지 못했습니다.")

    candidates.sort()
    latest = candidates[-1]
    print(f"[INFO] 원본 RS 파일 선택: {os.path.basename(latest)}")
    return latest


def load_smr_factors() -> pd.DataFrame | None:
    """
    engine/smr_factors.csv 를 읽어온다.
    파일이 없으면 None을 반환하고, SMR은 빈 값으로만 부여한다.
    """
    smr_path = os.path.join(BASE_DIR, "smr_factors.csv")

    if not os.path.exists(smr_path):
        print(f"[WARN] smr_factors.csv 파일이 없습니다. SMR 점수/등급은 이번 턴에서는 계산되지 않습니다.")
        return None

    df = pd.read_csv(smr_path)

    # 최소한 symbol 컬럼이 있어야 함
    if "symbol" not in df.columns:
        # 혹시 다른 이름으로 되어 있다면 여기서도 한번 더 처리할 수 있지만,
        # 기본 버전에서는 symbol 컬럼을 요구하도록 둔다.
        raise ValueError("smr_factors.csv 에 'symbol' 컬럼이 없습니다.")

    print(f"[INFO] smr_factors.csv 로딩 완료. 행 수: {len(df)}")
    return df


def compute_smr_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    smr_factors 에서 sales_growth, profit_margin, roe 를 0~100 점수로 표준화해서
    단순 평균으로 SMR 점수를 만들고, 등급(A~E)까지 부여한다.
    """
    work = df.copy()

    # 필요한 컬럼이 없으면 그냥 NaN으로 채운다.
    for col in ["sales_growth", "profit_margin", "roe"]:
        if col not in work.columns:
            work[col] = np.nan

    # 각 지표별로 순위(0~100) 계산
    def rank_to_0_100(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=series.index)

        # 높은 값이 좋은 것: rank(ascending=False)
        ranks = valid.rank(method="average", ascending=False)
        pct = (1 - (ranks - 1) / (len(valid) - 1)) * 100  # 0~100
        out = pd.Series(np.nan, index=series.index)
        out.loc[valid.index] = pct
        return out

    work["score_sales"] = rank_to_0_100(work["sales_growth"])
    work["score_margin"] = rank_to_0_100(work["profit_margin"])
    work["score_roe"] = rank_to_0_100(work["roe"])

    # 3개 점수 평균
    work["smr_score"] = work[["score_sales", "score_margin", "score_roe"]].mean(axis=1)

    # 등급 A~E 부여
    def to_grade(score: float) -> str | None:
        if pd.isna(score):
            return None
        if score >= 80:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 40:
            return "C"
        elif score >= 20:
            return "D"
        else:
            return "E"

    work["smr_grade"] = work["smr_score"].apply(to_grade)

    return work


def normalize_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    RS 원본 파일에서 'symbol' 컬럼이 없을 경우,
    ticker / TICKER / code / Code / Symbol 같은 후보 컬럼을 찾아서 'symbol'로 통일한다.
    하나도 찾지 못하면 ValueError.
    """
    if "symbol" in df.columns:
        return df

    # 후보 컬럼 이름들 (소문자 기준)
    candidates_lower = ["symbol", "ticker", "code", "종목코드", "secid"]

    col_map = {c.lower(): c for c in df.columns}

    found_col = None
    for c_low in candidates_lower:
        if c_low in col_map:
            found_col = col_map[c_low]
            break

    if found_col is None:
        raise ValueError(
            f"RS 원본 파일에서 티커 컬럼(symbol/ticker/code 등)을 찾지 못했습니다. "
            f"현재 컬럼들: {list(df.columns)}"
        )

    if found_col != "symbol":
        print(f"[INFO] RS 파일의 티커 컬럼 '{found_col}' 를 'symbol'로 이름 변경합니다.")
        df = df.rename(columns={found_col: "symbol"})

    return df


def main():
    print("=== RS + SMR 병합 시작 (enrich_smr.py) ===")

    # 1) 최신 RS 원본 파일 찾기
    base_rs_path = find_latest_base_rs()
    rs_df = pd.read_csv(base_rs_path)

    # 1-1) 티커 컬럼 이름을 자동 인식해서 'symbol'로 통일
    rs_df = normalize_symbol_column(rs_df)

    # 2) SMR 팩터 로딩 시도
    smr_factors = load_smr_factors()

    if smr_factors is None:
        # SMR 팩터 파일이 없다면, RS만 유지하고 SMR 관련 컬럼은 전부 None으로 추가
        print("[WARN] smr_factors.csv 미존재: SMR 점수/등급은 비워놓고 파일을 저장합니다.")
        enriched = rs_df.copy()
        for col in [
            "sales_growth",
            "profit_margin",
            "roe",
            "smr_score",
            "smr_grade",
        ]:
            if col not in enriched.columns:
                enriched[col] = np.nan

        out_path = base_rs_path.replace(".csv", "_smr.csv")
        enriched.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] SMR 없이 RS만 포함된 파일 저장 완료: {os.path.basename(out_path)}")
        print("=== RS + SMR 병합 종료 (SMR 없음, 경고만) ===")
        return

    # 3) SMR 점수/등급 계산
    smr_scored = compute_smr_score(smr_factors)

    # symbol 기준으로 머지
    merged = pd.merge(
        rs_df,
        smr_scored[["symbol", "sales_growth", "profit_margin", "roe", "smr_score", "smr_grade"]],
        on="symbol",
        how="left",
        suffixes=("", "_smrdup"),
    )

    # 혹시라도 중복 칼럼이 생겼으면 정리
    dup_cols = [c for c in merged.columns if c.endswith("_smrdup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    # 4) 출력 파일 이름: rs_onil_all_YYYYMMDD_smr.csv
    out_path = base_rs_path.replace(".csv", "_smr.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] RS+SMR 병합 파일 저장 완료: {os.path.basename(out_path)}")
    print("=== RS + SMR 병합 종료 ===")


if __name__ == "__main__":
    main()
