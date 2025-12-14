import os
import glob
import pandas as pd
import numpy as np

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
SMR_FILE = os.path.join(ENGINE_DIR, "smr_factors.csv")


def find_latest_base_rs() -> str:
    """
    rs_onil_all_YYYYMMDD.csv 중에서 가장 최신 파일을 찾는다.
    (SMR 붙기 전 '원본 RS' 파일)
    """
    pattern = os.path.join(ENGINE_DIR, "rs_onil_all_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("rs_onil_all_*.csv (원본 RS 파일)을 찾지 못했습니다.")
    files.sort(reverse=True)
    return files[0]


def load_smr_factors(path: str) -> pd.DataFrame:
    """
    smr_factors.csv 를 읽어온다.
    없거나 비어 있으면 예외 대신 None을 반환하도록 호출부에서 처리.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SMR 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)

    # 티커 컬럼 통일
    symbol_col = None
    for c in ["symbol", "ticker", "Symbol", "Ticker"]:
        if c in df.columns:
            symbol_col = c
            break

    if symbol_col is None:
        raise ValueError(
            f"SMR 파일에서 티커 컬럼(symbol/ticker)을 찾지 못했습니다. 현재 컬럼: {list(df.columns)}"
        )

    if symbol_col != "symbol":
        df = df.rename(columns={symbol_col: "symbol"})

    df["symbol"] = df["symbol"].astype(str).str.strip()

    return df


def main():
    print("=== RS + SMR 병합 시작 (enrich_smr.py) ===")

    base_rs_path = find_latest_base_rs()
    print(f"[INFO] 원본 RS 파일 선택: {base_rs_path}")

    rs_df = pd.read_csv(base_rs_path)

    # RS 파일에 symbol 컬럼 강제 통일
    symbol_col = None
    for c in ["symbol", "ticker", "Symbol", "Ticker"]:
        if c in rs_df.columns:
            symbol_col = c
            break

    if symbol_col is None:
        raise ValueError(
            f"RS 원본 파일에 티커 컬럼이 없습니다. 현재 컬럼: {list(rs_df.columns)}"
        )

    if symbol_col != "symbol":
        print(f"[INFO] RS 파일의 티커 컬럼 '{symbol_col}' 를 'symbol'로 이름 변경합니다.")
        rs_df = rs_df.rename(columns={symbol_col: "symbol"})

    rs_df["symbol"] = rs_df["symbol"].astype(str).str.strip()

    # SMR 요인 로드
    smr_df = None
    try:
        smr_df = load_smr_factors(SMR_FILE)
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
    except Exception as e:
        print(f"[WARN] SMR 로드 중 예외 발생: {e}")

    if smr_df is None or smr_df.empty:
        print("[WARN] smr_factors.csv 가 없거나 비어 있습니다. SMR 점수/등급은 이번 턴에서는 계산되지 않습니다.")
        # RS만 포함된 파일로 저장 (컬럼은 그대로 유지)
        out_rs_only = base_rs_path.replace(".csv", "_smr.csv")
        rs_df.to_csv(out_rs_only, index=False, encoding="utf-8-sig")
        print(f"[INFO] SMR 없이 RS만 포함된 파일 저장 완료: {out_rs_only}")
        print("=== RS + SMR 병합 종료 (SMR 없음, 경고만) ===")
        return

    # SMR 컬럼 이름 정리 (없으면 NaN 으로 채우기 위해 기본 이름만 맞춰둠)
    for col in ["sales_growth", "profit_margin", "roe", "smr_score", "smr_grade"]:
        if col not in smr_df.columns:
            smr_df[col] = np.nan

    # symbol 기준으로 LEFT JOIN
    merged = rs_df.merge(
        smr_df[
            ["symbol", "sales_growth", "profit_margin", "roe", "smr_score", "smr_grade"]
        ],
        on="symbol",
        how="left",
        suffixes=("", "_smrdup"),
    )

    # 혹시라도 suffix 붙은 중복 컬럼이 생기면 정리
    dup_cols = [c for c in merged.columns if c.endswith("_smrdup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    out_path = base_rs_path.replace(".csv", "_smr.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] RS + SMR 파일 저장 완료: {out_path}")
    print("=== RS + SMR 병합 종료 ===")


if __name__ == "__main__":
    main()
