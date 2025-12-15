import os
import datetime as dt
import requests
import pandas as pd


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _download_text(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def _read_pipe_table(text: str) -> pd.DataFrame:
    # 마지막 줄에 "File Creation Time:" 같은 메타 라인이 붙음 → 제거
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # 메타 라인 제거(보통 마지막 줄)
    if "File Creation Time" in lines[-1]:
        lines = lines[:-1]
    cleaned = "\n".join(lines)
    from io import StringIO
    df = pd.read_csv(StringIO(cleaned), sep="|", dtype=str)
    return df


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    print("[STEP1] Download NasdaqTrader symbol directories...")

    nasdaq_txt = _download_text(NASDAQ_LISTED_URL)
    other_txt  = _download_text(OTHER_LISTED_URL)

    nd = _read_pipe_table(nasdaq_txt)
    ot = _read_pipe_table(other_txt)

    # 표준 컬럼 맞추기
    # nasdaqlisted: Symbol, Security Name, Market Category, Test Issue, ETF, Round Lot Size, Financial Status, ...
    # otherlisted: ACT Symbol, Security Name, Exchange, CQS Symbol, ETF, Round Lot Size, Test Issue, NASDAQ Symbol, ...
    nd_out = pd.DataFrame({
        "symbol": nd.get("Symbol"),
        "company_name": nd.get("Security Name"),
        "exchange": "NASDAQ",
        "is_etf": nd.get("ETF"),
        "test_issue": nd.get("Test Issue"),
        "source": "nasdaqlisted",
    })

    ot_out = pd.DataFrame({
        "symbol": ot.get("ACT Symbol"),
        "company_name": ot.get("Security Name"),
        "exchange": ot.get("Exchange"),
        "is_etf": ot.get("ETF"),
        "test_issue": ot.get("Test Issue"),
        "source": "otherlisted",
    })

    df = pd.concat([nd_out, ot_out], ignore_index=True)

    # 정리
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["company_name"] = df["company_name"].astype(str).str.strip()
    df["exchange"] = df["exchange"].astype(str).str.strip()

    # 빈 심볼 제거
    df = df[df["symbol"].notna() & (df["symbol"] != "")]

    # Test Issue 제거(Y)
    df["test_issue"] = df["test_issue"].fillna("").str.upper().str.strip()
    df = df[df["test_issue"] != "Y"]

    # ETF 여부 플래그 정리(Y/N)
    df["is_etf"] = df["is_etf"].fillna("").str.upper().str.strip()

    # 흔한 잡심볼 정리(원하면 규칙 완화 가능)
    # 예: BRK.B 같은 점(.) 포함 종목도 실제 상장이라서, 여기서는 일단 "그대로 유지"
    # 대신 공백/이상문자만 제거
    df = df[~df["symbol"].str.contains(r"\s", regex=True)]

    df = df.drop_duplicates(subset=["symbol"]).sort_values("symbol")
    df["updated_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    out_path = os.path.join(data_dir, "universe.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] universe.csv saved: {out_path}")
    print(f"Total symbols: {len(df):,}")


if __name__ == "__main__":
    main()
