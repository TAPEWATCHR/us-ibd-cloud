import os
import datetime as dt
import requests
import pandas as pd


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


# 제외할 “비주식성/파생성” 증권 키워드 (회사명에 포함되는 경우가 많음)
# ✅ 보통주/우선주는 남기고,
# ❌ 워런트/유닛/라이트/노트(채권성)/ETN 성격은 제거
EXCLUDE_NAME_KEYWORDS = [
    "warrant", "warrants",
    "unit", "units",
    "right", "rights",
    "note", "notes",
    "debenture", "bond",
    "etn",
    "subordinated",
]


def _download_text(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def _read_pipe_table(text: str) -> pd.DataFrame:
    # 마지막 줄에 "File Creation Time:" 같은 메타 라인이 붙음 → 제거
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and "File Creation Time" in lines[-1]:
        lines = lines[:-1]
    cleaned = "\n".join(lines)

    from io import StringIO
    df = pd.read_csv(StringIO(cleaned), sep="|", dtype=str)
    return df


def _contains_any_keyword(series: pd.Series, keywords: list[str]) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for kw in keywords:
        mask = mask | s.str.contains(kw, regex=False)
    return mask


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
    df["is_etf"] = df["is_etf"].fillna("").astype(str).str.upper().str.strip()
    df["test_issue"] = df["test_issue"].fillna("").astype(str).str.upper().str.strip()

    # 빈 심볼 제거
    df = df[df["symbol"].notna() & (df["symbol"] != "")]

    # 테스트 종목 제거
    df = df[df["test_issue"] != "Y"]

    # ✅ ETF 제거 (오닐식 성장주 목적에 부적합)
    df = df[df["is_etf"] != "Y"]

    # ✅ 워런트/유닛/라이트/노트 등 비주식성 제거 (회사명 기준)
    bad_name = _contains_any_keyword(df["company_name"], EXCLUDE_NAME_KEYWORDS)
    df = df[~bad_name]

    # 심볼에 공백 포함 제거
    df = df[~df["symbol"].str.contains(r"\s", regex=True)]

    # 중복 제거
    df = df.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)
    df["updated_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    out_path = os.path.join(data_dir, "universe.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] universe.csv saved: {out_path}")
    print(f"Total symbols (common+preferred, no ETF/warrant/unit/right/notes): {len(df):,}")


if __name__ == "__main__":
    main()
