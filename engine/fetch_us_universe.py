# fetch_us_universe.py

import io
import requests
import pandas as pd
from datetime import datetime


NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _load_symbol_file(url: str) -> pd.DataFrame:
    """
    NASDAQ Trader symbol directory 파일을 읽어서 DataFrame으로 변환.
    마지막 줄 'File Creation Time:' 제거 후 파싱.
    """
    resp = requests.get(url)
    resp.raise_for_status()

    lines = resp.text.splitlines()

    # 맨 마지막 줄은 'File Creation Time: ...' 같이 메타정보라 제거
    if lines and "File Creation Time" in lines[-1]:
        lines = lines[:-1]

    text = "\n".join(lines)
    df = pd.read_csv(io.StringIO(text), sep="|")
    return df


def fetch_nasdaq_list() -> pd.DataFrame:
    """
    NASDAQ 상장 종목 리스트 가져오기
    """
    df = _load_symbol_file(NASDAQ_URL)

    # 주요 컬럼만 사용
    df = df.rename(
        columns={
            "Symbol": "symbol",
            "Security Name": "name",
            "Market Category": "market_category",
            "ETF": "is_etf",
            "Test Issue": "is_test",
            "Financial Status": "financial_status",
        }
    )

    df["exchange"] = "NASDAQ"
    return df[
        [
            "symbol",
            "name",
            "exchange",
            "market_category",
            "is_etf",
            "is_test",
            "financial_status",
        ]
    ]


def fetch_other_list() -> pd.DataFrame:
    """
    NYSE, AMEX 등 기타 거래소 상장 종목 리스트 가져오기
    """
    df = _load_symbol_file(OTHER_URL)

    df = df.rename(
        columns={
            "ACT Symbol": "symbol",
            "Security Name": "name",
            "Exchange": "exchange",
            "ETF": "is_etf",
            "Test Issue": "is_test",
        }
    )

    # 필요 컬럼만 정리
    return df[
        [
            "symbol",
            "name",
            "exchange",
            "is_etf",
            "is_test",
        ]
    ]


def build_us_universe() -> pd.DataFrame:
    """
    미국 상장 종목 전체 유니버스 구성:
    - NASDAQ + OTHER (NYSE, AMEX)
    - ETF, Test Issue, 특수 심볼 일부 제거
    """
    nasdaq = fetch_nasdaq_list()
    other = fetch_other_list()

    df = pd.concat([nasdaq, other], ignore_index=True)

    # 기본 클리닝
    # 1) ETF 제외
    df = df[df["is_etf"] != "Y"]

    # 2) Test Issue 제외
    df = df[df["is_test"] != "Y"]

    # 3) symbol이 비어 있는 행 제거
    df = df.dropna(subset=["symbol"])

    # 4) 심볼 대문자 정리
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # 5) $ 기호가 들어간 특수 심볼 제외 (기업 액션 등)
    mask_special = df["symbol"].str.contains(r"\$", regex=True, na=False)
    df = df[~mask_special]


    # 중복 제거
    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    return df


import datetime
import os

def main():
    """
    미국 상장 종목 유니버스를 만들어서
    1) 날짜 붙은 파일: us_universe_YYYYMMDD.csv
    2) 고정 이름 파일: us_universe.csv
    두 개를 동시에 저장한다.
    """
    today_str = datetime.date.today().strftime("%Y%m%d")

    df = build_us_universe()  # 위쪽에 이미 정의되어 있는 함수라고 가정

    if df is None or df.empty:
        print("[ERROR] 유니버스가 비어 있습니다. 소스 코드를 확인하세요.")
        return

    # 1) 날짜 붙은 파일
    dated_name = f"us_universe_{today_str}.csv"
    df.to_csv(dated_name, index=False, encoding="utf-8-sig")

    # 2) 항상 같은 이름으로 덮어쓰기 (다른 스크립트들이 이걸 사용)
    fixed_name = "us_universe.csv"
    df.to_csv(fixed_name, index=False, encoding="utf-8-sig")

    print(f"미국 상장 종목 유니버스 저장 완료: {dated_name}")
    print(f"[INFO] 고정 파일도 갱신됨: {fixed_name}")


if __name__ == "__main__":
    main()
