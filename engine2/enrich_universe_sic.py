import os
import datetime as dt
import pandas as pd
import simfin as sf


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    universe_path = os.path.join(data_dir, "universe.csv")
    if not os.path.exists(universe_path):
        raise FileNotFoundError(f"universe.csv not found: {universe_path} (먼저 fetch_universe_nasdaqtrader.py 실행)")

    # SimFin 설정 (무료키가 있다면 secrets로 넣고, 없으면 'free'로 시도)
    api_key = os.getenv("SIMFIN_API_KEY", "free")
    sf.set_api_key(api_key)

    # 캐시 디렉터리(다운로드 반복 방지)
    cache_dir = os.path.join(root, "engine2", "cache", "simfin")
    os.makedirs(cache_dir, exist_ok=True)
    sf.set_data_dir(cache_dir)

    print("[STEP1] Load SimFin companies (US)...")
    # 회사 테이블: ticker에 대한 sector/industry 정보를 제공(버전에 따라 컬럼명이 다를 수 있음)
    companies = sf.load_companies(market="us")

    # 표준화(컬럼 안전 처리)
    c = companies.copy()
    # SimFin 버전에 따라 'Ticker'/'Company Name'/'Sector'/'Industry' 또는 IndustryId 형태일 수 있음
    # 최대한 유연하게 처리
    colmap = {}
    for cand in ["Ticker", "ticker"]:
        if cand in c.columns:
            colmap[cand] = "symbol"
            break
    for cand in ["Company Name", "company_name", "Name", "name"]:
        if cand in c.columns:
            colmap[cand] = "simfin_company_name"
            break
    for cand in ["Sector", "sector"]:
        if cand in c.columns:
            colmap[cand] = "sector"
            break
    for cand in ["Industry", "industry"]:
        if cand in c.columns:
            colmap[cand] = "industry"
            break

    c = c.rename(columns=colmap)

    need_cols = ["symbol"]
    for x in ["sector", "industry", "simfin_company_name"]:
        if x in c.columns:
            need_cols.append(x)

    c = c[need_cols].copy()
    c["symbol"] = c["symbol"].astype(str).str.strip()

    print(f"[INFO] SimFin companies rows: {len(c):,}")

    u = pd.read_csv(universe_path, dtype=str)
    u["symbol"] = u["symbol"].astype(str).str.strip()

    merged = u.merge(c, on="symbol", how="left")

    merged["industry_updated_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    merged.to_csv(universe_path, index=False, encoding="utf-8-sig")

    filled_sector = merged["sector"].notna().sum() if "sector" in merged.columns else 0
    filled_ind = merged["industry"].notna().sum() if "industry" in merged.columns else 0
    print(f"[OK] universe.csv enriched with sector/industry: {universe_path}")
    if "sector" in merged.columns:
        print(f"Sector filled: {filled_sector:,}/{len(merged):,} ({filled_sector/len(merged)*100:.1f}%)")
    if "industry" in merged.columns:
        print(f"Industry filled: {filled_ind:,}/{len(merged):,} ({filled_ind/len(merged)*100:.1f}%)")


if __name__ == "__main__":
    main()
