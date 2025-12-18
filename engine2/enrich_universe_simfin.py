import os
import datetime as dt
import pandas as pd
import simfin as sf


def norm_ticker(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    # 클래스주 표기 통일: BRK.B -> BRK-B
    s = s.replace(".", "-").replace("/", "-")
    return s


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates_lower: list[str]) -> str | None:
    cols = {str(c).strip().lower(): str(c).strip() for c in df.columns}
    for cand in candidates_lower:
        if cand in cols:
            return cols[cand]
    return None


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    universe_path = os.path.join(data_dir, "universe.csv")
    if not os.path.exists(universe_path):
        raise FileNotFoundError(f"universe.csv not found: {universe_path}")

    api_key = os.getenv("SIMFIN_API_KEY", "").strip() or "free"
    sf.set_api_key(api_key)

    cache_dir = os.path.join(root, "engine2", "cache", "simfin")
    os.makedirs(cache_dir, exist_ok=True)
    sf.set_data_dir(cache_dir)

    print("[STEP1] Load universe.csv ...")
    u = pd.read_csv(universe_path, dtype=str)
    if "symbol" not in u.columns:
        raise ValueError(f"universe.csv에 symbol 컬럼이 없습니다. 현재 컬럼: {list(u.columns)}")
    u["symbol"] = u["symbol"].astype(str).str.strip()
    u["symbol_key"] = u["symbol"].map(norm_ticker)

    print("[STEP1] Load SimFin companies (US) ...")
    companies = sf.load_companies(market="us")
    if hasattr(companies, "index") and companies.index.name is not None:
        companies = companies.reset_index()

    c = _normalize_columns(companies)

    # 디버그: sector/industry 관련 컬럼이 뭔지 먼저 확인
    related_cols = [col for col in c.columns if ("sector" in col.lower()) or ("industry" in col.lower())]
    print(f"[DEBUG] companies total cols={len(c.columns)}")
    print(f"[DEBUG] sector/industry related cols={related_cols}")

    ticker_col = _find_col(c, ["ticker", "symbol"])
    if ticker_col is None:
        raise ValueError(f"SimFin companies에서 ticker/symbol 컬럼을 못 찾음. cols={list(c.columns)}")

    name_col = _find_col(c, ["company name", "name", "company"])

    # 이름 컬럼 후보
    sector_name_col = _find_col(c, ["sector", "sector name"])
    industry_name_col = _find_col(c, ["industry", "industry name"])

    # ID 컬럼 후보(중요!)
    sector_id_col = _find_col(c, ["sectorid", "sector_id", "sector id"])
    industry_id_col = _find_col(c, ["industryid", "industry_id", "industry id"])

    print(f"[DEBUG] ticker_col={ticker_col}, sector_name_col={sector_name_col}, industry_name_col={industry_name_col}, "
          f"sector_id_col={sector_id_col}, industry_id_col={industry_id_col}")

    out = pd.DataFrame()
    out["symbol_key"] = c[ticker_col].astype(str).str.strip().map(norm_ticker)

    if name_col:
        out["simfin_company_name"] = c[name_col].astype(str).str.strip()
    else:
        out["simfin_company_name"] = pd.NA

    # 이름이 있으면 채움, 없으면 NA로 두되 ID라도 확보
    out["sector"] = c[sector_name_col].astype(str).str.strip() if sector_name_col else pd.NA
    out["industry"] = c[industry_name_col].astype(str).str.strip() if industry_name_col else pd.NA

    out["sector_id"] = c[sector_id_col] if sector_id_col else pd.NA
    out["industry_id"] = c[industry_id_col] if industry_id_col else pd.NA

    # 그룹키는 우선 industry_id → 없으면 sector_id → 그래도 없으면 NA
    out["group_key"] = out["industry_id"].fillna(out["sector_id"])
    # group_key가 숫자/문자 섞여도 OK. 문자열로 통일
    out["group_key"] = out["group_key"].astype("string")

    out = out.dropna(subset=["symbol_key"]).drop_duplicates(subset=["symbol_key"])

    # 채움률 디버그
    id_sector_fill = out["sector_id"].notna().sum()
    id_ind_fill = out["industry_id"].notna().sum()
    group_fill = out["group_key"].notna().sum()
    name_sector_fill = out["sector"].notna().sum()
    name_ind_fill = out["industry"].notna().sum()

    print(f"[INFO] SimFin tickers(key) rows: {len(out):,}")
    print(f"[INFO] sector_name filled: {name_sector_fill:,}, industry_name filled: {name_ind_fill:,}")
    print(f"[INFO] sector_id filled: {id_sector_fill:,}, industry_id filled: {id_ind_fill:,}, group_key filled: {group_fill:,}")

    print("[STEP1] Merge into universe by symbol_key ...")
    merged = u.merge(
        out[["symbol_key", "group_key", "sector", "industry", "sector_id", "industry_id", "simfin_company_name"]],
        on="symbol_key",
        how="left"
    )

    merged["industry_updated_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    merged.to_csv(universe_path, index=False, encoding="utf-8-sig")

    filled_group = merged["group_key"].notna().sum()
    filled_sector = merged["sector"].notna().sum()
    filled_ind = merged["industry"].notna().sum()
    filled_sector_id = merged["sector_id"].notna().sum()
    filled_ind_id = merged["industry_id"].notna().sum()

    print(f"[OK] universe.csv enriched: {universe_path}")
    print(f"group_key filled: {filled_group:,}/{len(merged):,} ({filled_group/len(merged)*100:.1f}%)")
    print(f"sector name filled: {filled_sector:,}/{len(merged):,} ({filled_sector/len(merged)*100:.1f}%)")
    print(f"industry name filled: {filled_ind:,}/{len(merged):,} ({filled_ind/len(merged)*100:.1f}%)")
    print(f"sector_id filled: {filled_sector_id:,}/{len(merged):,} ({filled_sector_id/len(merged)*100:.1f}%)")
    print(f"industry_id filled: {filled_ind_id:,}/{len(merged):,} ({filled_ind_id/len(merged)*100:.1f}%)")


if __name__ == "__main__":
    main()
