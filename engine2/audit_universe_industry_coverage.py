import os
import re
import pandas as pd
import simfin as sf


def norm_ticker(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = s.replace(".", "-").replace("/", "-")
    return s


def classify_symbol(sym: str, name: str) -> str:
    s = (sym or "").upper()
    n = (name or "").upper()

    # 심볼 패턴 기반
    if "-" in s:
        return "has_hyphen(class/series?)"
    if s.endswith("P") or "PREFERRED" in n or "PREF" in n:
        return "preferred_like"
    if "ADR" in n:
        return "ADR"
    if "LP" in n or "L.P" in n or "PARTNERS" in n:
        return "LP/Partnership"
    if "TRUST" in n or "REIT" in n:
        return "Trust/REIT"
    if "HOLDINGS" in n:
        return "Holdings"
    return "other"


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    universe_path = os.path.join(data_dir, "universe.csv")
    if not os.path.exists(universe_path):
        raise FileNotFoundError(f"universe.csv not found: {universe_path}")

    # SimFin 설정
    api_key = os.getenv("SIMFIN_API_KEY", "").strip() or "free"
    sf.set_api_key(api_key)
    cache_dir = os.path.join(root, "engine2", "cache", "simfin")
    os.makedirs(cache_dir, exist_ok=True)
    sf.set_data_dir(cache_dir)

    print("[AUDIT] Load universe.csv ...")
    u = pd.read_csv(universe_path, dtype=str)
    u["symbol"] = u["symbol"].astype(str).str.strip()
    u["symbol_key"] = u["symbol"].map(norm_ticker)
    u["company_name"] = u.get("company_name", "").astype(str)
    u["exchange"] = u.get("exchange", "").astype(str)

    print("[AUDIT] Load SimFin companies (US) ...")
    c = sf.load_companies(market="us")
    if getattr(c, "index", None) is not None and c.index.name is not None:
        c = c.reset_index()
    c.columns = [str(x).strip() for x in c.columns]

    # SimFin 컬럼 찾기(현재 로그상 IndustryId만 존재)
    ticker_col = "Ticker" if "Ticker" in c.columns else ("Symbol" if "Symbol" in c.columns else None)
    if ticker_col is None:
        raise ValueError(f"SimFin companies에서 ticker 컬럼을 못 찾음. cols={list(c.columns)}")
    ind_col = "IndustryId" if "IndustryId" in c.columns else None
    if ind_col is None:
        print("[WARN] SimFin companies에 IndustryId가 없습니다. (현재 cols에 IndustryId가 없으면 커버리지 분석 불가)")
        c["IndustryId"] = pd.NA
        ind_col = "IndustryId"

    sim = pd.DataFrame({
        "symbol_key": c[ticker_col].astype(str).str.strip().map(norm_ticker),
        "simfin_industry_id": c[ind_col],
    }).dropna(subset=["symbol_key"]).drop_duplicates(subset=["symbol_key"])

    sim["in_simfin"] = True
    sim["simfin_has_industryid"] = sim["simfin_industry_id"].notna()

    merged = u.merge(sim, on="symbol_key", how="left")
    merged["in_simfin"] = merged["in_simfin"].fillna(False)
    merged["simfin_has_industryid"] = merged["simfin_has_industryid"].fillna(False)

    def reason(row):
        if not row["in_simfin"]:
            return "NOT_IN_SIMFIN"
        if row["in_simfin"] and not row["simfin_has_industryid"]:
            return "IN_SIMFIN_BUT_NO_INDUSTRYID"
        # 여기까지 왔으면 SimFin엔 있고 IndustryId도 있는데,
        # universe에 group_key가 비어있다면(현재 enrich 후 파일 기준) 매핑로직 문제 가능
        if pd.isna(row.get("group_key")) or str(row.get("group_key")) == "" or row.get("group_key") == "nan":
            return "IN_SIMFIN_HAS_INDUSTRYID_BUT_GROUPKEY_MISSING"
        return "OK"

    merged["reason"] = merged.apply(reason, axis=1)
    merged["pattern_tag"] = merged.apply(lambda r: classify_symbol(r["symbol"], r["company_name"]), axis=1)

    # 요약 출력
    total = len(merged)
    ok = (merged["reason"] == "OK").sum()
    not_in = (merged["reason"] == "NOT_IN_SIMFIN").sum()
    no_ind = (merged["reason"] == "IN_SIMFIN_BUT_NO_INDUSTRYID").sum()
    weird = (merged["reason"] == "IN_SIMFIN_HAS_INDUSTRYID_BUT_GROUPKEY_MISSING").sum()

    print(f"[AUDIT] total={total:,} OK={ok:,} NOT_IN_SIMFIN={not_in:,} NO_INDUSTRYID={no_ind:,} GROUPKEY_MISSING={weird:,}")

    # 패턴별 분포
    dist = merged[merged["reason"] != "OK"].groupby(["reason", "pattern_tag"]).size().reset_index(name="count")
    dist = dist.sort_values(["reason", "count"], ascending=[True, False])

    out_all = os.path.join(data_dir, "universe_audit_all.csv")
    out_bad = os.path.join(data_dir, "universe_audit_unknown_only.csv")
    out_dist = os.path.join(data_dir, "universe_audit_unknown_distribution.csv")

    merged.to_csv(out_all, index=False, encoding="utf-8-sig")
    merged[merged["reason"] != "OK"].to_csv(out_bad, index=False, encoding="utf-8-sig")
    dist.to_csv(out_dist, index=False, encoding="utf-8-sig")

    print(f"[AUDIT] saved: {out_all}")
    print(f"[AUDIT] saved: {out_bad}")
    print(f"[AUDIT] saved: {out_dist}")


if __name__ == "__main__":
    main()
