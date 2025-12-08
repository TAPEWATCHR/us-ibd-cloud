# fix_latest_smr.py
#
# latest_rs_smr.csv 안에 생긴 smr_score_x / smr_score_y, smr_grade_x / smr_grade_y,
# s_raw_x / s_raw_y 등 "_x", "_y" 중복 컬럼을
# 하나의 smr_score, smr_grade, s_raw, m_raw, r_raw ... 등으로 정리해준다.

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "latest_rs_smr.csv")


def merge_pair(df: pd.DataFrame, base: str) -> pd.DataFrame:
    """
    base_x, base_y 컬럼이 있다면 통합해서 base에 저장하고,
    *_x, *_y 컬럼은 삭제한다.
    """
    x = f"{base}_x"
    y = f"{base}_y"

    if x not in df.columns and y not in df.columns:
        # 둘 다 없으면 건드릴 필요 없음
        return df

    # 둘 중 하나라도 있으면 Series 준비
    if x in df.columns and y in df.columns:
        merged = df[x].combine_first(df[y])
    elif x in df.columns:
        merged = df[x]
    else:
        merged = df[y]

    df[base] = merged

    # *_x, *_y 칼럼 제거
    for col in (x, y):
        if col in df.columns:
            del df[col]

    print(f"[INFO] 병합 완료: {base}")
    return df


def main():
    print(f"[INFO] 대상 파일: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"latest_rs_smr.csv를 찾을 수 없습니다: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    # SMR 관련 주요 컬럼들 병합
    targets = [
        "smr_score",
        "smr_grade",
        "s_raw",
        "m_raw",
        "r_raw",
        "s_pct",
        "m_pct",
        "r_pct",
    ]

    for base in targets:
        df = merge_pair(df, base)

    # 정리된 결과를 같은 파일로 덮어쓰기
    df.to_csv(CSV_PATH, index=False)
    print("[DONE] latest_rs_smr.csv 중복 SMR 컬럼 정리 완료")


if __name__ == "__main__":
    main()
