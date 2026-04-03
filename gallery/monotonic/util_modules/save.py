import os
import pandas as pd

def save_unique_row(res: dict, csv_path: str):
    new_df = pd.DataFrame([res])

    if not os.path.exists(csv_path):
        new_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return

    old_df = pd.read_csv(csv_path)

    # 컬럼 합집합
    cols = list(old_df.columns.union(new_df.columns))
    old_df = old_df.reindex(columns=cols)
    new_df = new_df.reindex(columns=cols)

    # 중복 체크 (NaN 안전)
    sentinel = "__NA__"
    dup = (old_df.fillna(sentinel)
                 .eq(new_df.fillna(sentinel).iloc[0])
                 .all(axis=1)
                 .any())

    if dup:
        return

    pd.concat([old_df, new_df], ignore_index=True) \
      .to_csv(csv_path, index=False, encoding="utf-8-sig")



def is_config_already_run(config: dict, csv_path: str) -> bool:
    """
    csv에 config와 완전히 동일한 row가 이미 있으면 True 반환
    """
    if not os.path.exists(csv_path):
        return False  # csv 없으면 아직 아무 실험도 안 함

    df = pd.read_csv(csv_path)

    # config의 모든 key가 csv 컬럼에 있는지 체크
    missing_cols = set(config.keys()) - set(df.columns)
    if missing_cols:
        # 컬럼이 없다는 건 이전 실험 포맷이 다르다는 뜻 → 안전하게 다시 돌림
        return False

    # breakpoint()
    mask = pd.Series(True, index=df.index)
    for k, v in config.items():
        if pd.isna(v):
            mask &= pd.isna(df[k])
        else:
            mask &= (df[k] == v)

    return mask.any()
