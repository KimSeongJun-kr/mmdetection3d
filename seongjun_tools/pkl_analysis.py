import pickle
import argparse
from pprint import pprint
import pandas as pd
import json
import numpy as np
import jsonpickle
import pathlib

def to_jsonable(x):
    """넘파이·비표준 타입을 파이썬 기본 타입으로 변환."""
    # ── NumPy → 파이썬 ────────────────────────────────
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    # ── 컨테이너 재귀 처리 ────────────────────────────
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]          # tuple→list
    # ── 기타(예: bytes, set 등) 필요 시 추가 변환 ───
    return x  # str/int/float/bool/None 는 그대로

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NuScenes detection JSON to pandas DataFrame")
    parser.add_argument(
        "--pkl",
        type=str,
        help="Path to nuscenes_infos.pkl",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to print after parsing (default: 5)",
    )

    args = parser.parse_args()

    
    with open(args.pkl, 'rb') as f:
        pkl_data = pickle.load(f)

    # pprint(pkl_data, width=80)

    print("Loaded pkl_data is a ", type(pkl_data))
    if isinstance(pkl_data, dict):
        print("Available keys in pkl_data:", list(pkl_data.keys()))

    json_path = pathlib.Path(args.pkl).with_suffix(".json")
    json_path.write_text(jsonpickle.encode(pkl_data, indent=2), encoding="utf-8")

    # jsonable = to_jsonable(pkl_data)                # 🔑 변환!
    # with open(args.pkl.replace('.pkl', '.json'), "w", encoding="utf-8") as f:
    #     json.dump(jsonable, f, ensure_ascii=False, indent=2)   # 끝!


    # if isinstance(pkl_data, dict) and all(isinstance(v, list) for v in pkl_data.values()):
    #     data_list = []
    #     for category, infos in pkl_data.items():
    #         for info in infos:
    #             # category 컬럼 추가 (optional)
    #             info['category'] = category
    #             data_list.append(info)
    # else:
    #     raise KeyError(f"예상과 다른 구조입니다. keys: {list(pkl_data.keys())}")

    # # 데이터 프레임 생성
    # df = pd.DataFrame(data_list)

    # df['box3d_lidar'] = df['box3d_lidar'].apply(lambda arr: arr.tolist())

    # # 데이터 프레임 저장
    # df.to_csv(args.pkl.replace('.pkl', '.csv'), index=False)


if __name__ == "__main__":
    main()
    
