from pathlib import Path
from typing import Union, Dict
import argparse
import json
import pandas as pd
import pickle

def build_token2idx_from_info(info_pkl: str) -> Dict[str, int]:
    """
    info_pkl: nuscenes_infos_train/val/test.pkl 중 하나
    returns: {sample_token: sample_idx}
    """
    # pickle 파일을 직접 로드
    with open(info_pkl, 'rb') as f:
        infos = pickle.load(f)
    # MMDetection3D ≥ 1.1 에서는 dict 형식, 그 이하에선 list 형식일 수 있음
    data_list = infos.get('data_list', infos) if isinstance(infos, dict) else infos
    return {info['token']: info['sample_idx'] for info in data_list}

def parse_nuscenes_results(json_path: Union[str, Path],
                           score_threshold: float = 0.0) -> pd.DataFrame:
    """
    NuScenes submission-style JSON을 읽어 pandas DataFrame으로 변환한다.

    Parameters
    ----------
    json_path : str or Path
        results_nusc.json 경로
    score_threshold : float, optional
        detection_score가 이 값보다 작은 항목은 버린다.

    Returns
    -------
    df : pandas.DataFrame
        행 하나가 bounding box 하나에 대응되는 테이블.
    """
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    meta = data.get("meta", {})          # {'use_camera': False, 'use_lidar': True} 등
    det_dict = data["results"]           # 샘플 토큰 → 디텍션 리스트

    rows: list[dict] = []
    for sample_token, det_list in det_dict.items():
        for det in det_list:
            if det["detection_score"] < score_threshold:
                continue

            rows.append({
                "sample_token": sample_token,
                "detection_name": det["detection_name"],
                "detection_score": det["detection_score"],
                # 위치 (ENU 좌표계)
                "tx": det["translation"][0],
                "ty": det["translation"][1],
                "tz": det["translation"][2],
                # 크기 (dx, dy, dz)
                "sx": det["size"][0],
                "sy": det["size"][1],
                "sz": det["size"][2],
                # 회전 (quaternion w, x, y, z)
                "qw": det["rotation"][0],
                "qx": det["rotation"][1],
                "qy": det["rotation"][2],
                "qz": det["rotation"][3],
                # 속도 (vx, vy)
                "vx": det["velocity"][0],
                "vy": det["velocity"][1],
                # 속성 (있을 수도, 없을 수도)
                "attribute_name": det.get("attribute_name", "")
            })

    df = pd.DataFrame(rows)

    # DataFrame에 meta 정보도 보존하고 싶다면 속성(attribute)으로 달아두기
    df.attrs["meta"] = meta
    return df

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NuScenes detection JSON to pandas DataFrame")
    parser.add_argument(
        "--json",
        type=str,
        help="Path to results_nusc.json (NuScenes submission format)",
    )
    parser.add_argument(
        "--pkl",
        type=str,
        help="Path to nuscenes_infos.pkl",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum detection_score to keep (default: 0.0)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to print after parsing (default: 5)",
    )
    parser.add_argument("-o", "--output",
                        type=str,
                        default=None,
                        help="Where to save df (use .csv or .pkl extension)")

    args = parser.parse_args()

    
    df = parse_nuscenes_results(
        args.json,
        score_threshold=args.score_threshold,
    )
    
    token2idx = build_token2idx_from_info(args.pkl)
    df["frame_idx"] = df["sample_token"].map(token2idx)
    df = df.sort_values(by="frame_idx")
    df.insert(0, "frame_idx", df.pop("frame_idx"))

    print(df.head(args.head))
    print("\nmeta info:", df.attrs["meta"])

    if args.output:
        out_path = Path(args.output)
        suffix = out_path.suffix.lower()
        if suffix == ".csv":
            df.to_csv(out_path, index=False)
        elif suffix in {".pkl", ".pickle"}:
            df.to_pickle(out_path)
        else:
            raise ValueError(f"Unsupported output format '{suffix}'. Use .csv or .pkl")
        print(f"Saved DataFrame to {out_path}")


if __name__ == "__main__":
    main()