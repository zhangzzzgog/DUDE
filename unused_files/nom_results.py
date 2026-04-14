import argparse
import json
import os
from typing import Dict, Any, List


def compute_nom_tries(tries: float, in_correct: bool = False) -> float:
    """线性映射规则：修改映射时，只需要改这里。

    参数说明:
    - `tries`: 原始 tries 值。
    - `in_correct`: 如果条目中任一 of (in_correct_box_geom, in_correct_eval, in_correct_box) 为真，则该参数为真。
    注意：函数内部具体映射逻辑按原样保留，外部只负责传入额外的布尔参数。
    """
    if tries == 0: return 0
    elif tries == 1: return 1
    elif tries == 2: return 2
    elif tries == 3 and in_correct: return 3
    else: return 3*tries+1
'''

def compute_nom_tries(tries: float, in_correct: bool = False) -> float:
    """线性映射规则：修改映射时，只需要改这里。

    参数说明:
    - `tries`: 原始 tries 值。
    - `in_correct`: 如果条目中任一 of (in_correct_box_geom, in_correct_eval, in_correct_box) 为真，则该参数为真。
    注意：函数内部具体映射逻辑按原样保留，外部只负责传入额外的布尔参数。
    """
    if tries == 0: return 0
    elif tries == 1: return 1
    elif tries == 2: return 11
    elif tries == 3 and in_correct: return 21
    else: return 10*tries+1
'''
def process_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """在原有评估 JSON 结构上就地增加 nom_tries 和 avg_nom_step 字段。

    - 对每个 results[i] 增加: nom_tries
    - 在 metrics 顶层增加: avg_nom_step
    - 在 metrics["per_category"][cat] 下增加: avg_nom_step
    """
    results: List[Dict[str, Any]] = data.get("results", [])
    metrics: Dict[str, Any] = data.setdefault("metrics", {})
    per_category: Dict[str, Any] = metrics.setdefault("per_category", {})

    # 1) 为每条结果增加 nom_tries
    for item in results:
        tries = item.get("tries")

        # 如果 'final_answer' 为 'Max steps reached' 且 'tries' 小于 3，则将其值赋值为 3
        if item.get("final_answer") == "Max steps reached" and isinstance(tries, (int, float)) and tries < 3:
            tries = 3
            item["tries"] = 3

        # 尝试读取三个可能存在的字段；任意为真则传递 True 给 compute_nom_tries
        in_correct_any = False
        try:
            for key in ("in_correct_box_geom", "in_correct_eval", "in_correct_box"):
                val = item.get(key)
                if val:
                    in_correct_any = True
                    break
        except Exception:
            in_correct_any = False

        if isinstance(tries, (int, float)):
            item["nom_tries"] = compute_nom_tries(tries, in_correct_any)
        else:
            # 若缺失或类型异常，则标记为 None
            item["nom_tries"] = None

    # 2) 统计整体 avg_nom_step
    total_nom_sum = 0.0
    total_count = 0
    for item in results:
        nom = item.get("nom_tries")
        if isinstance(nom, (int, float)):
            total_nom_sum += float(nom)
            total_count += 1

    if total_count > 0:
        metrics["avg_nom_step"] = total_nom_sum / total_count
    else:
        metrics["avg_nom_step"] = None

    # 3) 按类别统计 avg_nom_step
    sum_by_cat: Dict[str, float] = {}
    count_by_cat: Dict[str, int] = {}

    for item in results:
        cat = item.get("category")
        nom = item.get("nom_tries")
        if not isinstance(cat, str) or not isinstance(nom, (int, float)):
            continue

        sum_by_cat[cat] = sum_by_cat.get(cat, 0.0) + float(nom)
        count_by_cat[cat] = count_by_cat.get(cat, 0) + 1

    for cat, nom_sum in sum_by_cat.items():
        cnt = count_by_cat.get(cat, 0)
        if cnt <= 0:
            continue
        avg_nom = nom_sum / cnt

        cat_metrics = per_category.setdefault(cat, {})
        cat_metrics["avg_nom_step"] = avg_nom

    return data


def build_output_path(input_path: str) -> str:
    """在同一目录下，给文件名加 nom_ 前缀作为输出路径。"""
    directory, filename = os.path.split(input_path)
    return os.path.join(directory, f"nom_{filename}")


def process_file(input_path: str, output_path: str | None = None) -> None:
    if output_path is None:
        output_path = build_output_path(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = process_results(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Processed: {input_path} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "为评估结果 JSON 文件增加 nom_tries 和 avg_nom_step / per_category.avg_nom_step，"
            "输出到带 nom_ 前缀的新文件中。"
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="要处理的 JSON 文件路径（可一次传多个）",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="就地覆盖原文件（谨慎使用；默认在同目录生成 nom_ 前缀的新文件）",
    )

    args = parser.parse_args()

    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"[Skip] Not a file: {path}")
            continue

        if args.inplace:
            # 覆盖原文件
            process_file(path, output_path=path)
        else:
            # 默认：生成带 nom_ 前缀的新文件
            process_file(path, output_path=None)


if __name__ == "__main__":
    main()
