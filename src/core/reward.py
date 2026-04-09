import re
import os
import json
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from .parser import parse_agent_output, extract_xml
from math import sqrt

try:
    import fcntl
except Exception:
    fcntl = None

def _extract_text_from_completions(completions):
    texts = []
    for completion in completions:
        if isinstance(completion, list):
            if len(completion) > 0 and isinstance(completion[0], dict) and "content" in completion[0]:
                texts.append(str(completion[0]["content"]))
            else:
                texts.append(str(completion[0]) if len(completion) > 0 else "")
        else:
            texts.append(str(completion))
    return texts

def _broadcast_to_len(x, n: int):
    """
    TRL 里 completions 的长度通常是 batch_size * num_generations，
    但标注字段（correct_box/dark_box）可能只有 batch_size。
    这里做一个简单广播：如果 n % len(x) == 0，就重复每条标注。
    """
    if x is None:
        return [None] * n
    if isinstance(x, list) and len(x) == n:
        return x
    if isinstance(x, list) and len(x) > 0 and (n % len(x) == 0):
        k = n // len(x)
        out = []
        for v in x:
            out.extend([v] * k)
        return out
    return [x] * n

def _safe_key_cmp(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a[0]) - float(b[0])) < 1e-3 and abs(float(a[1]) - float(b[1])) < 1e-3
    except Exception:
        return False

def update_status_in_snapshot(snapshot_path, id_val, click_val, incoming_status):
    if not snapshot_path or not os.path.exists(snapshot_path):
        return False
    # require both id and click to be present for unambiguous matching
    if id_val is None or click_val is None:
        return False
    
    try:
        with open(snapshot_path, 'r', encoding='utf-8') as fr:
            lines = [l for l in fr.readlines() if l.strip()]

        entries = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                entries.append(obj)
            except Exception:
                continue

        updated = False
        for idx, obj in enumerate(entries):
            if not isinstance(obj, dict):
                continue
            # require both id and click to match
            if obj.get('id') == id_val and 'click' in obj and _safe_key_cmp(obj.get('click'), click_val):
                old_status = obj.get('status', None)
                if old_status is True:
                    return True
                if incoming_status is True:
                    obj['status'] = True
                    updated = True
                elif old_status is None:
                    obj['status'] = False
                    updated = True
                entries[idx] = obj
                break

        if not updated:
            return False

        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(snapshot_path))
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as fw:
                for obj in entries:
                    fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
            
            if fcntl is not None:
                with open(snapshot_path, 'r') as flockf:
                    fcntl.flock(flockf, fcntl.LOCK_EX)
                    os.replace(tmp_path, snapshot_path)
                    fcntl.flock(flockf, fcntl.LOCK_UN)
            else:
                os.replace(tmp_path, snapshot_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        return True
    except Exception as e:
        print(f"Error in update_status_in_snapshot: {e}")
        return False

def hybrid_label_confidence_reward(
    completions,
    gen_type: List[int], 
    image_height,
    image_width,
    correct_box,
    dark_box,
    click,
    recorded_samples_path,
    snapshot_path,
    **kwargs,
):
    """
    基于目前论文逻辑的 Hybrid Reward（沿用原始代码的 completion 解析方式）：

    - completion 输出格式沿用： "<judge>...</judge>, <conf>...</conf>"
      其中 judge 现在允许 -1/0/1：
        1=benign, 0=useless(null), -1=deceptive

    Reward:
      if L == hatL:
         R = Conf
      else:
         R = p_c * w_s(L,hatL) * (-Conf)

    Severity w_s:
      C4 (fatal):  L=-1, hatL=1       -> 10
      C2/C3:       L=0,  hatL in {1,-1} -> 1 + p_a
      C1 (default mismatch): otherwise -> 1

    Attention scalar p_a（可选）:
      S0 = S_img - S_b - S_d
      p_a = S_hat / S_img,  S_hat = {S_b if hatL=1; S0 if hatL=0; S_d if hatL=-1}
      若不提供面积，p_a=0。

    Confidence adjustment scalar p_c（可选）:
      p_c = clip( (1/(d_cta_hat+eps)) * (S_L/S_img), p_min, p_max )
      S_L ∈ {S_b,S0,S_d} 按 L 取值
      若不提供距离/面积，p_c=1。
    """

    # ---- 超参 ----
    p_min: float = 0.1
    p_max: float = 10.0
    eps: float = 1e-6

    completion_texts = _extract_text_from_completions(completions)
    n = len(completion_texts)
    print(f"🔍[RWD][ENTER] n={n} gen_type={gen_type} snapshot_path={snapshot_path} ids_kw={kwargs.get('id')}")
    
    # 广播types到n长度（如果需要）
    types_list = _broadcast_to_len(gen_type, n)
    
    # 准备 snapshot 更新所需的列表
    ids = kwargs.get("id")
    ids_list = _broadcast_to_len(ids, n)
    clicks_list = _broadcast_to_len(click, n)

    def _normalize_click_item(c):
        if c is None: return None
        if isinstance(c, (list, tuple)) and len(c) > 0 and isinstance(c[0], (list, tuple)):
            a = c[0]
            return [float(a[0]), float(a[1])]
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            return [float(c[0]), float(c[1])]
        if isinstance(c, dict):
            if 'x' in c and 'y' in c: return [float(c['x']), float(c['y'])]
            if 0 in c and 1 in c: return [float(c[0]), float(c[1])]
        return None

    rewards = []
    recorded_samples = []  # 记录不匹配的样本

    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _area_S0(s_img: float, s_b: float, s_d: float) -> float:
        return float(s_img) - float(s_b) - float(s_d)

    def _S_for_label(L: int, s_img: float, s_b: float, s_d: float) -> float:
        s0 = _area_S0(s_img, s_b, s_d)
        if L == 1:
            return float(s_b)
        if L == -1:
            return float(s_d)
        return float(s0)  # L == 0

    def _attention_scalar(hatL: int, s_img: float, s_b: float, s_d: float) -> float:
        if s_img is None or s_b is None or s_d is None:
            return 0.0
        s_img = float(s_img)
        if s_img <= 0:
            return 0.0
        s0 = _area_S0(s_img, s_b, s_d)
        if hatL == 1:
            s_hat = float(s_b)
        elif hatL == -1:
            s_hat = float(s_d)
        else:
            s_hat = float(s0)
        return _clip(s_hat / s_img, 0.0, 1.0)

    def _severity_weight(L: int, hatL: int, p_a: float) -> float:
        # 仅 mismatch 时调用
        if L == -1 and hatL == 1:
            return 10.0  # C4
        if L == 0 and hatL in (1, -1):
            return 1.0 + float(p_a)  # C2/C3
        return 1.0  # C1

    def _confidence_adjustment(L: int, hatL: int, s_img, s_b, s_d, dL, dH) -> float:
        if (s_img is None) or (s_b is None) or (s_d is None) or (dL is None) or (dH is None):
            return 1.0
        s_img = float(s_img)
        if s_img <= 0:
            return 1.0
        S_L_val = sqrt(_S_for_label(L, s_img, s_b, s_d))
        raw = (1/ (float(dH) + eps)) / (S_L_val / s_img)
        return _clip(raw, p_min, p_max)
    
    def _point_to_box_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """计算点 (px, py) 到矩形框 (x1,y1,x2,y2) 边缘的最小欧氏距离"""
        # 将点坐标约束到矩形内部（如果在内部则距离为0）
        closest_x = max(x1, min(px, x2))
        closest_y = max(y1, min(py, y2))
        # 返回欧氏距离
        return ((px - closest_x)**2 + (py - closest_y)**2)**0.5
    
    gt_x1,gt_y1,gt_x2,gt_y2 = correct_box[0]["bbox"]
    dc_x1,dc_y1,dc_x2,dc_y2 = dark_box[0]["bbox"]
    x = click[0][0]
    y = click[0][1]

    s_img = image_height[0]*image_width[0]
    s_b = (gt_x2-gt_x1)*(gt_y2-gt_y1)
    s_d = (dc_x2-dc_x1)*(dc_y2-dc_y1)
    dL = 0

    
    for i, (text, L) in enumerate(zip(completion_texts, types_list)):
        # 1) 解析 judge 和 conf
        judge_match = re.search(
            r"<judge>\s*(-?1|0)\s*</judge>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        conf_match = re.search(
            r"<conf>\s*([0-9]*\.?[0-9]+)\s*</conf>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        
        if not judge_match or not conf_match:
            # 解析失败，负奖励=-10.0，记录（不打印）
            reward = -10.0
            recorded_samples.append({"text": text, "reason": "parse_fail"})
            rewards.append(float(reward))
            print(f"🚩 Hybrid Reward: {reward:.3f}, Recorded: {len(recorded_samples)} samples")
            continue

        hatL = int(judge_match.group(1))
        Conf = float(conf_match.group(1))
        Conf = max(0.0, min(1.0, Conf))

        # 2) 保护：标签域必须是 {-1,0,1}
        if (L not in (-1, 0, 1)) or (hatL not in (-1, 0, 1)):
            reward = -1.0
            recorded_samples.append({"text": text, "reason": "label_out_of_domain", "L": L, "hatL": hatL, "conf": Conf})
            rewards.append(float(reward))
            print(f"🚩 Hybrid Reward: {reward:.3f}, Recorded: {len(recorded_samples)} samples")
            continue

        # 3) reward 按当前论文逻辑计算
        if hatL == L:
            reward = Conf
        else:
            if hatL==1:
                dH = _point_to_box_distance(x,y,gt_x1,gt_y1,gt_x2,gt_y2)
            elif hatL==-1:
                dH = _point_to_box_distance(x,y,dc_x1,dc_y1,dc_x2,dc_y2)
            else:
                dH = min(_point_to_box_distance(x,y,gt_x1,gt_y1,gt_x2,gt_y2),
                         _point_to_box_distance(x,y,dc_x1,dc_y1,dc_x2,dc_y2))
            p_a = _attention_scalar(hatL, s_img, s_b, s_d)  # 缺面积则=0
            w_s = _severity_weight(L, hatL, p_a)            # severity
            p_c = _confidence_adjustment(L, hatL, s_img, s_b, s_d, dL, dH)  # 缺信号则=1

            reward = p_c * w_s * (-Conf)

            recorded_samples.append({
                "text": text,
                "reason": "label_mismatch",
                "L": L,
                "hatL": hatL,
                "conf": Conf,
                "p_c": p_c,
                "w_s": w_s,
                "p_a": p_a,
            })

        rewards.append(float(reward))
        print(f"🚩 Hybrid Reward: {reward:.3f}, Recorded: {len(recorded_samples)} samples")

        # --- 更新 snapshot 中对应条目的 status（只更新 status 字段） ---
        try:
            if snapshot_path:
                id_val = ids_list[i] if isinstance(ids_list, list) and len(ids_list) > i else None
                click_val = _normalize_click_item(clicks_list[i]) if isinstance(clicks_list, list) and len(clicks_list) > i else None
                incoming_status = True if hatL == L else False
                update_status_in_snapshot(snapshot_path, id_val, click_val, incoming_status)
        except Exception as e:
            print(f"Error updating snapshot: {e}")
            pass

    if recorded_samples:
        print("😭 Recorded mismatched samples:")
        for sample in recorded_samples:
            try:
                print(
                    f"  Reason: {sample['reason']}, L: {sample.get('L')}, hatL: {sample.get('hatL')}, "
                    f"Conf: {sample.get('conf')}, p_c: {sample.get('p_c')}, w_s: {sample.get('w_s')}, p_a: {sample.get('p_a')}, "
                    f"Text: {sample['text'][:100]}..."
                )
            except Exception:
                print(f"  Reason: {sample.get('reason')}, Text: {sample.get('text','')[:100]}...")

        # 如果提供了路径，将 recorded_samples 以 JSONL 追加写入文件，带时间戳和进程 id
        # 由于训练进程比较少，所以这样做是work的；如果训练进程较多的话，应该考虑其他的预防措施。
        if recorded_samples_path:
            try:
                import json, time, os

                ts = time.time()
                pid = os.getpid()
                # 尝试获取分布式 rank（如果可用）
                _rank = None
                try:
                    import torch
                    if hasattr(torch, 'distributed') and torch.distributed.is_available() and torch.distributed.is_initialized():
                        _rank = int(torch.distributed.get_rank())
                except Exception:
                    _rank = None

                run_ts = kwargs.get("run_ts") if isinstance(kwargs, dict) else None

                dirpath = os.path.dirname(recorded_samples_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                with open(recorded_samples_path, "a", encoding="utf-8") as f:
                    for sample in recorded_samples:
                        sample_meta = dict(sample)
                        sample_meta["_ts_write"] = ts
                        sample_meta["_pid"] = pid
                        sample_meta["_rank"] = _rank
                        sample_meta["_run_ts"] = run_ts
                        f.write(json.dumps(sample_meta, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Warning: failed to write recorded_samples to {recorded_samples_path}: {e}")

    return rewards

def label_confidence_reward(
    completions,
    type: List[int],  # 从数据集的 "type" 字段：1表示benign（期望judge=1），0表示deceptive（期望judge=0）
    **kwargs,
):
    """
    基于evaluator输出是否正常（judge=1/0）和置信度的奖励函数。
    
    - 输出格式：如 "Right: <judge>1</judge>, <conf>0.95</conf>" 或 "Suspicious: <judge>0</judge>, <conf>0.45</conf>"
    - 正常性：judge=1表示正常点击，judge=0表示无关点击，judge=-1表示欺骗点击。
    - 匹配：根据types判断期望judge（1=benign期望judge=1，0=deceptive期望judge=0）。
    - 奖励：计划修改一下功能，这里不再处理奖励，只反应推理结果，具体的问题由inference推理函数去处理。
    """
    
    completion_texts = _extract_text_from_completions(completions)
    n = len(completion_texts)
    
    # 广播types到n长度（如果需要）
    types_list = _broadcast_to_len(type, n)
    
    rewards = []
    recorded_samples = []  # 记录不匹配的样本
    
    for text, expected_judge in zip(completion_texts, types_list):
        # 1) 解析 judge 和 conf
        judge_match = re.search(
            r"<judge>\s*(-?1|0)\s*</judge>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        conf_match = re.search(
            r"<conf>\s*([0-9]*\.?[0-9]+)\s*</conf>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        
        if not judge_match or not conf_match:
            # 解析失败，负奖励=-1，记录
            reward = -1.0
            recorded_samples.append({"text": text, "reason": "parse_fail"})
        else:
            judge = int(judge_match.group(1))
            conf = float(conf_match.group(1))
            conf = max(0.0, min(1.0, conf))  # 确保在0-1
            
            if judge == expected_judge:
                reward = conf  # 正奖励
            else:
                reward = -conf  # 负奖励，记录
                recorded_samples.append({"text": text, "reason": "judge_mismatch", "expected": expected_judge, "actual": judge, "conf": conf})
        
        rewards.append(float(reward))
        print(f"Confidence Reward: {reward:.3f}, Recorded: {len(recorded_samples)} samples")
    
    # 输出记录的样本
    if recorded_samples:
        print("Recorded mismatched samples:")
        for sample in recorded_samples:
            try:
                print(f"  Reason: {sample['reason']}, Expected: {sample['expected']}, Actual: {sample['actual']}, Conf: {sample['conf']}, Text: {sample['text'][:100]}...")
            except:
                print(f"  Reason: {sample['reason']}, Text: {sample['text'][:100]}...")
    
    return rewards
