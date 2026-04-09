import json
import random


def generate_empty_clicks(sample, num=1, seed=42):
    
    null_clicks = []
    width = sample["image_width"]
    height = sample["image_height"]
    correct_bbox = sample["correct_box"]["bbox"]
    dark_bbox = sample["dark_box"]["bbox"]

    while len(null_clicks) <= num:
        click_x = random.randint(0,width)
        click_y = random.randint(0,height)
        if (correct_bbox[0] <= click_x <= correct_bbox[2] and 
            correct_bbox[1] <= click_y <= correct_bbox[3]):
            continue
        elif (dark_bbox[0] <= click_x <= dark_bbox[2] and 
            dark_bbox[1] <= click_y <= dark_bbox[3]):
            continue
        else:
            null_clicks.append((click_x,click_y))
            
    return [{"coordinates": null_click, "label": -1} for null_click in null_clicks]


def generate_clicks(sample):
    """
    为一个样本生成良性/欺骗点击坐标和标签。
    - 良性：correct_box 中心点。
    - 欺骗：dark_box 中心点（确保不在 correct_box 内）。
    """
    correct_bbox = sample["correct_box"]["bbox"]
    dark_bbox = sample["dark_box"]["bbox"]
    
    # 计算 correct_box 中心（良性点击）
    correct_x = (correct_bbox[0] + correct_bbox[2]) / 2
    correct_y = (correct_bbox[1] + correct_bbox[3]) / 2
    benign_click = (round(correct_x, 2), round(correct_y, 2))
    
    # 计算 dark_box 中心（欺骗点击）
    dark_x = (dark_bbox[0] + dark_bbox[2]) / 2
    dark_y = (dark_bbox[1] + dark_bbox[3]) / 2
    
    # 检查是否在 correct_box 内，如果是则调整到 dark_box 右下角
    if (correct_bbox[0] <= dark_x <= correct_bbox[2] and 
        correct_bbox[1] <= dark_y <= correct_bbox[3]):
        dark_x = dark_bbox[2] - 1  # 调整到右边缘
        dark_y = dark_bbox[3] - 1  # 调整到下边缘
    
    deceptive_click = (round(dark_x, 2), round(dark_y, 2))
    
    return {
        "benign": {"coordinates": benign_click, "label": 1},
        "deceptive": {"coordinates": deceptive_click, "label": -1}
    }

def generate_clicks_2(sample):
    """
    为一个样本仅生成良性点击坐标和标签。
    - 良性correct_box 中心点
    """

    correct_bbox = sample["correct_box"]["bbox"]
    
    # 计算 correct_box 中心（良性点击）
    correct_x = (correct_bbox[0] + correct_bbox[2]) / 2
    correct_y = (correct_bbox[1] + correct_bbox[3]) / 2
    benign_click = (round(correct_x, 2), round(correct_y, 2))
    
    return {
        "benign": {"coordinates": benign_click, "label": 1}
    }