import json
import logging
import os
from typing import Callable, List, Optional

import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm

from utils.image_utils import load_img_to_array

logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)


def extract_subtitles(
    frame_paths: List[str],
    config: dict,
    fps: float,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
):
    """
    从视频帧中提取字幕。

    此函数通过OCR技术识别视频帧中的字幕内容，处理并生成SRT格式的字幕文件。

    参数:
    - frame_paths: 视频帧的文件路径列表。
    - config: 配置字典，包含OCR和字幕提取的配置信息。
    - fps: 视频的帧率，用于时间计算。

    返回:
    - subtitles: 字幕文本。
    - center: 字幕文本的中心位置。
    """
    file_name = os.path.split(frame_paths[0])[0]

    # [THAY DOI - RESUME] Neu lan truoc da quet OCR xong (co san file ket qua) thi DUNG LAI,
    # khoi chay lai PaddleOCR (buoc nay rat ton thoi gian).
    #   - _ocr_check.json: ket qua OCR da loc
    #   - _ocr_center.json: vi tri tam (center) cua phu de, can cho buoc chen phu de
    check_path = f"{file_name}_ocr_check.json"
    center_path = f"{file_name}_ocr_center.json"
    # [THAY DOI - RESUME] Chi can co _ocr_check.json la dung lai duoc (khoi OCR).
    # Neu thieu _ocr_center.json (vd ket qua tao boi ban code cu) thi TINH LAI center
    # tu chinh _ocr_check.json, khong can quet OCR lai.
    if os.path.exists(check_path):
        ocr_result = load_ocr_result(check_path)
        if os.path.exists(center_path):
            with open(center_path, "r", encoding="utf-8") as f:
                center = json.load(f).get("center", 0)
        else:
            center = compute_center_from_result(
                ocr_result, config["video"]["groups_tolerance"]
            )
            with open(center_path, "w", encoding="utf-8") as f:
                json.dump({"center": float(center)}, f)
        return ocr_result, center

    # ===== (Giu nguyen) Quet OCR binh thuong neu chua co cache =====
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang=config["ocr"]["lang"],
        det_model_dir=config["ocr"]["det_model_dir"],
        rec_model_dir=config["ocr"]["rec_model_dir"],
    )
    ocr_result = get_ocr_result(ocr, frame_paths, config, progress_cb=progress_cb)
    save_ocr_result(ocr_result, f"{file_name}_ocr.json")

    ocr_result, center = check_ocr_result(ocr_result, config, fps, frame_paths[0])
    save_ocr_result(ocr_result, check_path)

    # [THAY DOI - RESUME] Luu them center de lan sau chay lai khoi tinh lai.
    # float(center) vi center co the la kieu numpy, json khong ghi truc tiep duoc.
    with open(center_path, "w", encoding="utf-8") as f:
        json.dump({"center": float(center)}, f)

    return ocr_result, center


def save_ocr_result(ocr_result: dict, ocr_path: str):
    """
    保存OCR识别结果到指定的JSON文件中

    参数:
    - ocr_result: OCR识别的结果，格式为字典
    - ocr_path: 保存OCR结果的文件路径

    返回:
    - 无
    """
    # [THAY DOI] them encoding=utf-8: ket qua OCR co the la tieng Trung (ensure_ascii=False)
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_result, f, ensure_ascii=False, indent=4)


# [THAY DOI - RESUME] Ham doc lai ket qua OCR da luu (de chay tiep, khoi quet lai).
def load_ocr_result(ocr_path: str) -> dict:
    """
    Doc lai ket qua OCR da luu tu file JSON.

    Tham so:
    - ocr_path: duong dan file JSON da luu truoc do.

    Tra ve:
    - dict ket qua OCR (cung dinh dang voi luc save_ocr_result).
    """
    with open(ocr_path, "r", encoding="utf-8") as f:
        return json.load(f)


# [THAY DOI - RESUME] Tinh lai vi tri tam (center) cua phu de tu ket qua OCR da luu.
# Dung khi da co _ocr_check.json nhung thieu _ocr_center.json (vd file cu tao boi
# ban code truoc day) -> khoi phai quet OCR lai chi de lay center.
def compute_center_from_result(ocr_result: dict, tolerance: int = 20) -> float:
    """
    Tinh center (toa do y trung binh cua dong phu de) tu ket qua OCR da loc.

    Tham so:
    - ocr_result: dict ket qua OCR (moi gia tri co truong "box" = [xmin, ymin, xmax, ymax]).
    - tolerance: do dung sai khi gom nhom (lay tu config["video"]["groups_tolerance"]).

    Tra ve:
    - float: vi tri y trung binh cua phu de.
    """
    center_list = []
    for value in ocr_result.values():
        try:
            _, ymin, _, ymax = value["box"]
            center_list.append((ymin + ymax) / 2)
        except (KeyError, TypeError, ValueError):
            continue
    return float(get_groups_mean(center_list, tolerance))


def sort_ocr_result(ocr_result: List[List]):
    """
    对OCR识别结果进行排序，以确定文本的垂直位置。
    """
    result = []
    for line in ocr_result:
        coords, _ = line
        x, y = 0, 0
        min_y = 10000
        max_y = 0

        for coord in coords:
            x += coord[0]
            y += coord[1]
            min_y = min(min_y, coord[1])
            max_y = max(max_y, coord[1])

        center_x = x / len(coords)
        center_y = y / len(coords)
        height = max_y - min_y

        result.append([center_x, center_y, height])

    y_groups = {}
    for i, (center_x, center_y, height) in enumerate(result):
        found_group = False
        for group_y in y_groups:
            if abs(center_y - group_y) <= height / 3:
                y_groups[group_y].append((i, center_x))
                found_group = True
                break
        if not found_group:
            y_groups[center_y] = [(i, center_x)]

    sorted_ocr_result = []
    for group_y in sorted(y_groups.keys()):
        sorted_group = sorted(y_groups[group_y], key=lambda x: x[1])
        sorted_ocr_result.extend([ocr_result[i] for i, _ in sorted_group])

    return sorted_ocr_result


def get_ocr_result(
    ocr: PaddleOCR,
    frame_paths: List[str],
    config: dict,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
):
    """
    对一系列图像帧进行OCR识别，提取并整理文本信息及其在图像中的位置。

    参数:
    ocr: PaddleOCR对象，用于执行OCR识别。
    frame_paths: 图像帧的文件路径列表。
    config: 配置字典，包含OCR和字幕提取的配置信息。

    返回:
    包含每帧中识别到的文本及其位置信息的字典。
    """
    img_array = load_img_to_array(frame_paths[0])
    min_height = int(img_array.shape[0] * config["ocr"]["min_height_ratio"])
    max_height = int(img_array.shape[0] * config["ocr"]["max_height_ratio"])

    ocr_result = {}
    total = len(frame_paths)
    step = max(1, total // 40) if total else 1
    for i, frame_path in enumerate(tqdm(frame_paths, desc="OCR")):
        if progress_cb and (i % step == 0 or i == total - 1):
            progress_cb(i + 1, total, "OCR")
        img_array = load_img_to_array(frame_path)
        results = ocr.ocr(img_array[min_height:max_height, :, :], cls=False, det=True, rec=True)
        result = results[0]
        if result is None:
            continue
        result = sort_ocr_result(result)
        for idx, line in enumerate(result):
            coords, texts = line
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]
            x4, y4 = coords[3]

            xmin = int(max(x1, x4))
            xmax = int(min(x2, x3))
            ymin = int(max(y1, y2)) + min_height
            ymax = int(min(y3, y4)) + min_height

            text = texts[0]
            ocr_result[frame_path + f",{idx}"] = {
                "box": [xmin, ymin, xmax, ymax],
                "text": text,
            }
    return ocr_result


def check_ocr_result(ocr_result: dict, config: dict, fps: float, frame_path: str):
    """
    根据配置参数和视频帧率，校验并整合OCR识别结果。

    参数:
    ocr_result: dict - OCR识别结果，键为帧路径，值为包含文字信息和 bounding box 的字典。
    config: dict - 配置参数，用于设定宽度、高度的偏差及分组容忍度。
    fps: float - 视频的帧率，用于计算最小持续时间的帧数。
    frame_path: str - 图像帧的路径，用于读取图像数组。

    返回:
    new_ocr_result: dict - 校验和整合后的OCR结果。
    center: float - 识别到的字幕文本的中心位置。
    """
    img_array = load_img_to_array(frame_path)
    x_center_frame = img_array.shape[1] / 2
    x_delta = img_array.shape[1] * config["video"]["width_delta"]
    y_delta = img_array.shape[0] * config["video"]["height_delta"]

    center_list = []
    word_height_list = []
    for key, value in tqdm(ocr_result.items(), desc="Word info"):
        xmin, ymin, xmax, ymax = value["box"]
        x_center = (xmin + xmax) / 2
        if x_center - x_delta < x_center_frame < x_center + x_delta:
            y_center = (ymin + ymax) / 2
            center_list.append(y_center)
            word_height_list.append(ymax - ymin)
    tolerance = config["video"]["groups_tolerance"]
    center = get_groups_mean(center_list, tolerance)
    word_height = get_groups_mean(word_height_list, tolerance)

    new_ocr_result = {}
    for key, value in tqdm(ocr_result.items(), desc="Word concat"):
        xmin, ymin, xmax, ymax = value["box"]
        y_center = (ymin + ymax) / 2
        x_center = (xmin + xmax) / 2
        if (
            center - y_delta < y_center < center + y_delta
            and word_height - tolerance <= ymax - ymin <= word_height + tolerance
        ):
            frame_path = key.split(",")[0]
            if frame_path not in new_ocr_result:
                new_ocr_result[frame_path] = value
            else:
                xmin_, ymin_, xmax_, ymax_ = new_ocr_result[frame_path]["box"]
                if (
                    (xmin - xmax_ <= x_delta / 2 or xmin_ - xmax <= x_delta / 2)
                    and -tolerance / 2 <= ymin_ - ymin <= tolerance / 2
                    and -tolerance / 2 <= ymax_ - ymax <= tolerance / 2
                ) or (x_center_frame - x_delta <= x_center <= x_center_frame + x_delta):
                    new_ocr_result[frame_path]["box"] = [
                        min(xmin, xmin_),
                        min(ymin, ymin_),
                        max(xmax, xmax_),
                        max(ymax, ymax_),
                    ]
                    new_ocr_result[frame_path]["text"] += value["text"]

    ocr_result = new_ocr_result.copy()
    new_ocr_result = {}
    frame_number_pre = 0
    text_pre = ""
    frames = fps * config["video"]["min_duration"]
    for frame_path, value in tqdm(ocr_result.items(), desc="OCR check"):
        xmin, ymin, xmax, ymax = value["box"]
        y_center = (ymin + ymax) / 2
        x_center = (xmin + xmax) / 2
        if (
            center - y_delta < y_center < center + y_delta
            and x_center_frame - x_delta <= x_center <= x_center_frame + x_delta
        ):
            frame_number = int(os.path.splitext(os.path.basename(frame_path))[0])
            text = value["text"]
            if text == text_pre and frame_number - frame_number_pre <= frames:
                for i in range(frame_number_pre + 1, frame_number):
                    frame_path_ = os.path.join(
                        os.path.split(frame_path)[0],
                        f"{i:04d}{os.path.splitext(frame_path)[-1]}",
                    )
                    new_ocr_result[frame_path_] = value
            new_ocr_result[frame_path] = value
            frame_number_pre = frame_number
            text_pre = text
    return new_ocr_result, center


def get_groups_mean(arr: list, tolerance=20):
    """
    计算分组后的平均值。

    对给定数组进行分组，每组内的元素与组内最小元素的差值不大于tolerance。
    然后计算最大组的平均值作为结果。

    参数:
    arr: list, 输入的整数列表。
    tolerance: int, 分组的差值容忍度，默认为20。

    返回:
    float, 最大组的平均值。
    """
    if not arr:
        return 0

    arr.sort()
    groups = []
    current_group = [arr[0]]

    for i in range(1, len(arr)):
        if abs(arr[i] - current_group[0]) <= tolerance:
            current_group.append(arr[i])
        else:
            groups.append(current_group)
            current_group = [arr[i]]

    groups.append(current_group)

    max_group = max(groups, key=len)

    return np.mean(max_group)
