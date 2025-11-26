# src/config/loader.py
# -*- coding: utf-8 -*-
"""
Config loader for speed estimation.
速度估计相关配置加载工具。
"""

import os
import yaml
from ultralytics import YOLO
from typing import Set



def load_speed_config(cfg_path: str) -> dict:
    """
    Load YAML config file.
    加载 YAML 配置文件。

    Args:
        cfg_path (str): Path to config YAML.
                        配置文件路径。
    """
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_dynamic_ids(cfg: dict, model_weights: str) -> Set[int]:
    """
    Resolve dynamic (vehicle) class ids from config + YOLO model.
    根据配置 + YOLO 模型自动解析“动态类”(车辆等)的 id。

    - 如果 cfg['dynamic_classes']['ids'] 非空，优先使用这些 id；
    - 否则根据 names 匹配 model.names 中的类别名。

    Args:
        cfg (dict): Loaded config dict.
                    已加载的配置字典。
        model_weights (str): Path to YOLO weights (.pt).
                             YOLO 权重路径。

    Returns:
        Set of class ids treated as dynamic vehicles.
        返回被认为是“动态车辆”的类别 id 集合。
    """
    dyn_cfg = cfg.get("dynamic_classes", {}) or {}
    id_list = dyn_cfg.get("ids") or []
    name_list = dyn_cfg.get("names") or []

    if id_list:
        return set(int(i) for i in id_list)

    if not name_list:
        return set()

    model = YOLO(model_weights)
    name_set = set(name_list)
    dynamic_ids: set[int] = set()

    for cid, cname in model.names.items():
        if cname in name_set:
            dynamic_ids.add(int(cid))

    print(f"[INFO] Dynamic class ids resolved from names: {dynamic_ids}")
    return dynamic_ids
