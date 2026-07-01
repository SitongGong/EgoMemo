#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中管理评测所需的路径。开源用户只需设置一个环境变量 EGOSERVE_DATA
(指向存放 GT 标注与模型预测的根目录),或直接修改本文件的 DATA_ROOT。

目录约定(推荐布局):
  $EGOSERVE_DATA/
    EgoServe_release/                 # 从 HuggingFace 下载的发布标注
      EgoLife/{A1_JAKE,A4_LUCIA,A5_KATRINA}/{instant,short_term,long_term,episodic}.json
      HoloAssist/holoassist_service_annotations.json
      CaptainCook4D/captaincook4d_service_annotations.json
    predictions/                      # 各模型的预测输出
      egolife_results/                # EgoMemo/消融 的 EgoLife 预测
      holoassist_rebuttal/            # EgoMemo/消融 的 HoloAssist 预测
      captioncook4d_rebuttal/         # EgoMemo/消融 的 CaptainCook4D 预测
      baselines/                      # GPT / Qwen baseline 的预测
        GPT_egolife/  GPT_holoassist/  GPT_captioncook4d/
        Qwen_egolife/ Qwen_holoassist/ Qwen_captioncook4d/
    outputs/                          # 评测结果 result_*.json / 表格
"""
import os

# 数据根目录:优先环境变量,否则用仓库同级的 ./data
DATA_ROOT = os.environ.get(
    "EGOSERVE_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
)

# ---- GT(发布标注,从 HuggingFace 下载)----
RELEASE_ROOT   = os.path.join(DATA_ROOT, "EgoServe_release")
EGOLIFE_GT     = os.path.join(RELEASE_ROOT, "EgoLife")
HOLOASSIST_GT  = os.path.join(RELEASE_ROOT, "HoloAssist", "holoassist_service_annotations.json")
CC4D_GT        = os.path.join(RELEASE_ROOT, "CaptainCook4D", "captaincook4d_service_annotations.json")

# ---- 主模型 / 消融 预测 ----
PRED_ROOT      = os.path.join(DATA_ROOT, "predictions")
EGOLIFE_PRED   = os.path.join(PRED_ROOT, "egolife_results")
HOLOASSIST_PRED= os.path.join(PRED_ROOT, "holoassist_rebuttal")
CC4D_PRED      = os.path.join(PRED_ROOT, "captioncook4d_rebuttal")

# ---- baseline 预测(GPT / Qwen)----
BASELINE_ROOT  = os.path.join(PRED_ROOT, "baselines")
GPT_EGO   = os.path.join(BASELINE_ROOT, "GPT_egolife")
GPT_HOLO  = os.path.join(BASELINE_ROOT, "GPT_holoassist")
GPT_CC4D  = os.path.join(BASELINE_ROOT, "GPT_captioncook4d")
QWEN_EGO  = os.path.join(BASELINE_ROOT, "Qwen_egolife")
QWEN_HOLO = os.path.join(BASELINE_ROOT, "Qwen_holoassist")
QWEN_CC4D = os.path.join(BASELINE_ROOT, "Qwen_captioncook4d")

# ---- 输出 ----
OUTPUT_ROOT    = os.path.join(DATA_ROOT, "outputs")

# ---- caption model 名(EgoLife 预测子目录用)----
CAPTION_MODEL  = os.environ.get("EGOSERVE_CAPTION_MODEL", "qwenvl_3_8b_instruct")

# ---- LLM 评判 API key(从环境变量读,勿硬编码)----
def openai_api_key():
    return os.environ.get("OPENAI_API_KEY", "")

def deepseek_api_key():
    return os.environ.get("DEEPSEEK_API_KEY", "")
