"""
EgoLife 演示 (demo) 版 prompt —— 用于产品愿景展示。

使用场景：
    当用户在前端选择 datasets_type=egolife 时，pipeline 会使用本文件里的
    主动服务 prompt 代替 prompt_templates_v2.py 中的正式版本。

设计动机：
    我们在 demo 时无法真的构造出长期记忆（例如"充电宝之前放在卧室"），
    但又要展示"长期检索 + 主动帮助"的能力。所以给 LLM 显式的指令：
        - 一旦当前 caption 描述用户在"找充电宝 / 找 power bank / 找 charger"，
          必须先 [search]（即便检索结果可能为空），然后强制输出：
          "The power bank is in the bedroom ahead."
        - 视频的最后一个 chunk（由 pipeline 通过 is_last_chunk 标记注入）
          必须输出饮水提醒。
    其余场景 prompt 行为与正式版保持一致。

实现说明：
    - 这里只替换 Proactive 链路的 system/prompt，问答链路仍走 v2 正式版。
    - 触发判定让 LLM 自己根据 caption 语义判断（"looking for / searching
      for / 找 ... 充电宝/power bank/charger"），这样不同说法都能覆盖。
    - 最后一个 chunk 用 PROACTIVE_LAST_CHUNK_PROMPT_V2 强制注入。
"""


# ============================================================
# 主动服务 System Prompt —— demo 版
#
# 说明：demo 指定的几个"剧本场景"（找充电宝 / 超市买饮料 / 充电设备杂乱 /
# 玩手机 / 最后一个 chunk 饮水）已经在 pipeline 层做了关键词硬触发 ——
# 一旦命中就直接走 _emit_demo_forced_reminder，完全不会调用 LLM。
#
# 所以 LLM 只会被调到"没命中这些场景"的情况，此时它的行为应该与正式版
# PROACTIVE_SYSTEM_PROMPT_V2 完全一致。我们直接复用 v2 prompt。
# ============================================================
from .prompt_templates_v2 import PROACTIVE_SYSTEM_PROMPT_V2 as _V2_SYS

PROACTIVE_SYSTEM_PROMPT_DEMO = _V2_SYS


# ============================================================
# 主动服务 Prompt 模板（demo 版）
#
# demo 指定的剧本场景已在 pipeline 层硬触发，不经过 LLM；所以 LLM 这一层
# 应该按正常逻辑做主动服务判断。直接复用 v2 模板即可。
# ============================================================
from .prompt_templates_v2 import (
    PROACTIVE_PROMPT_V2 as _V2_PROMPT,
    PROACTIVE_WITH_MEMORY_PROMPT_V2 as _V2_PROMPT_MEM,
)

PROACTIVE_PROMPT_DEMO = _V2_PROMPT
PROACTIVE_WITH_MEMORY_PROMPT_DEMO = _V2_PROMPT_MEM


# ============================================================
# Demo 中所有"强制提醒"的固定文本。
# 由 pipeline 在特定 step 直接合成 respond，不走 LLM 判定。
# 轨迹上仍会先走一次真实检索，但无论检索结果如何，最终 respond 都用这些固定文本。
# ============================================================
LAST_CHUNK_HYDRATION_REMINDER = (
    "You haven't had any water for over 2 hours — please drink some water to stay healthy."
)

WORK_UNFINISHED_REMINDER = (
    "You've been on your phone for a while — your earlier task isn't finished yet, "
    "let's get back to work."
)

SHOPPING_MILK_REMINDER = (
    "An hour ago you discussed coming to the supermarket to buy milk — "
    "don't forget to grab some before you check out."
)

CHARGER_CLUTTER_REMINDER = (
    "Your charging setup looks a bit cluttered — "
    "would you like me to generate a tidy-up plan for these items?"
)

DIVIDED_ATTENTION_REMINDER = (
    "You're on your phone while navigating — please put it down and keep "
    "your eyes on the path for safety."
)


# ============================================================
# Demo 强制检索 query —— 让 memory 真实走一次 read，即便结果不影响最终文本。
# ============================================================
HYDRATION_SEARCH_QUERY = "last water drinking event in the past 2 hours"
WORK_UNFINISHED_SEARCH_QUERY = "unfinished task before the user started using their phone"
POWER_BANK_SEARCH_QUERY = "power bank location last seen in the environment"
SHOPPING_MILK_SEARCH_QUERY = "earlier conversation about shopping list, buying milk at supermarket"
CHARGER_CLUTTER_SEARCH_QUERY = "previous tidy state of the charging area and desk items"
DIVIDED_ATTENTION_SEARCH_QUERY = "recent similar divided-attention incidents (phone + walking/pushing cart/cooking)"


# ============================================================
# 轻量关键词检测。只做粗筛，决定是否走强制分支。
# ============================================================

_POWER_BANK_KEYWORDS = (
    "power bank", "powerbank", "portable charger", "mobile battery",
    "power-bank", "charger", "充电宝", "移动电源",
)

_LOOKING_VERBS = (
    "look", "looking", "search", "searching", "find", "finding",
    "rummage", "rummaging", "where is", "can't find", "cannot find",
    "找", "寻找", "翻找",
)

_PHONE_KEYWORDS = (
    "phone", "smartphone", "cell phone", "mobile phone", "cellphone",
    "手机", "玩手机", "刷手机",
)

_PHONE_USAGE_VERBS = (
    "using", "looking at", "staring at", "scrolling", "swiping", "tapping",
    "watching", "playing", "texting", "chatting",
    "hold", "holding",
    "玩", "刷", "看", "盯着", "滑",
)

# --- "分心组合" 关键词：当这些场景和 phone 共现时，优先走 LLM proactive 而不是
#     demo "work_unfinished" 剧本分支（因为 divided-attention 是真实安全问题）。
_DIVIDED_ATTENTION_KEYWORDS = (
    # 推 xx / 推车
    "cart", "stroller", "trolley", "pushing",
    "推车", "推着", "推购物车", "推婴儿车",
    # 走路 / 过马路 / 交通
    "crossing", "street", "road", "traffic", "crosswalk", "crossroad",
    "过马路", "人行横道", "马路",
    # 厨房 / 烹饪
    "stove", "flame", "cooking", "boiling", "frying", "chopping", "slicing",
    "cutting", "knife", "hot pan", "kettle",
    "炒菜", "做饭", "切菜", "锅", "炉", "刀",
    # 楼梯 / 骑行 / 抱小孩
    "stairs", "staircase", "bike", "bicycle", "scooter", "e-scooter",
    "riding", "carrying a child", "baby in arms",
    "楼梯", "自行车", "电动车", "抱", "抱着",
)

# --- 超市场景 ---
_SUPERMARKET_KEYWORDS = (
    "supermarket", "grocery store", "grocery", "market",
    "convenience store", "mart", "aisle", "shopping cart",
    "cashier", "checkout",
    "超市", "超级市场", "便利店", "购物车", "收银",
)

_BEVERAGE_KEYWORDS = (
    "drink", "beverage", "bottle", "juice", "soda", "water", "soft drink",
    "carton", "pack",
    "饮料", "瓶装", "果汁", "可乐", "水", "牛奶",
)

# --- 充电设备混乱场景 ---
_CHARGER_DEVICE_KEYWORDS = (
    "charger", "charging cable", "power cable", "power adapter",
    "power bank", "laptop", "notebook", "computer", "macbook",
    "desk", "desktop",
    "充电器", "充电线", "数据线", "电源线", "充电宝", "电脑", "笔记本", "书桌",
)

_CLUTTER_KEYWORDS = (
    "messy", "cluttered", "clutter", "disorganized", "tangled",
    "scattered", "in a mess", "chaotic", "piled", "strewn",
    "混乱", "杂乱", "乱", "凌乱", "缠绕", "堆",
)


def caption_mentions_power_bank_search(caption_text: str) -> bool:
    """粗检：caption 是否同时包含"寻找"类动词 + "充电宝"类名词。"""
    if not caption_text:
        return False
    text = caption_text.lower()
    has_item = any(k in text for k in _POWER_BANK_KEYWORDS)
    has_verb = any(v in text for v in _LOOKING_VERBS)
    return has_item and has_verb


def caption_mentions_phone_usage(caption_text: str) -> bool:
    """粗检：caption 是否描述用户在使用/玩手机。

    用于触发"工作还没完成"的强制提醒。
    """
    if not caption_text:
        return False
    text = caption_text.lower()
    has_phone = any(k in text for k in _PHONE_KEYWORDS)
    has_verb = any(v in text for v in _PHONE_USAGE_VERBS)
    return has_phone and has_verb


def caption_mentions_divided_attention(caption_text: str) -> bool:
    """粗检：caption 是否描述"一边玩手机 + 一边做别的需要注意力的事"。

    命中时 demo 模式也让 LLM（正式 v2 proactive prompt）处理，因为 divided-
    attention 是真实安全问题，需要具体场景化提醒，而不是套用"工作未完成"剧本。
    """
    if not caption_text:
        return False
    text = caption_text.lower()
    has_phone = any(k in text for k in _PHONE_KEYWORDS)
    has_verb = any(v in text for v in _PHONE_USAGE_VERBS)
    has_divided = any(k in text for k in _DIVIDED_ATTENTION_KEYWORDS)
    return has_phone and has_verb and has_divided


# LLM 发出 [search] 时，我们用它的 query 文本判断是不是"持续玩手机"语义，
# 如果是 → demo pipeline 会跳过 pass2 直接输出固定的"工作未完成"提醒。
_SUSTAINED_PHONE_QUERY_KEYWORDS = (
    "sustained phone", "prolonged phone", "continuous phone",
    "phone use", "on phone", "on the phone", "scrolling",
    "screen time", "focus history",
    "持续", "长时间", "一直玩手机", "刷手机",
)


def llm_query_is_sustained_phone(query_text: str) -> bool:
    """判断 LLM [search] 的 query 是不是"持续玩手机"语义。

    只要 query 同时包含 phone 类名词 或者直接出现"sustained/prolonged"等
    时长性指示词，就认为命中。
    """
    if not query_text:
        return False
    q = query_text.lower()
    has_phone = any(k in q for k in _PHONE_KEYWORDS)
    has_sustain_cue = any(k in q for k in _SUSTAINED_PHONE_QUERY_KEYWORDS)
    # query 常见写法会同时提 "sustained phone"（两词连写），这时 has_sustain_cue 已命中
    return has_sustain_cue or (has_phone and "time" in q) or (has_phone and "focus" in q)


def llm_query_is_divided_attention(query_text: str) -> bool:
    """判断 LLM [search] 的 query 是不是"玩手机 + 推车/过马路/做饭..."语义。

    需要 phone 和 divided-attention 关键词同时在 query 里出现。
    """
    if not query_text:
        return False
    q = query_text.lower()
    has_phone = any(k in q for k in _PHONE_KEYWORDS)
    has_divided = any(k in q for k in _DIVIDED_ATTENTION_KEYWORDS)
    return has_phone and has_divided


def caption_mentions_supermarket_shopping(caption_text: str) -> bool:
    """粗检：caption 是否描述用户在超市买饮料/购物。

    触发"别忘了买牛奶"提醒（1h 前讨论过的任务）。
    """
    if not caption_text:
        return False
    text = caption_text.lower()
    has_market = any(k in text for k in _SUPERMARKET_KEYWORDS)
    has_beverage = any(k in text for k in _BEVERAGE_KEYWORDS)
    # 超市 + 饮料 同时出现；也允许单独提到超市/便利店 + 购物意图
    return has_market and has_beverage


def caption_mentions_charger_clutter(caption_text: str) -> bool:
    """粗检：caption 是否描述充电设备/桌面摆放混乱。

    触发"整理物品清单"提醒。
    """
    if not caption_text:
        return False
    text = caption_text.lower()
    has_device = any(k in text for k in _CHARGER_DEVICE_KEYWORDS)
    has_clutter = any(k in text for k in _CLUTTER_KEYWORDS)
    return has_device and has_clutter
