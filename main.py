import os
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Roblox ML Balance Server", version="1.0.2-ml")

API_KEY = os.getenv("API_KEY", "#1480epvp+6q9=07")

DEFAULT_RAIDS = [
    "좀비 골렘",
    "아이스 골렘",
    "피라미드의 수호자",
    "사이클롭스 로봇",
    "미노타우로스 로봇",
    "사이클롭스 쌍둥이1",
    "물질의 수호자 로봇",
    "미노타우로스",
]

DEFAULT_WEAPONS = [
    "하슘 블랙홀검",
    "로렌슘 검",
    "노벨륨 지팡이",
    "아인슈타이늄 지팡이",
    "니호늄 닌자검",
    "뢴트게늄 언월도",
    "티타늄 검",
    "텅스텐 검",
    "강철검",
]

# 기본 추천값입니다. 실제 레이드 기록이 쌓이면 RandomForest + 누적 기여도 기반으로 보정됩니다.
DEFAULT_RAID_PRIORS: Dict[str, Dict[str, float]] = {
    "좀비 골렘": {
        "티타늄 검": 28,
        "텅스텐 검": 20,
        "강철검": 15,
        "로렌슘 검": 12,
        "니호늄 닌자검": 9,
        "하슘 블랙홀검": 6,
        "노벨륨 지팡이": 5,
        "아인슈타이늄 지팡이": 3,
        "뢴트게늄 언월도": 2,
    },
    "아이스 골렘": {
        "아인슈타이늄 지팡이": 27,
        "노벨륨 지팡이": 20,
        "로렌슘 검": 16,
        "티타늄 검": 12,
        "텅스텐 검": 9,
        "하슘 블랙홀검": 6,
        "니호늄 닌자검": 5,
        "뢴트게늄 언월도": 3,
        "강철검": 2,
    },
    "피라미드의 수호자": {
        "뢴트게늄 언월도": 24,
        "니호늄 닌자검": 18,
        "텅스텐 검": 16,
        "티타늄 검": 12,
        "로렌슘 검": 10,
        "하슘 블랙홀검": 8,
        "노벨륨 지팡이": 6,
        "아인슈타이늄 지팡이": 4,
        "강철검": 2,
    },
    "사이클롭스 로봇": {
        "하슘 블랙홀검": 26,
        "로렌슘 검": 18,
        "뢴트게늄 언월도": 15,
        "티타늄 검": 12,
        "텅스텐 검": 10,
        "니호늄 닌자검": 8,
        "노벨륨 지팡이": 5,
        "아인슈타이늄 지팡이": 4,
        "강철검": 2,
    },
    "미노타우로스 로봇": {
        "로렌슘 검": 25,
        "하슘 블랙홀검": 19,
        "텅스텐 검": 14,
        "티타늄 검": 11,
        "뢴트게늄 언월도": 10,
        "니호늄 닌자검": 8,
        "노벨륨 지팡이": 6,
        "아인슈타이늄 지팡이": 5,
        "강철검": 2,
    },
    "사이클롭스 쌍둥이1": {
        "니호늄 닌자검": 23,
        "하슘 블랙홀검": 19,
        "로렌슘 검": 15,
        "뢴트게늄 언월도": 12,
        "티타늄 검": 10,
        "텅스텐 검": 8,
        "노벨륨 지팡이": 6,
        "아인슈타이늄 지팡이": 5,
        "강철검": 2,
    },
    "물질의 수호자 로봇": {
        "하슘 블랙홀검": 30,
        "로렌슘 검": 20,
        "아인슈타이늄 지팡이": 13,
        "노벨륨 지팡이": 10,
        "뢴트게늄 언월도": 9,
        "니호늄 닌자검": 7,
        "티타늄 검": 5,
        "텅스텐 검": 4,
        "강철검": 2,
    },
    "미노타우로스": {
        "뢴트게늄 언월도": 24,
        "로렌슘 검": 18,
        "니호늄 닌자검": 15,
        "티타늄 검": 12,
        "텅스텐 검": 11,
        "하슘 블랙홀검": 8,
        "노벨륨 지팡이": 6,
        "아인슈타이늄 지팡이": 4,
        "강철검": 2,
    },
}

raid_records: List[Dict[str, Any]] = []
balance_records: List[Dict[str, Any]] = []
raid_model: Optional[RandomForestClassifier] = None
balance_model: Optional[RandomForestClassifier] = None
last_raid_train_at: Optional[str] = None
last_balance_train_at: Optional[str] = None
cached_raid_recommendations: Dict[str, List[Dict[str, Any]]] = {}
cached_balance_panels: Dict[str, Any] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_api_key(x_api_key: Optional[str]) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def get_metadata_lists(metadata: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    metadata = metadata or {}

    raids = []
    for key in ["knownRaids", "raidNames", "bossNames"]:
        raw = metadata.get(key)
        if isinstance(raw, list):
            raids.extend([normalize_text(x) for x in raw if normalize_text(x)])

    weapons = []
    for key in ["knownWeapons", "raidRewardNames", "weaponNames"]:
        raw = metadata.get(key)
        if isinstance(raw, list):
            weapons.extend([normalize_text(x) for x in raw if normalize_text(x)])

    raids = sorted(list(dict.fromkeys(DEFAULT_RAIDS + raids)))
    weapons = sorted(list(dict.fromkeys(DEFAULT_WEAPONS + weapons)))
    return {"raids": raids, "weapons": weapons}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


class RaidTrainRequest(BaseModel):
    reason: str = "manual"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    records: List[Dict[str, Any]] = Field(default_factory=list)


class BalanceTrainRequest(BaseModel):
    reason: str = "manual"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    records: List[Dict[str, Any]] = Field(default_factory=list)


class BalanceAnalyzeRequest(BaseModel):
    mode: str = "pretrained"
    startDate: str = ""
    endDate: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


def normalize_raid_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    boss = normalize_text(record.get("boss") or record.get("monster") or record.get("raid"))
    weapon = normalize_text(record.get("weapon") or record.get("tool") or record.get("item"))
    if not boss:
        return None
    if not weapon:
        weapon = "Unknown"
    return {
        "boss": boss,
        "monster": boss,
        "weapon": weapon,
        "userId": int(safe_float(record.get("userId"), 0)),
        "contribution": safe_float(record.get("contribution"), 0),
        "timestamp": int(safe_float(record.get("timestamp"), 0)),
    }


def train_raid_model(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global raid_model, last_raid_train_at, cached_raid_recommendations

    lists = get_metadata_lists(metadata)
    raids = lists["raids"]
    weapons = lists["weapons"]
    raid_index = {name: i for i, name in enumerate(raids)}
    weapon_index = {name: i for i, name in enumerate(weapons)}

    x_rows = []
    y_rows = []

    # 실제 레이드 기록: 최고 기여도 플레이어가 장착한 무기를 정답 라벨로 사용합니다.
    for record in raid_records:
        boss = normalize_text(record.get("boss") or record.get("monster"))
        weapon = normalize_text(record.get("weapon"))
        if boss not in raid_index or weapon not in weapon_index:
            continue
        contribution = max(1.0, safe_float(record.get("contribution"), 0))
        x_rows.append([raid_index[boss], min(contribution, 10000) / 10000.0])
        y_rows.append(weapon_index[weapon])

    # 데이터가 적어도 반드시 RandomForestClassifier를 학습시키기 위한 기본 prior 학습 샘플입니다.
    # 실제 기록이 쌓이면 위의 실제 샘플이 함께 들어가 모델 확률에 반영됩니다.
    for boss in raids:
        prior = prior_for_boss(boss, weapons)
        ranked = sorted(prior.items(), key=lambda item: item[1], reverse=True)
        for rank, pair in enumerate(ranked[:min(5, len(ranked))]):
            weapon, percent = pair
            repeat_count = max(1, 5 - rank)
            for _ in range(repeat_count):
                x_rows.append([raid_index[boss], min(10000.0, percent * 100.0) / 10000.0])
                y_rows.append(weapon_index[weapon])

    raid_model = RandomForestClassifier(
        n_estimators=80,
        random_state=42,
        class_weight="balanced_subsample",
    )
    raid_model.fit(np.array(x_rows), np.array(y_rows))

    last_raid_train_at = now_iso()
    cached_raid_recommendations = {boss: recommendations_for_boss(boss, metadata) for boss in DEFAULT_RAIDS}
    return {
        "ok": True,
        "trained": True,
        "model": "RandomForestClassifier",
        "records": len(raid_records),
        "trainingRows": len(x_rows),
        "updatedAt": last_raid_train_at,
    }


def prior_for_boss(boss: str, weapons: List[str]) -> Dict[str, float]:
    prior = DEFAULT_RAID_PRIORS.get(boss)
    if prior is None:
        rng = random.Random(abs(hash(boss)) % 100000)
        raw = {weapon: rng.uniform(1, 20) for weapon in weapons}
    else:
        raw = {weapon: float(prior.get(weapon, 1)) for weapon in weapons}
    total = sum(raw.values()) or 1
    return {weapon: (value / total) * 100 for weapon, value in raw.items()}


def empirical_for_boss(boss: str, weapons: List[str]) -> Dict[str, float]:
    scores = {weapon: 0.0 for weapon in weapons}
    for record in raid_records:
        record_boss = normalize_text(record.get("boss") or record.get("monster"))
        weapon = normalize_text(record.get("weapon"))
        if record_boss == boss and weapon in scores:
            scores[weapon] += max(1.0, safe_float(record.get("contribution"), 0))
    total = sum(scores.values())
    if total <= 0:
        return {weapon: 0.0 for weapon in weapons}
    return {weapon: (value / total) * 100 for weapon, value in scores.items()}


def model_for_boss(boss: str, raids: List[str], weapons: List[str]) -> Dict[str, float]:
    if raid_model is None or boss not in raids:
        return {weapon: 0.0 for weapon in weapons}

    raid_idx = raids.index(boss)
    x = np.array([[raid_idx, 0.5]])
    proba = raid_model.predict_proba(x)[0]
    result = {weapon: 0.0 for weapon in weapons}
    for class_id, probability in zip(raid_model.classes_, proba):
        class_id = int(class_id)
        if 0 <= class_id < len(weapons):
            result[weapons[class_id]] = float(probability) * 100
    return result


def recommendations_for_boss(boss: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    lists = get_metadata_lists(metadata)
    raids = lists["raids"]
    weapons = lists["weapons"]

    prior = prior_for_boss(boss, weapons)
    empirical = empirical_for_boss(boss, weapons)
    model_scores = model_for_boss(boss, raids, weapons)

    result = []
    for weapon in weapons:
        # 기본값 55%, 실제 누적 데이터 25%, RandomForest 예측 20%를 혼합합니다.
        score = prior.get(weapon, 0) * 0.55 + empirical.get(weapon, 0) * 0.25 + model_scores.get(weapon, 0) * 0.20
        result.append({"weapon": weapon, "percent": round(score, 2)})

    total = sum(item["percent"] for item in result) or 1
    for item in result:
        item["percent"] = round((item["percent"] / total) * 100, 2)

    result.sort(key=lambda item: item["percent"], reverse=True)
    return result[:9]


def normalize_balance_record(record: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(record)
    copied["eventType"] = normalize_text(copied.get("eventType") or copied.get("type") or "unknown")
    copied["timestamp"] = int(safe_float(copied.get("timestamp"), 0))
    return copied


def parse_date_to_timestamp(value: str, is_end: bool = False) -> Optional[int]:
    value = normalize_text(value)
    if not value:
        return None
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"]
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            if is_end:
                dt = dt.replace(hour=23, minute=59, second=59)
            else:
                dt = dt.replace(hour=0, minute=0, second=0)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            pass
    try:
        return int(float(value))
    except Exception:
        return None


def filter_records_by_period(records: List[Dict[str, Any]], start_date: str = "", end_date: str = "") -> List[Dict[str, Any]]:
    start_ts = parse_date_to_timestamp(start_date, False)
    end_ts = parse_date_to_timestamp(end_date, True)
    if start_ts is None and end_ts is None:
        return records

    filtered: List[Dict[str, Any]] = []
    for record in records:
        ts = int(safe_float(record.get("timestamp"), 0))
        if ts <= 0:
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        filtered.append(record)
    return filtered


def aggregate_balance_features(records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    records = records if records is not None else balance_records

    weapon_usage: Dict[str, Dict[str, float]] = {}
    boss_times: Dict[str, List[float]] = {}
    quiz_stats: Dict[str, Dict[str, float]] = {}
    hourly_flow: Dict[int, float] = {}
    map_risk: Dict[str, Dict[str, float]] = {}

    for record in records:
        event_type = normalize_text(record.get("eventType"))

        if event_type in ["weapon_usage", "weapon_crafted", "weapon_equipped"]:
            weapon = normalize_text(record.get("weapon")) or "Unknown"
            bucket = weapon_usage.setdefault(weapon, {"crafted": 0.0, "equipped": 0.0})
            bucket["crafted"] += safe_float(record.get("crafted"), safe_float(record.get("amount"), 1))
            bucket["equipped"] += safe_float(record.get("equipped"), safe_float(record.get("used"), 0))

        elif event_type in ["boss_clear_time", "monster_clear_time"]:
            boss = normalize_text(record.get("boss") or record.get("monster")) or "Unknown"
            seconds = safe_float(record.get("seconds") or record.get("clearTime"), 0)
            if seconds > 0:
                boss_times.setdefault(boss, []).append(seconds)

        elif event_type == "quiz_result":
            level = normalize_text(record.get("level") or record.get("mission") or "기본")
            quiz_level_aliases = {"고급": "상급", "Knowledge": "지식", "knowledge": "지식", "KNOWLEDGE": "지식"}
            level = quiz_level_aliases.get(level, level)
            bucket = quiz_stats.setdefault(level, {"clear": 0.0, "giveup": 0.0})
            bucket["clear"] += safe_float(record.get("clear"), 0)
            bucket["giveup"] += safe_float(record.get("giveup"), 0)

        elif event_type == "hourly_flow":
            hour = int(safe_float(record.get("hour"), 0)) % 24
            hourly_flow[hour] = hourly_flow.get(hour, 0.0) + safe_float(record.get("players"), 0) + safe_float(record.get("huntMinutes"), 0) / 10.0

        elif event_type == "map_risk":
            map_name = normalize_text(record.get("map")) or "Unknown"
            map_name_aliases = {
                "숲": "연금술사의 숲",
                "사막": "신들의 사막",
                "빙하": "얼어붙은 세계",
                "얼어붙은 빙판길": "얼어붙은 세계",
                "화산": "잃어버린 도시",
                "입자 가속기": "입자가속기",
            }
            map_name = map_name_aliases.get(map_name, map_name)
            bucket = map_risk.setdefault(map_name, {"deaths": 0.0, "stayMinutes": 0.0})
            bucket["deaths"] += safe_float(record.get("deaths"), 0)
            bucket["stayMinutes"] += safe_float(record.get("stayMinutes"), 0)

    if not weapon_usage:
        weapon_usage = {
            "하슘 블랙홀검": {"crafted": 20, "equipped": 17},
            "로렌슘 검": {"crafted": 28, "equipped": 20},
            "노벨륨 지팡이": {"crafted": 12, "equipped": 5},
            "아인슈타이늄 지팡이": {"crafted": 15, "equipped": 8},
            "니호늄 닌자검": {"crafted": 18, "equipped": 12},
            "뢴트게늄 언월도": {"crafted": 14, "equipped": 10},
            "티타늄 검": {"crafted": 45, "equipped": 22},
            "텅스텐 검": {"crafted": 36, "equipped": 18},
            "강철검": {"crafted": 80, "equipped": 16},
        }

    if not boss_times:
        boss_times = {
            "좀비 골렘": [150, 170, 165],
            "아이스 골렘": [190, 210, 205],
            "피라미드의 수호자": [240, 260, 255],
            "사이클롭스 로봇": [280, 300, 295],
            "미노타우로스 로봇": [300, 320, 315],
            "사이클롭스 쌍둥이1": [320, 340, 335],
            "물질의 수호자 로봇": [340, 360, 350],
            "미노타우로스": [310, 330, 320],
        }

    if not quiz_stats:
        quiz_stats = {
            "초급": {"clear": 80, "giveup": 12},
            "중급": {"clear": 55, "giveup": 28},
            "상급": {"clear": 30, "giveup": 35},
            "지식": {"clear": 22, "giveup": 30},
        }

    if not hourly_flow:
        hourly_flow = {hour: float(20 + (hour % 6) * 9 + (35 if 18 <= hour <= 22 else 0)) for hour in range(24)}

    if not map_risk:
        map_risk = {
            "연금술사의 숲": {"deaths": 10, "stayMinutes": 450},
            "신들의 사막": {"deaths": 25, "stayMinutes": 380},
            "얼어붙은 세계": {"deaths": 18, "stayMinutes": 300},
            "잃어버린 도시": {"deaths": 42, "stayMinutes": 260},
            "입자가속기": {"deaths": 55, "stayMinutes": 210},
        }

    weapon_ratios = []
    for weapon, values in weapon_usage.items():
        crafted = max(1.0, values.get("crafted", 0.0))
        equipped = values.get("equipped", 0.0)
        weapon_ratios.append({"label": weapon, "value": round((equipped / crafted) * 100, 2)})

    boss_avg_times = []
    for boss, times in boss_times.items():
        avg_time = sum(times) / max(1, len(times))
        boss_avg_times.append({"label": boss, "value": round(avg_time, 2)})

    quiz_matrix = []
    quiz_y_labels = []
    for level, values in quiz_stats.items():
        clear = values.get("clear", 0.0)
        giveup = values.get("giveup", 0.0)
        quiz_y_labels.append(level)
        quiz_matrix.append([clear, giveup])

    hourly_items = [{"label": f"{hour:02d}시", "value": round(value, 2)} for hour, value in sorted(hourly_flow.items())]

    map_y_labels = []
    map_matrix = []
    for map_name, values in map_risk.items():
        map_y_labels.append(map_name)
        deaths = values.get("deaths", 0.0)
        stay = values.get("stayMinutes", 0.0)
        risk = deaths / max(1.0, stay / 60.0)
        map_matrix.append([round(deaths, 2), round(stay, 2), round(risk, 2)])

    avg_weapon_use = sum(item["value"] for item in weapon_ratios) / max(1, len(weapon_ratios))
    avg_boss_time = sum(item["value"] for item in boss_avg_times) / max(1, len(boss_avg_times))
    total_clear = sum(row[0] for row in quiz_matrix)
    total_giveup = sum(row[1] for row in quiz_matrix)
    giveup_ratio = total_giveup / max(1.0, total_clear + total_giveup)
    hourly_values = [item["value"] for item in hourly_items]
    peakiness = max(hourly_values) / max(1.0, sum(hourly_values) / max(1, len(hourly_values)))
    avg_map_risk = sum(row[2] for row in map_matrix) / max(1, len(map_matrix))

    return {
        "weaponRatios": sorted(weapon_ratios, key=lambda x: x["value"], reverse=True),
        "bossAvgTimes": sorted(boss_avg_times, key=lambda x: x["value"], reverse=True),
        "quizYLabels": quiz_y_labels,
        "quizMatrix": quiz_matrix,
        "hourlyItems": hourly_items,
        "mapYLabels": map_y_labels,
        "mapMatrix": map_matrix,
        "featureVector": [avg_weapon_use, avg_boss_time, giveup_ratio * 100, peakiness * 10, avg_map_risk],
    }


def train_balance_model() -> Dict[str, Any]:
    global balance_model, last_balance_train_at

    # 5개 패널별 후보 4개를 고르는 분류 모델입니다.
    # 실제 라이브 데이터가 적을 때도 동작하도록 합성 학습 샘플을 함께 사용합니다.
    rng = random.Random(42)
    x_rows = []
    y_rows = []
    for _ in range(240):
        weapon_use = rng.uniform(10, 95)
        boss_time = rng.uniform(100, 420)
        giveup = rng.uniform(2, 65)
        peak = rng.uniform(8, 35)
        map_risk = rng.uniform(1, 25)
        features = [weapon_use, boss_time, giveup, peak, map_risk]
        x_rows.append(features)

        if weapon_use < 35:
            label = 0
        elif boss_time > 300:
            label = 1
        elif giveup > 40:
            label = 2
        else:
            label = 3
        y_rows.append(label)

    if len(balance_records) >= 3:
        features = aggregate_balance_features()["featureVector"]
        x_rows.append(features)
        y_rows.append(int(max(range(4), key=lambda i: [100 - features[0], features[1], features[2], features[4]][i])))

    balance_model = RandomForestClassifier(n_estimators=100, random_state=77)
    balance_model.fit(np.array(x_rows), np.array(y_rows))
    last_balance_train_at = now_iso()
    return {"ok": True, "trained": True, "records": len(balance_records), "updatedAt": last_balance_train_at}


def choose_candidate(features: List[float], panel_offset: int) -> int:
    if balance_model is None:
        train_balance_model()
    assert balance_model is not None
    predicted = int(balance_model.predict(np.array([features]))[0])
    return (predicted + panel_offset) % 4


def build_balance_panels(mode: str, start_date: str = "", end_date: str = "") -> Dict[str, Any]:
    if mode == "retrain":
        train_balance_model()
    elif balance_model is None:
        train_balance_model()

    selected_records = filter_records_by_period(balance_records, start_date, end_date)
    ag = aggregate_balance_features(selected_records)
    features = ag["featureVector"]

    panel_candidates = {
        "Frame1": [
            "무기 제작량 대비 실제 장착률이 낮은 무기가 있습니다. 제작 재료 비용 또는 무기 성능을 조정하세요.",
            "상위 무기 쏠림이 감지됩니다. 중간 단계 무기의 성장 구간을 강화하세요.",
            "초반 무기의 사용률이 낮습니다. 초반 보상 루트를 더 명확하게 안내하세요.",
            "무기 사용 분포가 비교적 안정적입니다. 현재 밸런스를 유지하면서 신규 데이터만 관찰하세요.",
        ],
        "Frame2": [
            "일부 보스의 처치 시간이 너무 깁니다. 보스 체력 또는 패턴 난이도를 낮추는 것을 권장합니다.",
            "보스별 처치 시간 격차가 큽니다. 보상량을 난이도에 맞게 재분배하세요.",
            "짧은 시간에 처치되는 보스가 있습니다. 반복 파밍 속도를 제한하거나 보상 확률을 조정하세요.",
            "몬스터/보스 처치 시간이 안정적입니다. 신규 레이드 추가 전까지 현재 값을 유지하세요.",
        ],
        "Frame3": [
            "퀴즈 포기 비율이 높은 구간이 있습니다. 문제 난이도 또는 제한 시간을 완화하세요.",
            "중급 이후 클리어율이 급감합니다. 힌트 제공 또는 단계별 보상을 추가하세요.",
            "초급 퀴즈가 너무 쉽습니다. 초급 보상을 낮추거나 중급 진입을 빠르게 유도하세요.",
            "퀴즈 미션 흐름이 안정적입니다. 포기율이 높은 문제만 개별 점검하세요.",
        ],
        "Frame4": [
            "특정 시간대에 플레이가 집중됩니다. 피크 시간 보상 또는 서버 부하 관리를 준비하세요.",
            "비활성 시간대가 길게 나타납니다. 시간대별 접속 보너스를 추가해 흐름을 분산하세요.",
            "사냥 시간과 접속자 흐름이 어긋납니다. 이벤트 시간을 플레이 피크에 맞추세요.",
            "하루 플레이 흐름이 안정적입니다. 현재 이벤트 스케줄을 유지하세요.",
        ],
        "Frame5": [
            "일부 맵의 사망률이 높습니다. 위험 구간 배치 또는 몬스터 밀도를 조정하세요.",
            "체류 시간이 낮은 맵이 있습니다. 보상 또는 이동 동선을 개선하세요.",
            "체류 시간 대비 사망 수가 많은 맵이 있습니다. 체크포인트와 회복 수단을 추가하세요.",
            "맵별 위험도와 체류 시간이 안정적입니다. 신규 맵 추가 전까지 현재 구조를 유지하세요.",
        ],
    }

    panels = {}
    frame_names = ["Frame1", "Frame2", "Frame3", "Frame4", "Frame5"]
    for index, frame_name in enumerate(frame_names):
        candidate_index = choose_candidate(features, index)
        suggestion = panel_candidates[frame_name][candidate_index]
        if frame_name == "Frame1":
            graph = {"type": "horizontal_bar", "items": ag["weaponRatios"][:9]}
            title = "무기 제작 및 실제 사용 비율"
        elif frame_name == "Frame2":
            graph = {"type": "horizontal_bar", "items": ag["bossAvgTimes"][:8]}
            title = "몬스터/보스별 평균 처치 시간"
        elif frame_name == "Frame3":
            graph = {"type": "heatmap", "xLabels": ["클리어", "포기"], "yLabels": ag["quizYLabels"], "matrix": ag["quizMatrix"]}
            title = "퀴즈 미션 클리어/포기 비율"
        elif frame_name == "Frame4":
            graph = {"type": "vertical_bar", "items": ag["hourlyItems"]}
            title = "하루 시간대별 유저 플레이 흐름"
        else:
            graph = {"type": "heatmap", "xLabels": ["사망", "체류분", "위험도"], "yLabels": ag["mapYLabels"], "matrix": ag["mapMatrix"]}
            title = "맵별 사망 및 체류 시간"

        panels[frame_name] = {
            "title": title,
            "suggestion": "⚠️ " + suggestion,
            "graph": graph,
            "candidateIndex": candidate_index + 1,
        }

    result = {
        "ok": True,
        "mode": mode,
        "startDate": start_date,
        "endDate": end_date,
        "updatedAt": now_iso(),
        "trainedAt": last_balance_train_at,
        "model": "RandomForestClassifier",
        "panels": panels,
    }
    global cached_balance_panels
    cached_balance_panels = result
    return result


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "Roblox ML Balance Server", "updatedAt": now_iso()}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "raidRecords": len(raid_records),
        "balanceRecords": len(balance_records),
        "raidModelReady": raid_model is not None,
        "balanceModelReady": balance_model is not None,
        "updatedAt": now_iso(),
    }


@app.post("/api/raid/train")
def api_raid_train(payload: RaidTrainRequest, x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_api_key(x_api_key)
    added = 0
    for record in payload.records:
        normalized = normalize_raid_record(record)
        if normalized is not None:
            raid_records.append(normalized)
            added += 1
    result = train_raid_model(payload.metadata)
    result["added"] = added
    result["reason"] = payload.reason
    return result


@app.get("/api/raid/recommendations/all")
def api_raid_recommendations_all(x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_api_key(x_api_key)
    if last_raid_train_at is None or not cached_raid_recommendations:
        train_raid_model({})
    return {
        "ok": True,
        "updatedAt": now_iso(),
        "trainedAt": last_raid_train_at,
        "model": "RandomForestClassifier",
        "recommendations": cached_raid_recommendations,
        "data": cached_raid_recommendations,
    }


@app.post("/api/balance/train")
def api_balance_train(payload: BalanceTrainRequest, x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_api_key(x_api_key)
    added = 0
    for record in payload.records:
        balance_records.append(normalize_balance_record(record))
        added += 1
    result = train_balance_model()
    result["added"] = added
    result["reason"] = payload.reason
    return result


@app.post("/api/balance/analyze")
def api_balance_analyze(payload: BalanceAnalyzeRequest, x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_api_key(x_api_key)
    mode = "retrain" if payload.mode == "retrain" else "pretrained"
    return build_balance_panels(mode, payload.startDate, payload.endDate)


# 기존 RenderHttp 방식 호환용: 레이드 기록 1개 또는 여러 개를 저장합니다.
@app.post("/update_raid")
async def legacy_update_raid(request: Request, x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_api_key(x_api_key)
    body = await request.json()
    records = body.get("records") if isinstance(body, dict) else None
    if not isinstance(records, list):
        records = [body]
    added = 0
    for record in records:
        if isinstance(record, dict):
            normalized = normalize_raid_record(record)
            if normalized is not None:
                raid_records.append(normalized)
                added += 1
    result = train_raid_model({})
    result["added"] = added
    return result


# 기존 RenderHttp 방식 호환용: /predict?monster=ALL_MONSTERS 또는 /predict?monster=좀비 골렘
@app.get("/predict")
def legacy_predict(
    monster: str = Query(default="ALL_MONSTERS"),
    x_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    require_api_key(x_api_key)
    if last_raid_train_at is None:
        train_raid_model({})

    if not cached_raid_recommendations:
        train_raid_model({})

    if monster == "ALL_MONSTERS":
        return {"ok": True, "model": "RandomForestClassifier", "recommendations": cached_raid_recommendations, "data": cached_raid_recommendations}

    return {
        "ok": True,
        "model": "RandomForestClassifier",
        "monster": monster,
        "boss": monster,
        "recommendations": cached_raid_recommendations.get(monster, recommendations_for_boss(monster)),
    }


# Render가 서버를 시작한 직후에도 기본 모델이 준비되도록 초기 학습을 수행합니다.
train_raid_model({})
train_balance_model()
