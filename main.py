from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import json
import os
import threading

# Flask 웹 서버 애플리케이션 초기화
app = Flask(__name__)

# ==============================================================================
# 🔑 보안 설정
# ==============================================================================
SECRET_API_KEY = "#1480epvp+6q9=07"

# ==============================================================================
# 📁 JSON 데이터 저장 설정
# ==============================================================================

# JSON 파일 위치
# Render Persistent Disk를 사용한다면 이 경로를 디스크 경로에 맞게 변경하세요.
DATA_FILE = "raid_database.json"

# 여러 요청이 동시에 JSON을 수정하는 것을 방지
data_lock = threading.Lock()

# ==============================================================================
# 📦 데이터 저장소
# ==============================================================================

raid_database = []

# 🧠 캐시 메모리
cached_predictions = {}

# ==============================================================================
# 🕒 시간 설정
# ==============================================================================

THREE_MONTHS_SECONDS = 90 * 24 * 60 * 60

# ==============================================================================
# ⚔️ 무기 리스트
# ==============================================================================

WEAPONS = [
    '강철검',
    '구리검',
    '노벨륨 지팡이',
    '로렌슘 검',
    '뢴트게늄 언월도',
    '아인슈타이늄 지팡이',
    '텅스텐 검',
    '티타늄검',
    '하슘 블랙홀검'
]

# ==============================================================================
# 👹 몬스터 리스트
# ==============================================================================

MONSTERS = [
    "미노타우로스",
    "사이클롭스 쌍둥이1",
    "미노타우로스 로봇",
    "사이클롭스 로봇",
    "피라미드의 수호자",
    "아이스 골렘",
    "좀비 골렘",
    "물질의 수호자 로봇"
]

# ==============================================================================
# 💾 JSON 파일에서 데이터 불러오기
# ==============================================================================

def load_database():
    global raid_database

    if not os.path.exists(DATA_FILE):
        print("[서버] JSON 데이터 파일이 없습니다. 새로 생성합니다.")
        raid_database = []
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            raid_database = data
        else:
            raid_database = []

        print(
            f"[서버] JSON 데이터 불러오기 완료! "
            f"총 {len(raid_database)}건"
        )

    except Exception as e:
        print(f"[서버] JSON 데이터 불러오기 실패: {e}")
        raid_database = []


# ==============================================================================
# 💾 JSON 파일에 데이터 저장
# ==============================================================================

def save_database():
    global raid_database

    try:
        # 임시 파일에 먼저 저장
        temp_file = DATA_FILE + ".tmp"

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(
                raid_database,
                f,
                ensure_ascii=False,
                indent=2
            )

        # 저장이 정상적으로 끝났으면 실제 파일로 교체
        os.replace(temp_file, DATA_FILE)

        print(
            f"[서버] JSON 저장 완료! "
            f"총 {len(raid_database)}건"
        )

    except Exception as e:
        print(f"[서버] JSON 저장 실패: {e}")


# ==============================================================================
# 🔍 몬스터 이름 정규화
# ==============================================================================

def normalize_monster_name(name):
    if not name:
        return ""

    for m in MONSTERS:
        if m in name:
            return m

    return name


# ==============================================================================
# 🛡️ API Key 인증
# ==============================================================================

def verify_api_key():
    client_key = request.headers.get("X-API-Key")

    if client_key != SECRET_API_KEY:
        return False

    return True


# ==============================================================================
# 📊 데이터 부족 시 안전장치
# ==============================================================================

def get_fallback_stats(monster_records, index):

    weapon_counts = {
        w: 0 for w in WEAPONS
    }

    for record in monster_records:

        w = record.get("weapon")

        if w in weapon_counts:
            weapon_counts[w] += 1

        else:

            for valid_w in WEAPONS:

                if valid_w in str(w):
                    weapon_counts[valid_w] += 1
                    break

    total_count = sum(weapon_counts.values())

    results = []

    # --------------------------------------------------------------------------
    # 실제 데이터가 있는 경우
    # --------------------------------------------------------------------------

    if total_count > 0:

        for weapon in WEAPONS:

            count = weapon_counts[weapon]

            percent = round(
                float(count / total_count * 100),
                1
            )

            results.append({
                "weapon": weapon,
                "percent": percent
            })

    # --------------------------------------------------------------------------
    # 데이터가 하나도 없는 경우
    # --------------------------------------------------------------------------

    else:

        np.random.seed(
            len(raid_database) + index + 42
        )

        raw_probs = np.random.dirichlet(
            np.ones(len(WEAPONS)),
            size=1
        )[0]

        for weapon, prob in zip(
            WEAPONS,
            raw_probs
        ):

            results.append({
                "weapon": weapon,
                "percent": round(
                    float(prob * 100),
                    1
                )
            })

    return results


# ==============================================================================
# 🤖 머신러닝 파이프라인
# ==============================================================================

def run_machine_learning_pipeline():

    global raid_database
    global cached_predictions

    current_time = time.time()

    # --------------------------------------------------------------------------
    # 3개월보다 오래된 데이터 삭제
    # --------------------------------------------------------------------------

    old_count = len(raid_database)

    raid_database = [
        item
        for item in raid_database
        if (
            current_time
            - item.get("timestamp", current_time)
        ) <= THREE_MONTHS_SECONDS
    ]

    new_count = len(raid_database)

    # 오래된 데이터가 삭제되었다면 JSON에도 반영
    if old_count != new_count:

        print(
            f"[서버] 오래된 데이터 {old_count - new_count}건 삭제"
        )

        save_database()

    # --------------------------------------------------------------------------
    # 전체 몬스터 결과
    # --------------------------------------------------------------------------

    all_monster_results = {}

    # --------------------------------------------------------------------------
    # 몬스터별 처리
    # --------------------------------------------------------------------------

    for index, monster in enumerate(MONSTERS):

        monster_records = [
            item
            for item in raid_database
            if item.get("monster") == monster
        ]

        monster_results = []

        # ----------------------------------------------------------------------
        # 데이터가 5개 이상이면 RandomForest
        # ----------------------------------------------------------------------

        if len(monster_records) >= 5:

            try:

                df = pd.DataFrame(
                    monster_records
                )

                df['monster_code'] = df[
                    'monster'
                ].apply(
                    lambda x:
                    MONSTERS.index(x)
                    if x in MONSTERS
                    else 0
                )

                X = df[
                    ['monster_code']
                ]

                y = df[
                    'weapon'
                ]

                # RandomForest
                model = RandomForestClassifier(
                    n_estimators=50,
                    random_state=42
                )

                model.fit(X, y)

                classes = model.classes_

                test_X = pd.DataFrame(
                    [[index]],
                    columns=['monster_code']
                )

                probs = model.predict_proba(
                    test_X
                )[0]

                pred_dict = {
                    cls: prob * 100
                    for cls, prob
                    in zip(classes, probs)
                }

                for weapon in WEAPONS:

                    percent = round(
                        float(
                            pred_dict.get(
                                weapon,
                                0.0
                            )
                        ),
                        1
                    )

                    monster_results.append({
                        "weapon": weapon,
                        "percent": percent
                    })

            except Exception as e:

                print(
                    f"[서버] 머신러닝 학습 중 예외 "
                    f"({monster}): {e}"
                )

                monster_results = get_fallback_stats(
                    monster_records,
                    index
                )

        # ----------------------------------------------------------------------
        # 데이터가 5개 미만
        # ----------------------------------------------------------------------

        else:

            monster_results = get_fallback_stats(
                monster_records,
                index
            )

        # ----------------------------------------------------------------------
        # 높은 확률순 정렬
        # ----------------------------------------------------------------------

        monster_results = sorted(
            monster_results,
            key=lambda x: x['percent'],
            reverse=True
        )

        all_monster_results[
            monster
        ] = monster_results

    # --------------------------------------------------------------------------
    # 캐시 저장
    # --------------------------------------------------------------------------

    cached_predictions = all_monster_results

    print(
        "[서버] 머신러닝 학습 및 캐시 갱신 완료! "
        f"(유효 데이터 총: {len(raid_database)}건)"
    )


# ==============================================================================
# 📡 API 1
# Roblox → Render
# 레이드 데이터 저장
# ==============================================================================

@app.route(
    '/update_raid',
    methods=['POST']
)
def update_raid():

    # --------------------------------------------------------------------------
    # API Key 확인
    # --------------------------------------------------------------------------

    if not verify_api_key():

        return jsonify({
            "error":
            "Unauthorized: Invalid or missing API Key"
        }), 403

    global raid_database

    data = request.json

    if data is None:

        return jsonify({
            "error": "Invalid JSON"
        }), 400

    current_time = time.time()

    added_count = 0

    # --------------------------------------------------------------------------
    # 데이터 저장
    # --------------------------------------------------------------------------

    with data_lock:

        # 단일 데이터
        if isinstance(data, dict):

            item = data.copy()

            item["monster"] = normalize_monster_name(
                item.get("monster", "")
            )

            if "timestamp" not in item:
                item["timestamp"] = current_time

            raid_database.append(item)

            added_count = 1

        # 여러 데이터
        elif isinstance(data, list):

            for original_item in data:

                if not isinstance(
                    original_item,
                    dict
                ):
                    continue

                item = original_item.copy()

                item["monster"] = normalize_monster_name(
                    item.get("monster", "")
                )

                if "timestamp" not in item:
                    item["timestamp"] = current_time

                raid_database.append(item)

                added_count += 1

        else:

            return jsonify({
                "error":
                "Data must be object or array"
            }), 400

        # ----------------------------------------------------------------------
        # ⭐ JSON 파일에 영구 저장
        # ----------------------------------------------------------------------

        save_database()

    # --------------------------------------------------------------------------
    # 머신러닝 갱신
    # --------------------------------------------------------------------------

    run_machine_learning_pipeline()

    return jsonify({
        "status": "success",
        "added": added_count,
        "total": len(raid_database)
    }), 200


# ==============================================================================
# 📡 API 2
# Roblox → Render
# 예측 결과 요청
# ==============================================================================

@app.route(
    '/predict',
    methods=['GET']
)
def predict():

    # --------------------------------------------------------------------------
    # API Key 확인
    # --------------------------------------------------------------------------

    if not verify_api_key():

        return jsonify({
            "error":
            "Unauthorized: Invalid or missing API Key"
        }), 403

    global cached_predictions

    # --------------------------------------------------------------------------
    # 캐시가 비어 있으면 머신러닝 실행
    # --------------------------------------------------------------------------

    if not cached_predictions:

        run_machine_learning_pipeline()

    # --------------------------------------------------------------------------
    # 몬스터 확인
    # --------------------------------------------------------------------------

    monster_query = request.args.get(
        "monster"
    )

    # --------------------------------------------------------------------------
    # 전체 몬스터
    # --------------------------------------------------------------------------

    if (
        monster_query == "ALL_MONSTERS"
        or not monster_query
    ):

        return jsonify(
            cached_predictions
        ), 200

    # --------------------------------------------------------------------------
    # 특정 몬스터
    # --------------------------------------------------------------------------

    else:

        if monster_query in cached_predictions:

            return jsonify(
                cached_predictions[
                    monster_query
                ]
            ), 200

        else:

            return jsonify({
                "error":
                "Monster not found"
            }), 404


# ==============================================================================
# 🚀 서버 시작
# ==============================================================================

if __name__ == '__main__':

    # --------------------------------------------------------------------------
    # 서버 시작 전에 JSON 데이터 불러오기
    # --------------------------------------------------------------------------

    load_database()

    # --------------------------------------------------------------------------
    # 기존 데이터로 머신러닝 캐시 생성
    # --------------------------------------------------------------------------

    run_machine_learning_pipeline()

    # --------------------------------------------------------------------------
    # Flask 시작
    # --------------------------------------------------------------------------

    app.run(
        host='0.0.0.0',
        port=5000
    )
