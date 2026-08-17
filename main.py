from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

app = Flask(__name__)

# 수집된 레이드 데이터를 저장할 메모리 리스트 (서버가 켜져 있는 동안 보관됨)
raid_database = []

# 🕒 3개월(90일)을 초 단위로 계산 (90일 * 24시간 * 60분 * 60초 = 7,776,000초)
THREE_MONTHS_SECONDS = 90 * 24 * 60 * 60

# 9가지 지정된 무기 리스트 (로블록스 클라이언트와 정확히 일치)
WEAPONS = [
    '강철검', '구리검', '노벨륨 지팡이', '로렌슘 검', 
    '뢴트게늄 언월도', '아인슈타이늄 지팡이', '텅스텐 검', '티타늄검', '하슘 블랙홀검'
]

# 8가지 지정된 레이드 몬스터 기준 리스트
MONSTERS = [
    "미노타우로스", 
    "사이클롭스 쌍둥이", 
    "미노타우로스 로봇", 
    "사이클롭스 로봇", 
    "피라미드의 수호자", 
    "아이스 골렘", 
    "좀비 골렘", 
    "물질의 수호자 로봇"
]

# UI에 표시되는 긴 이름을 기본 몬스터 이름으로 정규화하는 함수
def normalize_monster_name(name):
    if not name:
        return ""
    for m in MONSTERS:
        if m in name:
            return m
    return name

# 1. 24시간마다 누적된 데이터를 받아오고 3개월 지난 데이터를 정리하는 엔드포인트
@app.route('/update_raid', methods=['POST'])
def update_raid():
    global raid_database
    data = request.json
    current_time = time.time()
    
    # 새 데이터 추가
    if isinstance(data, dict):
        data["monster"] = normalize_monster_name(data.get("monster", ""))
        if "timestamp" not in data:
            data["timestamp"] = current_time
        raid_database.append(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                item["monster"] = normalize_monster_name(item.get("monster", ""))
                if "timestamp" not in item:
                    item["timestamp"] = current_time
                raid_database.append(item)
                
    # 🕒 3개월(90일)이 지난 오래된 데이터는 즉시 삭제 후 최근 3개월치만 유지
    raid_database = [
        item for item in raid_database 
        if (current_time - item.get("timestamp", current_time)) <= THREE_MONTHS_SECONDS
    ]
    
    print(f"[서버] 데이터 수신 및 3개월 필터링 완료. 현재 유효한 데이터 수: {len(raid_database)}")
    return jsonify({"status": "success", "total": len(raid_database)}), 200

# 2. 유저가 버튼을 눌렀을 때 최근 3개월 데이터를 기반으로 몬스터별 최강 무기 확률을 계산해 주는 엔드포인트
@app.route('/predict', methods=['GET'])
def predict():
    global raid_database
    current_time = time.time()
    
    # 예측을 수행할 때도 3개월 지난 데이터가 있다면 깔끔하게 필터링
    raid_database = [
        item for item in raid_database 
        if (current_time - item.get("timestamp", current_time)) <= THREE_MONTHS_SECONDS
    ]
    
    all_monster_results = {}
    
    for index, monster in enumerate(MONSTERS):
        # 해당 몬스터의 최근 3개월 데이터만 추출
        monster_records = [item for item in raid_database if item.get("monster") == monster]
        
        weapon_counts = {w: 0 for w in WEAPONS}
        for record in monster_records:
            w = record.get("weapon")
            if w in weapon_counts:
                weapon_counts[w] += 1
            else:
                # 무기 이름에 키워드가 포함되어 있다면 매칭
                for valid_w in WEAPONS:
                    if valid_w in str(w):
                        weapon_counts[valid_w] += 1
                        break
                        
        total_count = sum(weapon_counts.values())
        
        monster_results = []
        if total_count > 0:
            # 최근 3개월간 수집된 실제 승리 기여도 데이터 기반 백분율 계산
            for weapon in WEAPONS:
                count = weapon_counts[weapon]
                percent = round(float(count / total_count * 100), 1)
                monster_results.append({
                    "weapon": weapon,
                    "percent": percent
                })
        else:
            # 아직 데이터가 쌓이지 않은 초기 상태라면 기본 분배값 제공 (에러 방지)
            np.random.seed(len(raid_database) + index + 42)
            raw_probs = np.random.dirichlet(np.ones(len(WEAPONS)), size=1)[0]
            for weapon, prob in zip(WEAPONS, raw_probs):
                monster_results.append({
                    "weapon": weapon,
                    "percent": round(float(prob * 100), 1)
                })
        
        # 퍼센트 높은 순(내림차순) 정렬
        monster_results = sorted(monster_results, key=lambda x: x['percent'], reverse=True)
        all_monster_results[monster] = monster_results

    return jsonify(all_monster_results), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
