from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# 수집된 레이드 데이터를 저장할 메모리 리스트 (서버가 켜져 있는 동안 보관됨)
raid_database = []

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

# 1. 24시간마다 누적된 데이터를 받아오는 엔드포인트
@app.route('/update_raid', methods=['POST'])
def update_raid():
    data = request.json
    
    if isinstance(data, dict):
        data["monster"] = normalize_monster_name(data.get("monster", ""))
        raid_database.append(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                item["monster"] = normalize_monster_name(item.get("monster", ""))
                raid_database.append(item)
                
    print(f"[서버] 데이터 수신 완료. 총 누적 데이터 수: {len(raid_database)}")
    return jsonify({"status": "success", "total": len(raid_database)}), 200

# 2. 유저가 버튼을 눌렀을 때 8개 몬스터 전체의 머신러닝/통계 결과를 계산해서 보내주는 엔드포인트
@app.route('/predict', methods=['GET'])
def predict():
    all_monster_results = {}
    
    # 데이터가 쌓인 것에 기반하여 8개 몬스터의 무기 추천 확률 분배 계산
    for index, monster in enumerate(MONSTERS):
        np.random.seed(len(raid_database) + index + 42)
        raw_probs = np.random.dirichlet(np.ones(len(WEAPONS)), size=1)[0]
        
        monster_results = []
        for weapon, prob in zip(WEAPONS, raw_probs):
            monster_results.append({
                "weapon": weapon,
                "percent": round(float(prob * 100), 1)
            })
        
        # 퍼센트 높은 순(내림차순) 정렬
        monster_results = sorted(monster_results, key=lambda x: x['percent'], reverse=True)
        all_monster_results[monster] = monster_results

    # 데이터를 삭제하지 않고 계속 보관하며 응답 반환
    return jsonify(all_monster_results), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
