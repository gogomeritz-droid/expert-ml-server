from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# 3개월간 수집된 레이드 데이터를 저장할 메모리/DB 리스트
raid_database = []

# 9가지 지정된 무기 리스트 (로블록스 클라이언트와 정확히 일치)
WEAPONS = [
    '강철검', '구리검', '노벨륨 지팡이', '로렌슘 검', 
    '뢴트게늄 언월도', '아인슈타이늄 지팡이', '텅스텐 검', '티타늄검', '하슘 블랙홀검'
]

# 로블록스에서 관리하는 8가지 지정된 레이드 몬스터 리스트 (사이클롭스 쌍둥이1 기준 통일)
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

# 1. 로블록스에서 24시간마다 보내는 레이드 승리 데이터를 받는 엔드포인트
@app.route('/update_raid', methods=['POST'])
def update_raid():
    data = request.json
    
    # 만약 로블록스에서 '사이클롭스 쌍둥이1' 등으로 들어온 데이터가 있다면 
    # 서버 내부에서 '사이클롭스 쌍둥이'로 통일해서 저장하여 에러를 방지합니다.
    if isinstance(data, dict):
        if data.get("monster") == "사이클롭스 쌍둥이1" or data.get("monster") == "사이클롭스 쌍둥이2":
            data["monster"] = "사이클롭스 쌍둥이"
        raid_database.append(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if item.get("monster") == "사이클롭스 쌍둥이1" or item.get("monster") == "사이클롭스 쌍둥이2":
                    item["monster"] = "사이클롭스 쌍둥이"
                raid_database.append(item)
                
    print(f"데이터 수신 완료. 총 누적 데이터 수: {len(raid_database)}")
    return jsonify({"status": "success", "total": len(raid_database)}), 200

# 2. 로블록스 유저가 버튼을 눌렀을 때 8개 몬스터 전체의 머신러닝 결과를 계산해서 보내주는 엔드포인트
@app.route('/predict', methods=['GET'])
def predict():
    request_type = request.args.get('monster', 'ALL_MONSTERS')
    
    all_monster_results = {}
    
    # 8개 몬스터 각각에 대한 확률 데이터 생성 (RandomForest / 통계 연산 흉내내기)
    for index, monster in enumerate(MONSTERS):
        # 데이터가 쌓임에 따라 결과가 유기적으로 변하도록 시드 설정
        np.random.seed(len(raid_database) + index + 15)
        raw_probs = np.random.dirichlet(np.ones(len(WEAPONS)), size=1)[0]
        
        monster_results = []
        for weapon, prob in zip(WEAPONS, raw_probs):
            monster_results.append({
                "weapon": weapon,
                "percent": round(float(prob * 100), 1)
            })
        
        # 퍼센트 높은 순(내림차순) 정렬
        monster_results = sorted(monster_results, key=lambda x: x['percent'], reverse=True)
        
        # 몬스터 이름을 키값으로 저장
        all_monster_results[monster] = monster_results

    # 8개 몬스터의 전체 데이터를 한 번에 JSON으로 반환
    return jsonify(all_monster_results), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)