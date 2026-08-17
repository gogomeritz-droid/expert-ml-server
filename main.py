from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

app = Flask(__name__)

# 🔑 서버 간 보안을 위한 비밀 API Key 설정 (로블록스 서버와 동일하게 맞추어야 합니다)
SECRET_API_KEY = "#1480epvp+6q9=07"  # 원하는 안전한 문자열로 변경하세요

# 수집된 레이드 데이터를 저장할 메모리 리스트
raid_database = []

# 미리 계산된 머신러닝 예측 결과를 보관하는 캐시 메모리
cached_predictions = {}

# 🕒 3개월(90일) 초 단위 계산
THREE_MONTHS_SECONDS = 90 * 24 * 60 * 60

# 9가지 지정된 무기 리스트
WEAPONS = [
    '강철검', '구리검', '노벨륨 지팡이', '로렌슘 검', 
    '뢴트게늄 언월도', '아인슈타이늄 지팡이', '텅스텐 검', '티타늄검', '하슘 블랙홀검'
]

# 8가지 지정된 레이드 몬스터 리스트
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

def normalize_monster_name(name):
    if not name:
        return ""
    for m in MONSTERS:
        if m in name:
            return m
    return name

# 🛡️ 요청 헤더의 API Key를 검증하는 공통 함수
def verify_api_key():
    client_key = request.headers.get("X-API-Key")
    if client_key != SECRET_API_KEY:
        return False
    return True

# 데이터가 부족하거나 에러 시 사용하는 보조 함수 (빈도 기반 통계)
def get_fallback_stats(monster_records, index):
    weapon_counts = {w: 0 for w in WEAPONS}
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
    
    if total_count > 0:
        # 데이터가 1건 이상 5건 미만일 때는 실제 기록된 빈도 비율로 계산
        for weapon in WEAPONS:
            count = weapon_counts[weapon]
            percent = round(float(count / total_count * 100), 1)
            results.append({"weapon": weapon, "percent": percent})
    else:
        # ==============================================================================
        # [주석] 🛑 바로 이 구간이 데이터가 '0건(아예 없을 때)' 가짜(더미) 데이터를 만드는 코드입니다!
        # - 디리클레 분포(Dirichlet distribution)를 이용해 9개 무기의 확률을 합이 100%가 되도록
        #   무작위로 임의로 쪼개서 가짜 확률 데이터를 만들어 반환합니다.
        # ==============================================================================
        np.random.seed(len(raid_database) + index + 42)
        raw_probs = np.random.dirichlet(np.ones(len(WEAPONS)), size=1)[0]
        for weapon, prob in zip(WEAPONS, raw_probs):
            results.append({"weapon": weapon, "percent": round(float(prob * 100), 1)})
            
    return results

# 🧠 머신러닝 학습 및 예측을 수행하여 캐시를 갱신하는 내부 함수
def run_machine_learning_pipeline():
    global raid_database, cached_predictions
    current_time = time.time()
    
    # 3개월 지난 데이터 필터링
    raid_database = [
        item for item in raid_database 
        if (current_time - item.get("timestamp", current_time)) <= THREE_MONTHS_SECONDS
    ]
    
    all_monster_results = {}
    
    for index, monster in enumerate(MONSTERS):
        monster_records = [item for item in raid_database if item.get("monster") == monster]
        monster_results = []
        
        # 데이터가 충분히 쌓인 경우 (5건 이상) 머신러닝 학습 및 예측 구동
        if len(monster_records) >= 5:
            try:
                df = pd.DataFrame(monster_records)
                df['monster_code'] = df['monster'].apply(lambda x: MONSTERS.index(x) if x in MONSTERS else 0)
                
                X = df[['monster_code']] 
                y = df['weapon']         
                
                # RandomForest 모델 학습
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)
                
                classes = model.classes_
                test_X = pd.DataFrame([[index]], columns=['monster_code'])
                probs = model.predict_proba(test_X)[0]
                
                pred_dict = {cls: prob * 100 for cls, prob in zip(classes, probs)}
                
                for weapon in WEAPONS:
                    percent = round(float(pred_dict.get(weapon, 0.0)), 1)
                    monster_results.append({
                        "weapon": weapon,
                        "percent": percent
                    })
            except Exception as e:
                print(f"머신러닝 학습 중 예외 발생 ({monster}): {e}")
                monster_results = get_fallback_stats(monster_records, index)
        else:
            monster_results = get_fallback_stats(monster_records, index)
            
        # 퍼센트 높은 순(내림차순) 정렬
        monster_results = sorted(monster_results, key=lambda x: x['percent'], reverse=True)
        all_monster_results[monster] = monster_results
        
    # 최종 결과를 캐시에 저장 (이후 클라이언트 요청 시 즉시 반환됨)
    cached_predictions = all_monster_results
    print(f"[서버] 24시간 주기 머신러닝 학습 및 캐시 갱신 완료! (유효 데이터: {len(raid_database)}건)")

# 1. 24시간마다 누적된 데이터를 받아오고, 즉시 머신러닝 학습을 돌려 캐시를 갱신하는 엔드포인트
@app.route('/update_raid', methods=['POST'])
def update_raid():
    # 🔒 API Key 인증 검사
    if not verify_api_key():
        return jsonify({"error": "Unauthorized: Invalid or missing API Key"}), 403

    global raid_database
    data = request.json
    current_time = time.time()
    
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
                
    # 데이터 수신 즉시 머신러닝 학습 파이프라인 가동 (캐시 갱신)
    run_machine_learning_pipeline()
    
    return jsonify({"status": "success", "total": len(raid_database)}), 200

# 2. 클라이언트 요청 시, 머신러닝을 새로 돌리지 않고 미리 학습해 둔 캐시 데이터만 즉시 반환하는 엔드포인트
@app.route('/predict', methods=['GET'])
def predict():
    # 🔒 API Key 인증 검사
    if not verify_api_key():
        return jsonify({"error": "Unauthorized: Invalid or missing API Key"}), 403

    global cached_predictions
    
    # 만약 서버가 막 켜져서 아직 24시간 데이터가 안 들어와 캐시가 비어있다면 한 번 학습 수행
    if not cached_predictions:
        run_machine_learning_pipeline()
        
    return jsonify(cached_predictions), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
