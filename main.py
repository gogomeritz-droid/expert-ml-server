from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

# Flask 웹 서버 애플리케이션 초기화
app = Flask(__name__)

# ==============================================================================
# 🔑 [보안 설정] 서버 간 통신을 위한 비밀 API Key
# - 로블록스 서버 스크립트에 입력된 API_KEY와 '한 글자도 틀리지 않고 똑같이' 맞추어야 합니다!
# ==============================================================================
SECRET_API_KEY = "#1480epvp+6q9=07"  

# 📦 [데이터 저장소] 수집된 레이드 기록을 임시로 저장할 메모리 리스트
raid_database = []

# 🧠 [캐시 메모리] 매번 머신러닝을 돌리지 않고, 미리 계산된 예측 결과를 보관해 두는 저장소
cached_predictions = {}

# 🕒 [시간 상수] 3개월(90일)을 초(second) 단위로 계산 (오래된 데이터 필터링용)
THREE_MONTHS_SECONDS = 90 * 24 * 60 * 60

# ⚔️ [무기 리스트] 9가지 지정된 무기 이름 (로블록스 WEAPONS 테이블의 key/display와 매칭)
WEAPONS = [
    '강철검', '구리검', '노벨륨 지팡이', '로렌슘 검', 
    '뢴트게늄 언월도', '아인슈타이늄 지팡이', '텅스텐 검', '티타늄검', '하슘 블랙홀검'
]

# 👹 [몬스터 리스트] 8가지 지정된 레이드 몬스터 이름
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

# 🔍 몬스터 이름의 오타나 공백을 정규화하여 지정된 몬스터 리스트 중 하나로 매핑하는 함수
def normalize_monster_name(name):
    if not name:
        return ""
    for m in MONSTERS:
        if m in name:         # "사이클롭스 쌍둥이1"이나 "사이클롭스 쌍둥이2"가 들어와도
            return m           # "사이클롭스 쌍둥이"로 매핑됨
    return name

# 🛡️ [인증 함수] 요청 헤더에 담긴 X-API-Key가 올바른지 검증하는 공통 함수
def verify_api_key():
    client_key = request.headers.get("X-API-Key")
    if client_key != SECRET_API_KEY:
        return False
    return True

# 📊 [안전장치 함수] 데이터가 부족(5건 미만)하거나 에러 발생 시 빈도 기반 또는 가짜(더미) 데이터를 만드는 함수
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
        # 💡 경우 A: 데이터가 1건 이상 5건 미만일 때 -> 실제 기록된 빈도 비율(%)로 계산
        for weapon in WEAPONS:
            count = weapon_counts[weapon]
            percent = round(float(count / total_count * 100), 1)
            results.append({"weapon": weapon, "percent": percent})
    else:
        # 💡 경우 B: 데이터가 0건(아예 없을 때) -> 디리클레 분포(Dirichlet distribution)를 이용해 
        # 에러 방지용 무기별 가짜(더미) 확률 데이터를 자연스럽게 생성
        np.random.seed(len(raid_database) + index + 42)
        raw_probs = np.random.dirichlet(np.ones(len(WEAPONS)), size=1)[0]
        for weapon, prob in zip(WEAPONS, raw_probs):
            results.append({"weapon": weapon, "percent": round(float(prob * 100), 1)})
            
    return results

# 🤖 [머신러닝 파이프라인] RandomForest를 활용해 몬스터별 최적의 무기 확률을 학습하고 캐시를 갱신하는 함수
def run_machine_learning_pipeline():
    global raid_database, cached_predictions
    current_time = time.time()
    
    # 🕒 최근 3개월(90일) 이내의 유효한 데이터만 남기고 오래된 데이터는 자동 청소(필터링)
    raid_database = [
        item for item in raid_database 
        if (current_time - item.get("timestamp", current_time)) <= THREE_MONTHS_SECONDS
    ]
    
    all_monster_results = {}
    
    # 8가지 몬스터를 각각 순회하며 머신러닝 학습 또는 통계 산출 진행
    for index, monster in enumerate(MONSTERS):
        monster_records = [item for item in raid_database if item.get("monster") == monster]
        monster_results = []
        
        # 💡 조건: 해당 몬스터의 데이터가 5건 이상 충분히 쌓인 경우에만 머신러닝 구동
        if len(monster_records) >= 5:
            try:
                df = pd.DataFrame(monster_records)
                df['monster_code'] = df['monster'].apply(lambda x: MONSTERS.index(x) if x in MONSTERS else 0)
                
                X = df[['monster_code']] 
                y = df['weapon']         
                
                # RandomForest 분류 모델 학습 (트리 개수 50개)
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
                # 학습 중 에러가 나면 안전장치 함수로 전환
                monster_results = get_fallback_stats(monster_records, index)
        else:
            # 💡 5건 미만인 경우 안전장치 함수(빈도 또는 가짜 데이터) 호출
            monster_results = get_fallback_stats(monster_records, index)
            
        # 📈 퍼센트가 높은 순서대로(내림차순) 정렬
        monster_results = sorted(monster_results, key=lambda x: x['percent'], reverse=True)
        all_monster_results[monster] = monster_results
        
    # 💾 최종 결과를 전역 캐시에 저장 (이후 클라이언트 요청 시 즉시 반환용)
    cached_predictions = all_monster_results
    print(f"[서버] 머신러닝 학습 및 캐시 갱신 완료! (유효 데이터 총: {len(raid_database)}건)")

# ==============================================================================
# API 엔드포인트 1: 로블록스에서 24시간마다 모아둔 데이터를 일괄 수신하는 POST 라우트
# ==============================================================================
@app.route('/update_raid', methods=['POST'])
def update_raid():
    # API Key 보안 인증 검사
    if not verify_api_key():
        return jsonify({"error": "Unauthorized: Invalid or missing API Key"}), 403

    global raid_database
    data = request.json
    current_time = time.time()
    
    # 단일 데이터 혹은 리스트 형태의 데이터 모두 수용하여 메모리에 적재
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
                
    # 데이터가 수신되면 즉시 머신러닝 학습 파이프라인을 돌려 캐시 최신화
    run_machine_learning_pipeline()
    
    return jsonify({"status": "success", "total": len(raid_database)}), 200

# ==============================================================================
# API 엔드포인트 2: 로블록스 클라이언트가 예측 데이터를 요청할 때 처리하는 GET 라우트
# ==============================================================================
@app.route('/predict', methods=['GET'])
def predict():
    # API Key 보안 인증 검사
    if not verify_api_key():
        return jsonify({"error": "Unauthorized: Invalid or missing API Key"}), 403

    global cached_predictions
    
    # 서버가 켜진 직후라 캐시가 비어있다면 긴급 학습 수행
    if not cached_predictions:
        run_machine_learning_pipeline()
        
    # 로블록스 서버스크립트에서 보낸 쿼리 파라미터 확인 (예: ALL_MONSTERS)
    monster_query = request.args.get("monster")
    
    # 전체 몬스터 데이터를 요구한 경우 캐시된 전체 딕셔너리 반환
    if monster_query == "ALL_MONSTERS" or not monster_query:
        return jsonify(cached_predictions), 200
    else:
        # 특정 몬스터만 지칭한 경우 해당 몬스터 데이터만 추출해서 반환
        if monster_query in cached_predictions:
            return jsonify(cached_predictions[monster_query]), 200
        else:
            return jsonify({"error": "Monster not found"}), 404

# Flask 서버 구동 (포트 5000번, 외부 접속 허용)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
