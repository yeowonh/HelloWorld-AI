from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
from pymongo import MongoClient
import os
from model import ChatModel
import json

CONFIG_NAME = "chat_config.json"
print("## config_name : ", CONFIG_NAME)

with open(f'configs/{CONFIG_NAME}', 'r') as f:
    config = json.load(f)

app = Flask(__name__)
CORS(app)
run_with_ngrok(app)

# MongoDB 연결 설정
try:
    client = MongoClient(os.environ.get('MONGO_URI', 'mongodb://localhost:27017/'))
    db = client['chatdb']
    collection = db['translate_log']

except Exception as e:
    raise Exception(
        "The following error occurred: ", e)

chat_model = ChatModel()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/echo_call/<param>', methods=['GET'])
def get_echo_call(param):
    return jsonify({"param": param})

@app.route('/question', methods=['POST'])
def question():
    # 사용자 쿼리 받기
    query = request.get_json()

    # MongoDB에서 데이터 가져오기 (이전 대화 n개)
    # time 기준으로 정렬
    prev_conversation_data = list(collection.find().sort("time", -1).limit(config['config']['prev_turns']))

    if not prev_conversation_data:
        return jsonify({"error": "No conversation data found"})
    
    # MongoDB에서 가져온 데이터를 JSON 형태로 변환
    prev_conversation_data = "\n".join(prev_conversation_data)
    
    # 대화를 모델에 넣어 답변 생성
    answer = chat_model.get_answer(query["text"], prev_conversation_data)
    
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
