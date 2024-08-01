from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
from pymongo import MongoClient
import os
from model import ChatModel, SummaryModel
import json
import pymongo

CONFIG_NAME = "chat_config.json"
print("## config_name : ", CONFIG_NAME)

with open(f'configs/{CONFIG_NAME}', 'r') as f:
    config = json.load(f)

app = Flask(__name__)
CORS(app)
run_with_ngrok(app)

# MongoDB 연결 설정
try:
    # 아직 URI 설정 안된 상태
    client = MongoClient(os.environ.get('MONGO_URI', 'mongodb://localhost:27017/'))
    db = client['chatdb']
    collection = db['translate_log']

except Exception as e:
    raise Exception(
        "The following error occurred: ", e)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/get_test/<param>', methods=['GET'])
def get_echo_call(param):
    return jsonify({"param": param})

@app.route('/question', methods=['POST'])
def question():
    chat_model = ChatModel()
    # 사용자 쿼리 받기
    query = request.get_json()

    # MongoDB에서 데이터 가져오기 (이전 대화 n개)
    # time 기준으로 정렬
    prev_conversation_data = collection.find().sort({"time": -1}).limit(5)
    # 최신 순으로 정렬
    prev_conversation_data.reverse()
    
    prev_conversation_data = [{'sender' : x['sender'], 'content' : x['content']} for x in prev_conversation_data]
    if not prev_conversation_data:
        return jsonify({"error": "No conversation data found"})
    
    # 대화를 모델에 넣어 답변 생성
    answer = chat_model.get_answer(query["text"], prev_conversation_data)
    
    return jsonify({"answer": answer})


@app.route('/summary', methods=['POST'])
def summary():
    summary_model = SummaryModel()

    # MongoDB에서 데이터 가져오기 (이전 대화 n개)
    # time 기준으로 최신순 정렬
    prev_conversation_data = collection.find({}).sort({"time": 1})
    prev_conversation_data = [{'sender' : x['sender'], 'content' : x['content']} for x in prev_conversation_data]

    if not prev_conversation_data:
        return jsonify({"error": "No conversation data found"})
    
    
    # 대화를 모델에 넣어 요약문 생성
    summarization = summary_model.get_summary(prev_conversation_data)
    
    return jsonify({"summary": summarization})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
