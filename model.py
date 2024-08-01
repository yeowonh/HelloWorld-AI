import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings

# dotenv setting
from dotenv import load_dotenv
load_dotenv(verbose=True)
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")

CONFIG_NAME = "chat_config.json"
print("## config_name : ", CONFIG_NAME)

with open(f'configs/{CONFIG_NAME}', 'r') as f:
    config = json.load(f)

# socket 설정 : 모델 다운로드 중 연결 끊김 방지
import socket
socket.setdefaulttimeout(500)


class ChatModel:
  def __init__(self):
    self.initialize()
    
  
  def initialize(self):
    self.tokenizer = AutoTokenizer.from_pretrained(config['config']['model_id'])
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.terminators = [
        self.tokenizer.eos_token_id,
        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    def load_db():
        torch.cuda.empty_cache()
        return ElasticsearchStore(
            es_cloud_id=ES_CLOUD_ID,
            es_user=ES_USER,
            es_password=ES_PASSWORD,
            es_api_key=ES_API_KEY,
            index_name=config['path']['db_name'],
            embedding = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
        )
    
    def load_model():
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(config['config']['model_id'], 
                                                     device_map=config['device'],
                                                     low_cpu_mem_usage=True) 
        model.eval()
        return model
    
    print('## Loading DB... ##')
    self.db = load_db()
    
    print('## Loading Model... ##')
    self.chatbot_model = load_model()
    print(f'## We will retrieve top-{config['config']['top_k']} relevant documents!')

  
  def get_answer(self, query:str, prev_turn: list) -> str:
    similar_docs = self.db.similarity_search(query)
    informed_context= ' '.join([x.page_content for x in similar_docs[:config['config']['top_k']]])

    PROMPT = f"""당신은 유능한 AI 어시스턴트입니다. [관련 문서]를 참조하여, [대화]에 적절한 답변을 생성해주세요.\n\n[관련 문서]: {informed_context}"""
    QUERY_PROMPT = f"""[대화]: {prev_turn}\n{query}"""
    message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": QUERY_PROMPT},
    ]

    source = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
    )

    with torch.no_grad():
        outputs = self.chatbot_model.generate(
                source.to(config['device']),
                max_new_tokens=config['inference']['max_new_tokens'],
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=config['inference']['do_sample'],
                num_beams=config['inference']['num_beams'],
                temperature=config['inference']['temperature'],
                top_k=config['inference']['top_k'],
                top_p=config['inference']['top_p'],
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
            )
    
    inference = self.tokenizer.decode(outputs[0][source.shape[-1]:], skip_special_tokens=True)

    return inference
