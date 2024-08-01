import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

import argparse
from typing import Dict
import json
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from langchain_elasticsearch import ElasticsearchStore
import gradio as gr
from langchain_openai import OpenAIEmbeddings


load_dotenv(verbose=True)
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")

CONFIG_NAME = "chat_config.json"
print("## config_name : ", CONFIG_NAME)

def main(config: Dict):
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--model_id", type=str, default=config["config"]["model_id"], help="model id")
    g.add_argument("--chunk_size", type=int, default=config["config"]["chunk_size"], help="data chunk size")
    g.add_argument("--top_k", type=int, default=config["config"]["top_k"], help="How many documents are retrieved")
    g.add_argument("--db_name", type=str, default=config["path"]["db_name"], help="data index name to save")
    g.add_argument("--cache_dir", type=str, default="./cache", help="cache directory path")
    g.add_argument("--template_name", type=str, default=config["path"]["template_name"], help="What template to load")
    
    args = parser.parse_args()

    print("## Settings ##")
    print("## db_name : ", args.db_name)
    print("## top_k : ", args.top_k)
    print("## chunk_size : ", args.chunk_size)
    print("## template_name : ", args.template_name)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    def load_db():
        torch.cuda.empty_cache()
        return ElasticsearchStore(
            es_cloud_id=ES_CLOUD_ID,
            es_user=ES_USER,
            es_password=ES_PASSWORD,
            es_api_key=ES_API_KEY,
            index_name=args.db_name,
            # embedding=HuggingFaceEmbeddings(model_name=args.model_id,
            #                                 cache_folder=args.cache_dir)
            embedding = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
        )
    
    def load_model():
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                     cache_dir=args.cache_dir,
                                                     device_map=config['device'],
                                                     low_cpu_mem_usage=True) 
        model.eval()
        return model

    print('## Loading DB... ##')
    db = load_db()
    print('## Loading Model... ##')
    chatbot_model = load_model()

    ## how to ask a question
    def ask_a_question(question, top_k=args.top_k):
        similar_docs = db.similarity_search(question) # 임베딩한 벡터 간 코사인 유사도
        # bm25 사용하면 full-text 기반 검색 : Tf-idf 방식 활용 가능 
        print(f'## We retrieved top-{top_k} relevant documents!')
        for i in range(top_k):
            document = similar_docs[i].page_content
            print(f"- top{i+1} : {document}")
        
        retrieved_documents = f"""## We retrieved top-{top_k} relevant documents!\n## The most relevant passage with query:\n"""
        
        
        for i in range(top_k):
            retrieved_documents += f"top{i+1} : {similar_docs[i].page_content}\n"

        ## Ask Local LLM context informed prompt
        informed_context = ' '.join([x.page_content for x in similar_docs[:top_k]])
        # informed_response = chatbot_model.run(context=informed_context,question=question)
        PROMPT = f"""당신은 유능한 AI 어시스턴트입니다. [관련 문서]를 참조하여 [질문]에 대한 적절한 [답변]을 생성해주세요.\n\n[관련 문서]:{informed_context}"""

        message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": question},
        ]

        source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        outputs = chatbot_model.generate(
            source.to(config['device']),
            max_new_tokens=config['inference']['max_new_tokens'],
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=config['inference']['do_sample'],
            num_beams=config['inference']['num_beams'],
            temperature=config['inference']['temperature'],
            top_k=config['inference']['top_k'],
            top_p=config['inference']['top_p'],
            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
        )
        inference = tokenizer.decode(outputs[0][source.shape[-1]:], skip_special_tokens=True)

        return inference, retrieved_documents

    # conversational loop
    def inference(user_query, top_k=args.top_k):
        response, documents = ask_a_question(user_query, top_k=top_k)
        return response, documents

    gr.Interface(
        fn=inference,
        inputs=[
            gr.components.Textbox(lines=2, label="Input", placeholder="사장님이 월급을 세 달째 안줘요.."),
            gr.components.Slider(
                minimum=0, maximum=10, step=1, value=3, label="Top k"
            )
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            ),
            gr.components.Textbox(
                lines=5,
                label="Retrieved documents",
            )
        ],
        title="상담 챗봇 프로토타입 (한국어 only)",
        description="Hello! I am a QA chat bot for Foreigners, ask me any question about it. (Model : MLP-KTLim/llama-3-Korean-Bllossom-8B)",
    ).queue().launch(share=True, debug=True)

        

if __name__ == "__main__":
    with open(f'configs/{CONFIG_NAME}', 'r') as f:
        config = json.load(f)
    
    main(config=config)