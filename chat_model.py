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
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv(verbose=True)
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")

CONFIG_PATH = ""


def main(config: Dict):
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--model_id", type=str, default=config["config"]["model_id"], help="model id")
    g.add_argument("--chunk_size", type=int, default=config["config"]["chunk_size"], help="data chunk size")
    g.add_argument("--top_k", type=int, default=config["config"]["top_k"], help="How many documents are retrieved")
    g.add_argument("--data_path", type=str, default=config["path"]["data_path"], help="data path")
    g.add_argument("--db_name", type=str, default=config["path"]["db_name"], help="data index name to save")
    g.add_argument("--cache_dir", type=str, default="./cache", help="cache directory path")
    g.add_argument("--template_name", type=str, default=config["path"]["template_name"], help="What template to load")
    
    args = parser.parse_args()

    print("## Settings ##")
    print("## data_path : ", args.data_path)
    print("## db_name : ", args.db_name)
    print("## top_k : ", args.top_k)
    print("## chunk_size : ", args.chunk_size)
    print("## template_name : ", args.template_name)
    

    def load_db():
        return ElasticsearchStore(
            es_cloud_id=ES_CLOUD_ID,
            es_user=ES_USER,
            es_password=ES_PASSWORD,
            es_api_key=ES_API_KEY,
            index_name=args.db_name,
            embedding=HuggingFaceEmbeddings(model_name=args.model_id)
        )
    
    def load_model():
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(args.model_id) 
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, cache_dir=args.cache_dir) 
        pipe = pipeline(
                "text2text-generation",
                model=model, 
                tokenizer=tokenizer, 
                max_length=256,
                device=-1
            )
        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"## Get {args.model_id} ready to go ##")

        with open(f'templates/{args.template_name}.txt', 'r') as f:
            template = f
        
            template = """
            나는 외국인 노동자를 위한 챗봇.~~
            When I don't know the answer I say I don't know.
            I know context: {context}
            when asked: {question}
            my response using only information in the context is: 
            """

        prompt_informed = PromptTemplate(template=template, input_variables=["context", "question"])
        return LLMChain(prompt=prompt_informed, llm=llm)
    
    print('## Loading DB... ##')
    db = load_db()
    print('## Loading Model... ##')
    chatbot_model = load_model()

    print("## Conversation Start !! ##")
    print(f'## We will retrieve top-{args.top_k} relevant documents!')
    while True:
        query = input("User query >> ")
        print("user question : ", query)
        similar_docs = db.similarity_search(query)
        print(f"## The most relevant top-{args.top_k} passage:\n")
        for i in range(args.top_k):
            document = similar_docs[i].page_content
            print(f"- top{i+1} : {document}")
        
        ## Ask Local LLM context informed prompt
        informed_context= ' '.join([x.page_content for x in similar_docs[:args.top_k]])
        informed_response = chatbot_model.run(context=informed_context,question=query)

        print(f"\tAnswer  : {informed_response}")

        

if __name__ == "__main__":
    with open(f'configs/{CONFIG_PATH}', 'r') as f:
        config = json.load(f)
    
    main(config=config)