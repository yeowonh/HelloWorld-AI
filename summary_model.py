
import argparse
import json
import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, os, os.path
import re
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict
from dotenv import load_dotenv

load_dotenv(verbose=True)
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")

def main(config: Dict):
    # fmt: off
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
    

    model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=args.device,
            low_cpu_mem_usage=True
    )
    model.eval()

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    ## mongoDB -> json 형태로 대화 기록 받아오기
    dataset = CustomDataset(os.path.join("resource/data/", f"일상대화요약_{filename}.json"), tokenizer)
    print(f"## Dataset length : {len(dataset)} ##")

    result = ""

    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]
        outputs = model.generate(
            inp.to(args.device).unsqueeze(0),
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            no_repeat_ngram_size=args.no_repeat_ngram_size
        )
        
        # 반환 형태? 일단 json
        result += tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)

    
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    selected_config = input('## input config path ##\n')
    try:
        with open(f'configs/{selected_config}', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"File not found: configs/{selected_config}")
    except json.decoder.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    main(config=config)