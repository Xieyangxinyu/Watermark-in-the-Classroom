import os
os.environ['HF_HOME'] = './hf_home'
os.environ['TRANSFORMERS_CACHE'] = './hf_home/hub'

from typing import Dict, List
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def get_path(human: bool=False, data: str="ETS_corpus", model: str="llama", temp: float=1, ngram: int=4, improve_id: int=0, method: str="coupling", **kwargs) -> str:
    config = json.load(open("config.json", "r"))
    results_path = config["results_path"]
    if 'tokenizer' in kwargs.keys():
        model = kwargs['tokenizer']
    if human:
        return f"{results_path}/{data}/{model}/{method}/human" # use model tokenizer
    return f"{results_path}/{data}/{model}/{method}/temp{temp}_ngram{ngram}/{improve_id}"


def load_prompts(json_path: str, tokenizer, nsamples: int=None, improve_id: int=None) -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] # load jsonl
    new_prompts = prompts[:nsamples]
    print(f"Filtered {len(new_prompts)} prompts from {len(prompts)}")
    new_prompts = format_prompts(new_prompts, improve_id, tokenizer)
    return new_prompts

def load_results(json_path: str, nsamples: int=None, result_key: str=None, logging: bool=True) -> List[str]:
    if not os.path.exists(json_path):
        if logging:
            print(f"File not found: {json_path}")
        return []
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] # load jsonl
    if result_key:
        new_prompts = [o[result_key] for o in prompts]
    else:
        new_prompts = prompts
    if nsamples is not None:
        new_prompts = new_prompts[:nsamples]
    return new_prompts

def clean_text(text: str) -> str:
    # only clean the text if it starts with "Here is"
    if not text.startswith("Here is"):
        return text
    # Remove any "Here is ..." style leading sentence before the essay
    text = re.sub(r'^.*?Here is.*?\n\n', '', text, flags=re.DOTALL)

    # Trim leading and trailing whitespace, then add a single trailing newline
    text = text.lstrip('\n')
    text = re.sub(r'\s+$', '', text)

    return text + "\n"

def clean_results(results: List[str]) -> List[str]:
    return [clean_text(result) for result in results]

def format_prompts(prompts: List[Dict], improve_id: int, tokenizer) -> List[str]:
    def user_prompt(example, improve_id):
        if "Topic" in example.keys():
            topic = example["Topic"]
        else:
            topic = None
        if "Human" in example.keys():
            human_response = example["Human"]
        else:
            human_response = example["human_orig"]
        return {"role": "user", "content": get_prompt(topic, human_response, improve_id)}
    
    try:
        prompts = [
            tokenizer.apply_chat_template(
                [user_prompt(example, improve_id)],
                tokenize=False,
                add_generation_prompt=True
            )
            for example in prompts
        ]
    except:
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    user_prompt(example, improve_id),
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for example in prompts
        ]
    return prompts


import json
import random

def get_prompt(topic, human_response, improve_id, prompt_file='prompts.json'):
    with open(prompt_file, 'r') as f:
        templates = json.load(f)

    entry = templates[str(improve_id)]

    # Handle truncation
    if entry.get("truncate", False):
        # sentences can be split by "." or "!" or "?"
        blocks = re.split(r'\n+', human_response)
        sentences = []
        
        # For each block, split using punctuation and preserve it
        for block in blocks:
            parts = re.split(r'([.!?])', block)
            combined = [a.strip() + b for a, b in zip(parts[0::2], parts[1::2]) if a.strip()]
            sentences.extend(combined)
        
        if len(sentences) >= 5:
            # keep the first 3 sentences and sample 2 more
            sentences = sentences[:3] + random.sample(sentences[3:], min(2, len(sentences) - 3))
            human_response = " ".join(sentences)

    if topic is not None and entry.get("template_with_topic", False):
        template = entry["template_with_topic"]
        return template.format(topic=topic, human_response=human_response)
    else:
        template = entry["template"]
        return template.format(human_response=human_response)

def load_tokenizer(model_name):
    if model_name == 'phi':
        model_id = "microsoft/Phi-4-mini-instruct"
        vocab_size = 200064
    elif model_name == 'qwen':
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        vocab_size = 152064
    else:
        raise NotImplementedError(f"{model_name} not implemented")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer, vocab_size

def load_model(model_name):
    tokenizer, vocab_size = load_tokenizer(model_name)
    if model_name == 'phi':
        model_id = "microsoft/Phi-4-mini-instruct"
        model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2").eval()
        model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    elif model_name == 'qwen':
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2").eval()
    else:
        raise NotImplementedError(f"{model_name} not implemented")
    return model, tokenizer, vocab_size
