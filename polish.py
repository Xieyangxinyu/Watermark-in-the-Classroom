import os
from utils import load_prompts, load_model
import argparse
import time
import json
from datetime import datetime

import tqdm
import pandas as pd
import numpy as np

import torch

from wm import *

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str, default='qwen',
                        help='model to use: phi, qwen')

    # prompts parameters
    parser.add_argument('--prompt_path', type=str, default="data/ETS_corpus_sampled.jsonl")
    parser.add_argument('--improve_id', type=int, default=1)

    # generation parameters
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--max_gen_len', type=int, default=512)
    # watermark parameters
    parser.add_argument('--method', type=str, default='none', 
                        help='Choose among: none (no watermarking), openai (Aaronson et al.), maryland (Kirchenbauer et al.)')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.5, 
                        help='gamma for maryland/coupling: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=2.0, 
                        help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None, 
                        help='number of samples to generate, if None, take all prompts')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results',)

    return parser
    
def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # (re)start experiment
    os.makedirs(args.output_dir, exist_ok=True)
    start_point = 0 # if resuming, start from the last line of the file
    if os.path.exists(os.path.join(args.output_dir, f"results.jsonl")):
        with open(os.path.join(args.output_dir, f"results.jsonl"), "r") as f:
            for _ in f:
                start_point += 1
    print(f"Starting from {start_point}")

    # build model
    model, tokenizer, _ = load_model(args.model_name)

    for param in model.parameters():
        param.requires_grad = False

    # load prompts
    prompts = load_prompts(json_path=args.prompt_path, tokenizer=tokenizer, nsamples=args.nsamples, improve_id=args.improve_id)

    # build watermark generator
    if args.method == "none":
        generator = WmGenerator(model, tokenizer)
    elif args.method == "openai":
        generator = OpenaiGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method == "maryland":
        generator = MarylandGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)
    else:
        raise NotImplementedError("method {} not implemented".format(args.method))

    # generate
    all_times = []
    with open(os.path.join(args.output_dir, f"results.jsonl"), "a") as f:
        for ii in tqdm.tqdm(range(start_point, len(prompts), args.batch_size)):
            # generate chunk
            time0 = time.time()
            chunk_size = min(args.batch_size, len(prompts) - ii)
            results = generator.generate(
                prompts[ii:ii+chunk_size], 
                max_gen_len=args.max_gen_len, 
                temperature=args.temperature, 
                top_p=args.top_p
            )
            time1 = time.time()
            # time chunk
            speed = chunk_size / (time1 - time0)
            eta = (len(prompts) - ii) / speed
            eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta)) 
            all_times.append(time1 - time0)
            print(f"Generated {ii:5d} - {ii+chunk_size:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
            # log
            for j, (prompt, result) in tqdm.tqdm(enumerate(zip(prompts[ii:ii+chunk_size], results))):
                if ii + j > args.nsamples:
                    print(f"!!!!!Exceed n_samples!!!!!---{ii + j}--->{args.output_dir}")
                try:
                    f.write(json.dumps({
                        "text_index": ii + j,
                        "prompt": prompt, 
                        "result": result,
                        "speed": speed,
                        "eta": eta}) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"Error writing result for ii: {ii}: {e}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    # args.nsamples = 1000
    print("\n\nStart time:", datetime.now().strftime("%m/%d %H:%M:%S"))
    main(args)
    print("Finish time:", datetime.now().strftime("%m/%d %H:%M:%S"), "\n\n")
