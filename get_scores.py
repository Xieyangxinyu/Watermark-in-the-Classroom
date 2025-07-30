import os
from utils import load_results, clean_results, load_tokenizer
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

    # tokenizer parameters
    parser.add_argument('--tokenizer', type=str, default='llama', 
                        help='tokenizer to use: llama, qwen, ')

    # watermark parameters
    parser.add_argument('--method_detect', type=str, default='none', 
                        help='Choose among: none (no watermarking), openai (Aaronson et al.), maryland (Kirchenbauer et al.)')
    
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.5, 
                        help='gamma for maryland/coupling: proportion of greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='v2', 
                        help='method for scoring. choose among: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None, 
                        help='number of samples to generate, if None, take all prompts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--input_filename', type=str, default='results.jsonl')
    parser.add_argument('--input_key', type=str, default='result')
    parser.add_argument('--output_dir', type=str, default='same')
    parser.add_argument('--output_filename', type=str, default='scores.jsonl')

    return parser
    
def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.output_dir == 'same':
        args.output_dir = args.input_dir
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    # check if the computation has done
    scores = load_results(json_path=output_path, nsamples=args.nsamples, result_key="score")
    start_point = len(scores)
    print(f"{start_point} scores already computed in {output_path}")
    if start_point >= args.nsamples:
        return
    print(f"Starting from {start_point}")
    
    os.makedirs(args.input_dir, exist_ok=True)
    
    tokenizer, vocab_size = load_tokenizer(args.tokenizer)
    if args.method_detect == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, vocab_size = vocab_size)
    elif args.method_detect == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, vocab_size = vocab_size)
    else:
        raise NotImplementedError(f"Watermarking method {args.method_detect} not implemented")
    

    # evaluate of watermark
    input_path = os.path.join(args.input_dir, args.input_filename)
    inputs = load_results(json_path=input_path, nsamples=args.nsamples, result_key=args.input_key)
    inputs = clean_results(inputs)
    log_stats = []

    with open(output_path, 'a') as f:
        for ii in tqdm.tqdm(range(start_point, len(inputs))):

            # if input[ii] begins with 'Here is', clean it up

            scores_raw = detector.get_scores_by_t([inputs[ii]], scoring_method=args.scoring_method)
            scores = detector.aggregate_scores(scores_raw)
            pvalues = detector.get_pvalues(scores_raw)

            scores = [float(s) for s in scores]
            num_tokens = [len(score_raw) for score_raw in scores_raw]
            # log stats and write
            
            log_stat = {
                'text_index': ii,
                'num_token': num_tokens[0],
                'score': scores[0],
                'pvalue': float(pvalues[0])
            }
            log_stats.append(log_stat)
            f.write(json.dumps(log_stat)+'\n')
    

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    # args.nsamples = 1200
    print("\n\nStart time:", datetime.now().strftime("%m/%d %H:%M:%S"))
    main(args)
    print("Finish time:", datetime.now().strftime("%m/%d %H:%M:%S"), "\n\n")
