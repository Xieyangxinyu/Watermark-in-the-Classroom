import os
from utils import load_results, clean_results
import argparse
import json
from datetime import datetime

import tqdm
# Need to install: pip install nltk
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Download necessary NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Catch LookupError (includes DownloadError)
    print("NLTK 'punkt' tokenizer not found. Attempting to download...")
    try:
        nltk.download('punkt')
        print("NLTK 'punkt' tokenizer downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK 'punkt' tokenizer: {e}")
except Exception as e: # Catch any other unexpected errors during find
    print(f"An unexpected error occurred while checking for NLTK 'punkt' tokenizer: {e}")

def calculate_bleu(reference_text, candidate_text, weights=(0.5, 0.5)):
    """
    Calculates BLEU score. Treats reference_text as the reference and candidate_text as the candidate.
    Returns 0.0 if either text is empty or calculation fails.
    Adjusts weights if candidate is shorter than the max n-gram size in weights.
    """
    if not reference_text or not candidate_text:
        return 0.0

    reference_tokens = [nltk.word_tokenize(reference_text.lower())] # BLEU expects a list of references
    candidate_tokens = nltk.word_tokenize(candidate_text.lower())

    # Adjust weights for shorter sequences to avoid errors
    max_n = len(weights)
    if len(candidate_tokens) < max_n:
         # Use only weights for n-grams up to the length of the candidate
        valid_weights = [w for i, w in enumerate(weights) if len(candidate_tokens) >= i + 1]
        if not valid_weights:
            return 0.0 # Candidate is too short for any n-gram size
        # Normalize weights if adjusted
        sum_weights = sum(valid_weights)
        weights = tuple(w / sum_weights for w in valid_weights)
        # Update max_n based on adjusted weights
        max_n = len(weights)

    # sentence_bleu can still raise errors for very short or empty inputs after tokenization
    if not candidate_tokens or any(len(ref) < max_n for ref in reference_tokens):
         return 0.0 # Ensure minimum length for BLEU calculation

    try:
        score = sentence_bleu(reference_tokens, candidate_tokens, weights=weights)
        return score
    except Exception as e:
        # Catch potential errors during BLEU calculation (e.g., due to very short texts)
        # print(f"Error calculating BLEU for texts: '{reference_text[:50]}...' vs '{candidate_text[:50]}...': {e}")
        return 0.0


def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--input_dir', type=str, default='data/',
                        help='Directory containing the input JSONL file.')
    parser.add_argument('--input_filename', type=str, default='ETS_corpus_sampled.jsonl',
                        help='Name of the input JSONL file.')
    parser.add_argument('--human_key', type=str, default='Human',
                        help='Key in the JSONL file for the human essay.')
    parser.add_argument('--result_key', type=str, default='result',
                        help='Key in the JSONL file for the comparison essay.')
    parser.add_argument('--output_dir', type=str, default='results/ETS_corpus_sampled/qwen/openai/temp0.7_ngram4/1',
                        help='Directory to save the output JSONL file. Use "same" to save in input_dir.')
    parser.add_argument('--output_filename', type=str, default='similarity.jsonl',
                        help='Name of the output JSONL file.')
    parser.add_argument('--nsamples', type=int, default=None,
                        help='Number of samples to process (default: all).')

    return parser

def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # Seed is not used in the provided snippet for any random operations
    # If needed for similarity calcs (e.g., sampling), add it.
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    if args.output_dir == 'same':
        args.output_dir = args.input_dir
    os.makedirs(args.output_dir, exist_ok=True)

    input_path = os.path.join(args.input_dir, args.input_filename)
    results_path = os.path.join(args.output_dir, "results.jsonl")
    output_path = os.path.join(args.output_dir, args.output_filename)

    # Load input essays (e.g., Human)
    print(f"Loading human essays from {input_path} with key '{args.human_key}'...")
    inputs_human = load_results(json_path=input_path, nsamples=args.nsamples, result_key=args.human_key)

    # Load comparison essays (e.g., Result)
    print(f"Loading comparison essays from {results_path} with key '{args.result_key}'...")
    comparison_texts = load_results(json_path=results_path, nsamples=args.nsamples, result_key=args.result_key)

    # Load existing results to resume
    print(f"Checking for existing results in {output_path}...")
    existing_results = load_results(json_path=output_path, nsamples=None, result_key=None) # Load full lines
    existing_indices = {res.get('text_index') for res in existing_results if isinstance(res, dict) and 'text_index' in res}
    start_point = len(existing_indices) # Count unique text_index values

    print(f"{start_point} results already computed in {output_path}")

    # Clean texts
    inputs_human = inputs_human
    comparison_texts = clean_results(comparison_texts)

    # Ensure we don't go out of bounds if files have different lengths
    max_samples = min(len(inputs_human), len(comparison_texts))

    if start_point >= max_samples:
        print("All required samples processed or input files are empty/mismatched.")
        return

    print(f"Starting processing from index {start_point} up to {max_samples}")


    with open(output_path, 'a') as f:
        # If starting from scratch, add a header or clear the file if desired (be cautious with 'w' mode)
        # If resuming, 'a' mode is correct.

        for ii in tqdm.tqdm(range(start_point, max_samples)):

            essay_human = inputs_human[ii]
            essay_comparison = comparison_texts[ii]

            # Calculate metrics
            # Calculate BLEU (Human as reference, Comparison as candidate)
            bleu_score = calculate_bleu(essay_human, essay_comparison, weights=(0.5, 0.5)) # Using 1-2 gram weights

            log_stat = {
                'text_index': ii,
                'bleu': bleu_score,
            }

            # Write results to the output file
            f.write(json.dumps(log_stat) + '\n')
            f.flush() # Ensure data is written to disk periodically

    print(f"Finished processing {max_samples} samples.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    print("\n\nStart time:", datetime.now().strftime("%m/%d %H:%M:%S"))
    main(args)
    print("Finish time:", datetime.now().strftime("%m/%d %H:%M:%S"), "\n\n")
