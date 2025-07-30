import json
from collections import Counter
import pandas as pd

def get_counts(counts):
    # Create a DataFrame for better visualization
    df = pd.DataFrame(counts.items(), columns=['Category', 'Count'])
    df['Proportion'] = df['Count'] / df['Count'].sum()
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    print(df)

try:
    # Load the data
    with open('ETS_corpus.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame for easier sampling
    df = pd.DataFrame(data)

    # remove data with less than 250 words
    # compute the number of words in the essay
    df['num_words'] = df['Human'].str.split().str.len()
    # filter out essays with less than 250 words
    df = df[df['num_words'] >= 250]
    # truncate essays to a maximum of 400 words
    df = df[df['num_words'] <= 400]

    # save sampled data into original json file
    df.to_json('ETS_corpus_sampled.jsonl', orient='records', lines=True, force_ascii=False)
except FileNotFoundError:
    df = pd.read_json('ETS_corpus_sampled.jsonl', lines=True)

print(f"Total number of sampled entries: {len(df)}")

# check if there is duplicate Filename
if df['Filename'].unique().size != df['Filename'].size:
    print("Warning: There are duplicate filenames in the dataset.")

# Compute summary statistics
language_counts = Counter(df['Language'])
print("\nLanguage Distribution:")
get_counts(language_counts)

score_counts = Counter(df['Score Level'])
print("\nScore Level Distribution:")
get_counts(score_counts)

prompt_counts = Counter(df['Prompt'])
print("\nPrompt Distribution:")
get_counts(prompt_counts)

# summarize the distribution of the length of the essays
print(f"Essay Length Distribution: {df['num_words'].describe()}")