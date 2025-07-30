import json
from collections import Counter
import pandas as pd
import re
import random

def get_counts(counts):
    # Create a DataFrame for better visualization
    df = pd.DataFrame(counts.items(), columns=['Category', 'Count'])
    df['Proportion'] = df['Count'] / df['Count'].sum()
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    print(df)

# def truncate_by_paragraph(text, max_words=400):
#     """
#     Splits text into paragraphs, then drops paragraphs from the end
#     until word‐count ≤ max_words.  If a single paragraph is still
#     too long, falls back to a hard word‐level truncate.
#     """
#     # split on blank lines (one or more)
#     paras = re.split(r'\n\s*\n', text.strip())
    
#     # drop ending paragraphs until under limit
#     while paras:
#         truncated = "\n".join(paras)
#         words = truncated.split()
#         if len(words) <= max_words:
#             # print(f"Truncated to {len(words)} words")
#             return truncated
#         paras.pop()  # remove last paragraph
    
#     # in case all paragraphs dropped (shouldn't happen), do truncation by sentences
#     sentences = re.split(r'(?<=[.!?]) +', text.strip())
#     while sentences:
#         truncated = " ".join(sentences)
#         words = truncated.split()
#         if len(words) <= max_words:
#             return truncated
#         sentences.pop()  # remove last sentence
#     # if all else fails, return empty string
#     return " "

# # Load the data
# with open('LOCNESS.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # Convert to DataFrame for easier sampling
# df = pd.DataFrame(data)

# df['Human'] = df['Human'].apply(lambda x: truncate_by_paragraph(x, max_words=400) if len(x.split()) > 400 else x)
# df['num_words'] = df['Human'].str.split().str.len()
# df = df[df['num_words'] >= 250]
# print(f"Number of essays after filtering: {len(df)}")

# # remove the last .* from the essay_id
# df['essay_id'] = df['essay_id'].str.replace(r'\.[^\.>]+(?=>)', '', regex=True)

# df.to_json('LOCNESS_sampled.jsonl', orient='records', lines=True, force_ascii=False)

with open('LOCNESS_sampled.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]

# Convert to DataFrame for easier sampling
df = pd.DataFrame(data)

# Compute summary statistics
category = Counter(df['essay_id'])
print("Topic Distribution:")
get_counts(category)

# summarize the distribution of the length of the essays
print(f"Essay Length Distribution: {df['num_words'].describe()}")