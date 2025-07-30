
# Debiasing Watermarks for Large Language Models via Maximal Coupling

This repository reproduces the results reported in [Debiasing Watermarks for Large Language Models via Maximal Coupling](https://arxiv.org/abs/2411.11203). The following content guides you through the process of reproducing the results in the paper using Python.


# Table of Contents
- [Reproducing Results in Python](#reproducing-results-in-the-paper)
  - [Requirements](#requirements)
  - [Models](#models)
  - [Data](#data)
  - [Usage](#usage)


# Reproducing Results in the Paper

## Requirements
First, install [PyTorch](https://pytorch.org/get-started/locally/). The remaining dependency can be installed using the following command:
```
python3 -m venv myenv
myenv\Scripts\activate # or for Linux: source myenv/bin/activate
pip install -r requirements.txt
```

All experiments were run on NVIDIA A100-SXM4 GPUs with 40GB of VRAM on [Polaris Compute Nodes](https://docs.alcf.anl.gov/polaris/running-jobs/using-gpus/) at Argonne National Laboratory. The code is designed to be run on a single GPU, but it can be adapted for multi-GPU setups if needed.

## Models

We use two instruction fine-tuned models: [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) and [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). In all our codes, `{model_name}` are either `phi` or `qwen`.

## Data
Due to the liscensing restrictions, we cannot provide the original data files in this repository, part of which are also included in our partial results. The original data and the processing scripts are available in the [data](data\readme.md) folder. The processed data is stored in JSONL format, which is a common format for storing structured data.

Specifically, we omitted `results.jsonl` under the `results` folder, which contains the generated watermarked text for each prompt in JSONL format. You can generate this file by running the [`polish.py`](./polish.py) script below with the appropriate command-line arguments.

## Usage
You can reproduce the results of the paper by the following steps:
1. Run the [`polish.py`](./polish.py) file with the appropriate command-line arguments.
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for polish.py</span></summary>

- `--model_name`: The name of the pre-trained model to use for text generation and analysis. Supported model names include "phi" and "qwen".
- `--prompt_path`: The path to the JSON file containing prompts. This file should contain a list of prompts in JSONL format. You can choose from `data/ETS_corpus_sampled.jsonl`, or `data/LOCNESS_sampled.jsonl`.
- `--improve_id`: The ID of the prompt to improve. This is used to select a specific prompt from the [`prompts.json`](/prompts.json) file. The range of `improve_id` is from 1 to 7. 
- `--method`: Choose a watermarking method for text generation. phiions: "none" (no watermarking), "openai" (Aaronson et al.), "maryland" ([Kirchenbauer et al.](https://arxiv.org/abs/2301.10226)). Default value: "none."
- `--temperature`: The temperature for text generation. This controls the randomness of the generated text. Default value: 1.0.
- `--ngram`: The size of the n-gram context for watermarking. This is used to determine the context width for the watermarking method. Default value: 4.
- `--output_dir`: The directory where the generated watermarked text and analysis results will be saved. 
- `--batch_size`: The number of prompts to process in a single batch. Default value: 64.

</details>

Here is an example:
```cmd
python polish.py \
        --model_name phi \
        --prompt_path data/ETS_corpus_sampled.jsonl \
        --improve_id 1 \
        --method openai \
        --temperature 0.7 \
        --ngram 4 \
        --output_dir output/ETS_corpus_sampled/phi/openai/temp0.7_ngram_4 \
        --batch_size 32
```

This script will produce `results.jsonl` in the specified output directory, which contains the generated watermarked text (AI-edited essays) for each prompt in JSONL format. 

2. Run the [`get_scores.py`](./get_scores.py) file to analyze the generated watermarked text and compute the watermark scores.
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for get_scores.py</span></summary>

- `--tokenizer`: The tokenizer to use for text analysis. Supported values are "phi" and "qwen".
- `--method_detect`: The watermark detection method to use. Supported values are "openai" (Aaronson et al.) and "maryland" (Kirchenbauer et al.). Default value: "openai".
- `--ngram`: The size of the n-gram context for watermarking. This is used to determine the context width for the watermarking method. Default value: 4.
- `--scoring_method`: The scoring method to use for watermark detection. Supported values are "none" (score every token), "v1" (score token when watermark context is unique), and "v2" (score token when {watermark context + token} is unique). Default value: "v2".
- `--input_dir`: The directory containing the input JSONL file with the generated watermarked text. 
- `--input_filename`: The name of the input JSONL file with the generated watermarked text. The default value is `results.jsonl`.
- `--input_key`: The key in the JSON file that contains the generated text. The default value is `result`.
- `--output_dir`: The directory where the analysis results will be saved. The default value is `same`, which means the output will be saved in the same directory as the input file.
- `--output_filename`: The name of the output JSONL file where the analysis results will be saved. The default value is `scores.jsonl`.

</details>
Here is an example:
```cmd
python get_scores.py \
        --tokenizer phi \
        --method_detect openai \
        --ngram 4 \
        --scoring_method v2 \
        --input_dir output/ETS_corpus_sampled/phi/openai/temp0.7_ngram_4/ \
        --input_filename results.jsonl \
        --input_key result \
        --output_dir output/ETS_corpus_sampled/phi/openai/temp0.7_ngram_4/ \
        --output_filename scores.jsonl
```

The script will produce `scores.jsonl` in the specified output directory. In particular, the `scores.jsonl` file contains the following fields:

| Field | Description |
| --- | --- |
| `text_index` | Index of the prompt in the JSON file |
| `num_token` | Number of analyzed tokens in the text |
| `score` | Watermark score of the text |
| `pvalue` | p-value of the detection test |

The [`similarity.py`](./similarity.py) script can be used to compute the similarity between the generated AI-editted text and the original essay. The output will be saved in `similarity.jsonl` in the specified output directory.
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for similarity.py</span></summary>

- `--input_dir`: The directory containing the input JSONL file with the original text. The default value is `data/`.
- `--input_filename`: The name of the input JSONL file with the original text. The default value is `ETS_corpus_sampled.jsonl`.
- `--output_dir`: The directory where the similarity results will be saved. The default value is `results/ETS_corpus_sampled/qwen/openai/temp0.7_ngram4/1`.

</details>

Here is an example:
```cmd
python similarity.py \
        --input_dir data/ \
        --input_filename ETS_corpus_sampled.jsonl \
        --output_dir results/ETS_corpus_sampled/qwen/openai/temp0.7_ngram4/1
```

3. You can reproduce the results of the paper by running the [`detection_results.py`](./detection_results.py) and use [`Summarize_results.ipynb`](./Summarize_results.ipynb) to visualize the results.

The [`detection_results.py`](./detection_results.py) script runs hypothesis testing via conformal prediction to analyze watermark detection robustness across different scenarios and datasets. This script supports three different experiments:

* `1`: ETS Corpus
* `2`: LOCNESS Corpus
* `3`: Weighted combination of ETS and LOCNESS

<details><span style="font-weight: bold;">Command Line Arguments for detection\_results.py</span>

* `--tokenizer`: The tokenizer/model to use. Supported values: `phi`, `qwen`, `llama`.
* `--method_detect`: Watermark detection method. Supported values: `openai`, `maryland`.
* `--ngram`: Size of the n-gram context used in watermark detection (default: `4`).
* `--experiment`: Selects which experiment to run:

  * `1` = ETS only
  * `2` = LOCNESS only (grouped)
  * `3` = Weighted mixture of ETS and LOCNESS (default)
</details>

Here is an example:
```bash
python detection_results.py \
        --tokenizer phi \
        --method_detect openai \
        --ngram 4 \
        --experiment 3
```

This will evaluate watermark detection robustness on the **weighted ETS + LOCNESS** setup using the **Phi tokenizer**, **OpenAI watermarking method**, and **n-gram size of 4**.

#### Output

The script will produce CSV result files under the `results/` directory based on the selected experiment:

* `results/ETS_conformal_phi_openai.csv` — for ETS-only experiment (`--experiment 1`)
* `results/LOCNESS_conformal_phi_openai.csv` — for LOCNESS-only (`--experiment 2`)
* `results/ETS_conformal_weighted_phi_openai.csv` — for weighted testing (`--experiment 3`)

Each output file contains statistics such as:

* False positive rate
* Power
* Number of outliers
* Total samples
* Prompt identifiers
* Experimental configuration (e.g., base and alternative prompt IDs, seed, method)

These results can be visualized and further summarized using [`Summarize_results.ipynb`](./Summarize_results.ipynb).

4. To reproduce the visuals in the introduction of the paper and to produce an example of the three classroom settings, you can run [`detection.ipynb`](./detection.ipynb).