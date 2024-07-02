# FineSurE: Fine-grained Summarization Evaluation using LLMs (ACL'23-main, Long Paper)

The structure of the projects:
- dataset: FRANK and REALSumm (in JSON format) are located in this folder
- reproduce: the code to reproduce the results of FineSurE in Table 1 and Table 2 
- finesure: the code to run our FineSurE method to evaluate the summary generated from language models

## Running FineSurE on Model Summareis

We create sample datasets with 10 examples for fact-checking and keyfact-alignment tasks, respectively.

Please replace the ```openai api key``` with your api key in ```finesure/fact-checking.py``` and ```finesure/keyfact-alignmnet.py```.

#### Runnining Command:
```bash
cd CodeRelease
python finesure/fact-checking.py [input-path] [output-folder]

# example code for fact checking on sampled data.
python finesure/fact-checking.py dataset/frank/frank-data-sample-10.json result/fact-checking
```

#### Runnining Command:
```bash
cd CodeRelease
python finesure/keyfact-alignment.py [input-path] [keyfact-path] [output-folder]

# example code for keyfact alignment on sampled data.
python finesure/keyfact-alignment.py dataset/realsumm/realsumm-data-sample-10.json dataset/realsumm/human-keyfact-list.json result/keyfact-alignment
```

## Reproduce the Main Table of the Paper

```bash
cd CodeRelease/reproduce
python reproduce-main-results.py results/frank-result-by-gpt4-w-finesure.json results/realsumm-result-by-gpt4-w-finesure.json
```


## Citation

Please consider citation if our paper is useful in your research.

```BibTeX
@inproceedings{song2024finesure,
  title={FineSurE: Fine-grained Summarization Evaluation using LLMs},
  author={Song, Hwanjun and Su, Hang and Shalyminov, Igor and Cai, Jason and Mansour, Saab},
  booktitle={ACL},
  year={2024}
}
```
