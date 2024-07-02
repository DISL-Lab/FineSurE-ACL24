# FineSurE: Fine-grained Summarization Evaluation using LLMs (ACL'23-main, Long Paper)

The structure of the projects:
- dataset: FRANK and REALSumm (in JSON format) are located in this folder
- reproduce: the code to reproduce the results of FineSurE in Table 1 and Table 2 
- finesure: the code to run our FineSurE method to evaluate the summary generated from language models

### Reproduce the Results

Runnining Command:

1) cd CodeRelease/reproduce

2) python reproduce-main-results.py results/frank-result-by-gpt4-w-finesure.json results/realsumm-result-by-gpt4-w-finesure.json


### Running FineSure on model summareis

We create sample datasets with 10 examples for fact-checking and keyfact-alignment tasks, respectively.

Please replace the <openai api key> with your api key.

Runnining Command:
1) cd CodeRelease

2) python finesure/fact-checking.py [input-path] [output-folder]

e.g., python finesure/fact-checking.py dataset/frank/frank-data-sample-10.json result/fact-checking


Runnining Command:
1) cd CodeRelease

2) python finesure/keyfact-alignment.py [input-path] [keyfact-path] [output-folder]

e.g., python finesure/keyfact-alignment.py dataset/realsumm/realsumm-data-sample-10.json dataset/realsumm/human-keyfact-list.json result/keyfact-alignment
