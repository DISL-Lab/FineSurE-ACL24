This is the source code for the paper titled "FineSurE: Fine-grained Summarization Evaluation using LLMs"

The structure of the projects:
- dataset: frank and realsumm (in JSON format) are located in this folder
- reproduce: the code to reproduce the results of FineSurE in Table 1 and Table 2 
- finesure: the code to run our FineSurE method to evaluate the summary generated from language models

### Running FineSure on model summareis

We create sample datasets with an example for fact-checking and keyfact-alignment tasks, respectively. The data is a synthetic toy example here, only providing the information on what format you use. 

Please replace the <openai api key> with your api key in finesure/fact-checking.py and finesure/keyfact-alignmnet.py.

The LLM backbone is set to be GPT-4o by default.

Runnining Command:
1) cd CodeRelease

2) python finesure/fact-checking.py [input-path] [output-folder]

e.g., python finesure/fact-checking.py dataset/frank/frank-data-sample-10.json result/fact-checking


Runnining Command:
1) cd CodeRelease

2) python finesure/keyfact-alignment.py [input-path] [keyfact-path] [output-folder]

e.g., python finesure/keyfact-alignment.py dataset/realsumm/realsumm-data-sample-10.json dataset/realsumm/human-keyfact-list.json result/keyfact-alignment
