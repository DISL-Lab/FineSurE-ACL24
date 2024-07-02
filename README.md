# FineSurE: Fine-grained Summarization Evaluation using LLMs (ACL'23-main, Long Paper)

The structure of the projects:
- dataset: FRANK and REALSumm (in JSON format) are located in this folder
- reproduce: the code to reproduce the results of FineSurE in Table 1 and Table 2 
- finesure: the code to run our FineSurE method to evaluate the summary generated from language models

## Highlight

**FineSurE** is a *multi-dimensional*, *fine-grained* automated evaluation framework for text summarization. It covers there distinctive evaluation dimensions, namely faithfulness, completeness, and conciseness. These dimensions are crucial to assess the capability of modern language models in summarization, as they are susceptible to incorrect statement, information omission, and verbosity.

FineSurE framework breaks down a complicate evaluation process into ```two simple human-like evaluation tasks``` using LLMs. 

- **Fact Checking:** This is a task of solving a categorization problem involving nine categories. These include the seven factuality errors, along with an additional category "other error" for error outside the seven errors, and an additional category "no error" for cases whether no error was detected. Given a pair of input text and model summary, the LLM is expected to output the error type classified into one of the nine categories for each sentence along with a concise reason.
  
- **Keyfact Alignment:** This is an alignment task of matching each keyfact into the summary sentences from which the keyfact is inferable. Given a pari of keyfact list and model summary, the output should be the binary label (whether inferable or not) an dth elist of line numbers of all summary sentences matched for each keyfact.

<p align="center">
<img width="755" alt="스크린샷 2024-07-02 오후 4 56 27" src="https://github.com/DISL-Lab/FineSurE-ACL24/assets/10972556/e5c733b7-d863-4e39-92ac-98f63e8bbee5">
</p>

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
