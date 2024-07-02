import openai
import json
import sys
import os
from utils import get_response, parsing_llm_fact_checking_output, compute_faithfulness_percentage_score
from utils import get_fact_checking_prompt

# api key
_api_key = #'your openai api key'
_client = openai.OpenAI(api_key=_api_key)
#_model = "gpt-3.5-turbo"
#_model = "gpt-4-1106-preview"
_model = "gpt-4o-2024-05-13"

def main(input_path, output_path, print_interval=2):
    '''
    Argument:
        input_path: path for input data
        output_path: path for output data (saving the logs and the eval results)
        print_interval: print the percentage scores every 'print_interval' 
    '''

    # loads data for faithfulness evaluation using FineSurE
    inputs = []
    for line in open(input_path, 'r'):
        line = json.loads(line)
        inputs.append(line)

    # variables for evaluation 
    cnt_total_inference = 0
    cnt_success_inference = 0
    model_labels = {}
    
    # writer to store the output from LLM evaluation
    raw_data_writer = open(os.path.join(output_path, 'raw-data.json'), 'w')
    result_writer = open(os.path.join(output_path, 'result.json'), 'w')

    # processes each data instance using for loop
    for input_id, input_json in enumerate(inputs):

        # input json parsing
        doc_id = input_json['doc_id']
        model_name = input_json['model']
        src = input_json['transcript']
        sentences = input_json['sentences']

        # prompt generation
        prompt = get_fact_checking_prompt(input=src, sentences=sentences)

        # get response from LLMs
        output = get_response(client=_client, prompt=prompt, model=_model)
 
        input_json['llm_output'] = output
        input_json['pred_faithfulness_labels'], input_json['pred_faithfulness_error_type'] = parsing_llm_fact_checking_output(output)

        # check if the parsing is success
        success_flag = True
        if len(input_json['pred_faithfulness_labels']) == 0:
            success_flag = False

        print("\nInput ID:", doc_id, "Model Name:", model_name)
        print("Success:", success_flag)
        print('\t[Error Label]:', input_json['pred_faithfulness_labels'])
        print('\t[Error Type]:', input_json['pred_faithfulness_error_type'])

        # count the success cases
        cnt_total_inference += 1
        if success_flag:
            cnt_success_inference += 1
        else:
            # fail to evalaute -> skip
            continue

        # compute the percentage score for faithfulness
        faithfulness_score = compute_faithfulness_percentage_score(input_json['pred_faithfulness_labels'])
        print('\t[Faithfulness Score]:', '{:.1%}'.format(faithfulness_score))

        # put the score into the aggregation dictionary
        if model_name not in model_labels:
            model_labels[model_name] = {'faithfulness_scores': [], 'binary_labels': []}
        model_labels[model_name]['faithfulness_scores'].append(faithfulness_score)
        model_labels[model_name]['binary_labels'].extend(input_json['pred_faithfulness_labels'])

        def print_results_faithfulness(model_labels):
            sentence_level_errors = {}
            summary_level_scores = {}
            for model_name, error_labels in model_labels.items():
                sentence_level_errors[model_name] = sum(error_labels['binary_labels']) / len(error_labels['binary_labels'])
                summary_level_scores[model_name] = sum(error_labels['faithfulness_scores']) / len(error_labels['faithfulness_scores'])

            text_output = "\n\n\n[Evaluation Results]\n"
            text_output += '* sentence-level factuality error ratio per model (lower is better)\n'
            for model_name, error_rate in sentence_level_errors.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(error_rate)) + '\n'

            text_output += '\n* summary-level faithfulness score per model (higher is better)\n'
            for model_name, score in summary_level_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* system-level model ranking (left is better)\n'
            sorted_dict = dict(sorted(summary_level_scores.items(), key=lambda item: item[1], reverse=True))
            model_ranking = list(sorted_dict.keys())
            text_output += str(model_ranking) + '\n'

            success_ratio = '{:.1%}'.format(cnt_success_inference/float(cnt_total_inference))
            text_output += '\n* success rate: ' + str(success_ratio) + '\n\n\n'

            print(text_output)
            return text_output

        # print percentage score
        if cnt_total_inference % print_interval == 0:
            print_results_faithfulness(model_labels=model_labels)
           
        json.dump(input_json, raw_data_writer)
        raw_data_writer.write('\n')
        raw_data_writer.flush()
    raw_data_writer.close()

    # print final results
    text_output = print_results_faithfulness(model_labels=model_labels)
    result_writer.write(text_output)
           

if __name__ == "__main__":
    
    '''
    Runnining Command:
        1) cd CodeRelease
        
        2) python finesure/fact-checking.py [input-path] [output-folder]
        e.g., python finesure/fact-checking.py dataset/frank/frank-data-sample-10.json result/fact-checking
    '''

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # print logs every 10 inferences
    print_interval = 10

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    main(input_path, output_path, print_interval)

