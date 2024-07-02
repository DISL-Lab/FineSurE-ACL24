import openai
import json
import sys
import os
from utils import get_response
from utils import get_keyfact_alighment_prompt, parsing_llm_keyfact_alighment_output
from utils import compute_completeness_percentage_score, compute_conciseness_percentage_score


# api key
_api_key = #'your openai api key'
_client = openai.OpenAI(api_key=_api_key)
#_model = "gpt-3.5-turbo"
#_model = "gpt-4-1106-preview"
_model = "gpt-4o-2024-05-13"

def main(input_path, keyfact_path, output_path, print_interval=2):
    '''
    Argument:
        input_path: path for input data
        keyfact_path: path for human or machine keyfacts
        output_path: path for output data (saving the logs and the eval results)
        print_interval: print the percentage scores every 'print_interval' 
    '''

    # loads data for completeness and conciseness evaluation using FineSurE
    inputs = []
    for line in open(input_path, 'r'):
        line = json.loads(line)
        inputs.append(line)

    # loads keyfacts 
    keyfacts = {}
    for line in open(keyfact_path, 'r'):
        line = json.loads(line)
        keyfacts[line['doc_id']] = line['key_facts']

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
        list_keyfacts = keyfacts[doc_id]

        # prompt generation
        prompt = get_keyfact_alighment_prompt(keyfacts=list_keyfacts, sentences=sentences)

        # get response from LLMs
        output = get_response(client=_client, prompt=prompt, model=_model)
 
        input_json['llm_output'] = output
        input_json['pred_alignment_labels'], input_json['pred_sentence_line_numbers'] = parsing_llm_keyfact_alighment_output(output)

        # check if the parsing is success
        success_flag = True
        if len(input_json['pred_alignment_labels']) == 0:
            success_flag = False

        print("\nInput ID:", doc_id, "Model Name:", model_name)
        print("Success:", success_flag)
        print('\t[Alignment Label]:', input_json['pred_alignment_labels'])
        print('\t[Matched Sentence Line Numbers]:', input_json['pred_sentence_line_numbers'])

        # count the success cases
        cnt_total_inference += 1
        if success_flag:
            cnt_success_inference += 1
        else:
            # fail to evalaute -> skip
            continue      

        # compute the percentage score for faithfulness
        completeness_score = compute_completeness_percentage_score(input_json['pred_alignment_labels'])
        conciseness_score = compute_conciseness_percentage_score(input_json['pred_sentence_line_numbers'], len(sentences))

        # put the score into the aggregation dictionary
        if model_name not in model_labels:
            model_labels[model_name] = {'completeness_scores': [], 'conciseness_scores': []}
        model_labels[model_name]['completeness_scores'].append(completeness_score)
        model_labels[model_name]['conciseness_scores'].append(conciseness_score)

        print('\t[Completeness Score]:', '{:.1%}'.format(completeness_score))
        print('\t[Conciseness Score]:', '{:.1%}'.format(conciseness_score))

        def print_results_faithfulness(model_labels):
            summary_level_completeness_scores = {}
            summary_level_conciseness_scores = {}

            for model_name, error_labels in model_labels.items():
                summary_level_completeness_scores[model_name] = sum(error_labels['completeness_scores']) / len(error_labels['completeness_scores'])
                summary_level_conciseness_scores[model_name] = sum(error_labels['conciseness_scores']) / len(error_labels['conciseness_scores'])

            text_output = "\n\n\n[Evaluation Results]\n"
            text_output += '\n* completeness score per model (higher is better)\n'
            for model_name, score in summary_level_completeness_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* completeness model ranking (left is better)\n'
            sorted_dict = dict(sorted(summary_level_completeness_scores.items(), key=lambda item: item[1], reverse=True))
            model_ranking = list(sorted_dict.keys())
            text_output += str(model_ranking) + '\n'

            text_output += '\n* conciseness score per model (higher is better)\n'
            for model_name, score in summary_level_conciseness_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* conciseness model ranking (left is better)\n'
            sorted_dict = dict(sorted(summary_level_conciseness_scores.items(), key=lambda item: item[1], reverse=True))
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
        
        2) python finesure/keyfact-alignment.py [input-path] [keyfact-path] [output-folder]
        e.g., python finesure/keyfact-alignment.py dataset/realsumm/realsumm-data-sample-10.json dataset/realsumm/human-keyfact-list.json result/keyfact-alignment
    '''

    input_path = sys.argv[1]
    keyfact_path = sys.argv[2]
    output_folder = sys.argv[3]

    # print logs every 10 inferences
    print_interval = 10

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    main(input_path, keyfact_path, output_folder, print_interval)
