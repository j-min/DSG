from pathlib import Path
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Optional
from collections import Counter
from pprint import pprint


from dsg.data_utils import *
from dsg.vqa_utils import calc_vqa_score


print("Calculate Correlation")

QG_models = ['tifa', 'vq2a', 'dsg', 'dsg without dependency']
VQA_models = ['mplug-large', 'instruct-blip', 'pali-17b']
T2I_models = ['mini-dalle', 'sd1dot1', 'sd1dot5', 'sd2dot1', 'vq-diffusion']

item_ids = all_tifa160_item_ids

for vqa_model in VQA_models:
    print('vqa_model:', vqa_model)
    for qg_model in QG_models:
        print(' qg_model:', qg_model)

        n_sample_list = []
        spearman_corr_list = []
        spearman_p_value_corr_list = []
        kendall_corr_list = []
        kendall_p_value_corr_list = []
    
        vqa_scores = []
        human_likert_scores = []

        for t2i_model in T2I_models:
            print('  t2i_model:', t2i_model)

            qg_model_str = qg_model if 'dsg' not in qg_model else 'dsg'

            current_df = all_vqa_dfs[(qg_model_str, vqa_model, t2i_model)]

            for item_id in item_ids:

                item_df = current_df[current_df.item_id == item_id]
                qa_output = load_vqa_output(item_df, qg_model=qg_model, item_id=item_id)
                try:
                    qid2answer = {qa['question_id']: qa['vqa_answer'] for qa in qa_output}
                    qid2gtanswer = {qa['question_id']: qa['gt_answer'] for qa in qa_output}
                except Exception:
                    qid2answer = {1: ""}
                    qid2gtanswer = {1: "XXX"}

                if qg_model == 'dsg':
                    qid2dependency = dsg_id2dependency[item_id]
                else:
                    qid2dependency = None

                vqa_score_dict = calc_vqa_score(qid2answer, qid2dependency, qid2gtanswer)
                avg_score = vqa_score_dict['average_score_with_dependency']
                
                vqa_scores += [avg_score]
                human_likert_scores += [np.mean(load_human_likert_ann(t2i_model, item_id)['likert_scores'])]

        
        correlation_results = compute_correlation(vqa_scores, human_likert_scores, verbose=True)

        n_samples = correlation_results['n_samples']
        kendall_p_value = correlation_results['kendall_pvalue']
        spearman_p_value = correlation_results['spearman_pvalue']
        spearman = correlation_results['spearman']
        kendall = correlation_results['kendall']

        n_sample_list += [n_samples]
        spearman_corr_list += [spearman]
        kendall_corr_list += [kendall]
        spearman_p_value_corr_list += [spearman_p_value]
        kendall_p_value_corr_list += [kendall_p_value]

        print('='* 30)
        print("Summary")
        print(f"QG: {qg_model} | VQA: {vqa_model}")
        print('VQA vs. Likert correlation')
        # print(correlation_results)
        # print in three decimal points
        print(f"Spearman: {spearman:.3f} (p-value: {spearman_p_value})")
        print(f"Kendall: {kendall:.3f} (p-value: {kendall_p_value})")
        print(f"n_samples: {n_samples}")
        print('=' * 30)