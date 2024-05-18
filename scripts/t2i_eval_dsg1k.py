from pathlib import Path
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Optional
from collections import Counter
from pprint import pprint
from copy import deepcopy


from dsg.data_utils import *
from dsg.vqa_utils import calc_vqa_score


print("Evaluating T2I models on DSG1k")

QG_models = ['dsg']
VQA_models = ['mplug-large', 'instruct-blip', 'pali-17b']
T2I_models = ['sd2dot1']

item_ids = all_dsg1k_item_ids

qg_model = 'dsg'
vqa_model = 'pali-17b'

for t2i_model in T2I_models:
    print('t2i_model:', t2i_model)
    print()

    for dataset_category in all_data_categories + ['all']:

        vqa_scores = []

        current_df = all_vqa_dfs[(qg_model, vqa_model, t2i_model)]

        # item_ids = all_dsg1k_item_ids
        if dataset_category == 'all':
            item_ids = all_dsg1k_item_ids
        else:
            item_ids = [item_id for item_id in all_dsg1k_item_ids if get_data_category(item_id) == dataset_category]
        
        for item_id in item_ids:
            qg_df_name = qg_model if 'dsg' not in qg_model else 'dsg'
            item_df = current_df[current_df.item_id == item_id]
            qa_output = load_vqa_output(item_df, qg_model=qg_df_name, item_id=item_id)

            try:
                qid2answer = {qa['question_id']: qa['vqa_answer'] for qa in qa_output}
                # qid2gtanswer = {qa['question_id']: qa['gt_answer'] for qa in qa_output}
            except Exception:
                qid2answer = {1: ""}
                # qid2gtanswer = {1: "XXX"}

            if qg_model == 'dsg':
                qid2dependency = dsg_id2dependency[item_id]
            else:
                qid2dependency = None

            vqa_score_dict = calc_vqa_score(qid2answer, qid2dependency,
                                            # qid2gtanswer
                                            )
            avg_score = vqa_score_dict['average_score_with_dependency']

            vqa_scores += [avg_score]

        avg_vqa_score = np.mean(vqa_scores)
        n_items = len(vqa_scores)

        print(f"Dataset category: {dataset_category} | # items: {n_items}")
        # print(f"VQA: {vqa_model} | T2I: {t2i_model}")
        print(f"Avg. score: {avg_vqa_score * 100:.1f}%")