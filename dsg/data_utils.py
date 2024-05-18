
from pathlib import Path
import json
import pandas as pd
import numpy as np

###############################
# Load TIFA160 Likert Scores
###############################

human_tifa160_likert_df = pd.read_csv(Path(__file__).parent / 'data/tifa160-likert-anns.csv')
def load_human_likert_ann(t2i_model, item_id):
    """Load the Likert scores of human annotations on DSG-1k prompts"""

    assert t2i_model in ['mini-dalle', 'sd1dot1', 'sd1dot5', 'sd2dot1', 'vq-diffusion'], t2i_model
    
    human_likert_df = human_tifa160_likert_df[human_tifa160_likert_df.t2i_model == t2i_model]
    item_df = human_likert_df[human_likert_df.item_id == item_id]

    worker_ids = item_df.worker_id.unique().tolist()
    n_workers = len(worker_ids)

    human_likert_output = {
        "item_id": item_id,
        "t2i_model": t2i_model,
        "worker_ids": worker_ids,
        "n_workers": n_workers,
        "likert_scores": item_df.answer.tolist(),
    }

    return human_likert_output

dsg_id_to_tifa_id = {}
tifa_id_to_dsg_id = {}
for i, row in human_tifa160_likert_df.iterrows():
    dsg_id_to_tifa_id[row['item_id']] = row['source_id']
    tifa_id_to_dsg_id[row['source_id']] = row['item_id']


###############################
# Load DSG annotations
###############################

dsg_df = pd.read_csv(Path(__file__).parent / 'data/dsg-1k-anns.csv')

dsg_itemid2data = {}
for idx, row in dsg_df.iterrows():
    item_id = row['item_id']
    
    if item_id not in dsg_itemid2data:
        data = []
    else:
        data = dsg_itemid2data[item_id]

    # add the row
    data.append(row)
    dsg_itemid2data[item_id] = data
# merge the data
for item_id, data in dsg_itemid2data.items():
    dsg_itemid2data[item_id] = pd.concat(data, axis=1).T

dsg_id2tuple = {}
dsg_id2question = {}
dsg_id2dependency = {}

for item_id, item_df in dsg_itemid2data.items():
    try:
        qid2tup = {}
        for idx, row in item_df.iterrows():
            qid = row['proposition_id']
            output = row['tuple']
            qid2tup[qid] = output

    except Exception:
        qid2tup = {}
    dsg_id2tuple[item_id] = qid2tup

for item_id, item_df in dsg_itemid2data.items():
    try:
        qid2q = {}
        for idx, row in item_df.iterrows():
            qid = row['proposition_id']
            output = row['question_natural_language']
            qid2q[qid] = output

    except Exception:
        qid2q = {}
    dsg_id2question[item_id] = qid2q

for item_id, item_df in dsg_itemid2data.items():
    try:
        qid2dep = {}
        for idx, row in item_df.iterrows():
            qid = row['proposition_id']
            output = row['dependency']
            if type(output) == str:
                output = list(output.split(","))
                output = [int(x.strip()) for x in output]
            qid2dep[qid] = output

    except Exception:
        qid2dep = {}
    dsg_id2dependency[item_id] = qid2dep

###############################
# Load TIFA question annotations (only need for TIFA's original questions)
###############################

# from https://github.com/Yushi-Hu/tifa/blob/main/tifa_v1.0/tifa_v1.0_question_answers.json
tifa_annotation_path = 'tifa_ann/tifa_v1.0_question_answers.json'
with open(tifa_annotation_path, 'r') as f:
    tifa_ann = json.load(f)

dsg_id_to_tifa160_question_data = {}
for d in tifa_ann:
    tifa_id = d['id']
    if tifa_id not in tifa_id_to_dsg_id:
        continue
    dsg_id = tifa_id_to_dsg_id[tifa_id]

    if dsg_id not in dsg_id_to_tifa160_question_data:
        dsg_id_to_tifa160_question_data[dsg_id] = []
    dsg_id_to_tifa160_question_data[dsg_id] += [d]
    
assert len(dsg_id_to_tifa160_question_data) == 160


###############################
# Load VQA answers
###############################

vqa_answers_dir = Path('vqa_answers_release')

def get_vqa_df(qg_model, vqa_model, t2i_model):
    assert qg_model in [
        'tifa',
        'vq2a',
        'dsg'
        ], qg_model
    assert vqa_model in ['pali-17b', 'mplug-large', 'instruct-blip'], vqa_model

    assert t2i_model in ['mini-dalle', 'sd1dot1', 'sd1dot5', 'sd2dot1', 'vq-diffusion'], t2i_model

    if vqa_model == 'pali-17b':
        vqa_str = 'pali17b'
    elif vqa_model == 'mplug-large':
        vqa_str = 'mplug'
    elif vqa_model == 'instruct-blip':
        vqa_str = 'instructblip'

    vqa_file_path = vqa_answers_dir / f"{vqa_str}_{qg_model}.csv"
    df = pd.read_csv(vqa_file_path)
    df = df[df.t2i_model == t2i_model]
    return df

all_vqa_dfs = {}
for qg_model in [
        'tifa',
        'vq2a',
        'dsg'
        ]:
    for vqa_model in [
        'pali-17b',
        'instruct-blip',
        'mplug-large',
        ]:

        t2i_model_list = ['mini-dalle', 'sd1dot1', 'sd1dot5', 'sd2dot1', 'vq-diffusion']

        for t2i_model in t2i_model_list:
            df = get_vqa_df(qg_model, vqa_model, t2i_model)
            all_vqa_dfs[(qg_model, vqa_model, t2i_model)] = df
            # print((qg_model, vqa_model, t2i_model), df.shape)


###############################
all_dsg1k_item_ids = all_vqa_dfs[('dsg', 'pali-17b', 'sd2dot1')].item_id.unique().tolist()
if 'tifa160_134' not in all_dsg1k_item_ids:
    all_dsg1k_item_ids += ['tifa160_134']
# print(len(all_dsg1k_item_ids))
all_tifa160_item_ids = [f"tifa160_{i}" for i in range(160)]
# print(len(all_tifa160_item_ids))


###############################
# Assign dataset categories
###############################

def get_data_category(data_str):

    if 'tifa' in data_str:
        return 'tifa'
    elif 'stanford' in data_str or 'localized' in data_str:
        return 'paragraph'

    elif 'countbench' in data_str:
        return 'count'

    elif 'vrd' in data_str:
        return 'relation'

    elif 'diffusion' in data_str or 'midjourney' in data_str:
        return 'real_user'

    elif 'posescript' in data_str:
        return 'pose'

    elif 'whoops' in data_str:
        return 'defying'

    elif 'drawtext' in data_str:
        return 'text'

    else:
        print("Error", data_str)

all_data_names = []
for item_id in all_dsg1k_item_ids:
    data_name = item_id.split('_')[0]

    get_data_category(item_id)

    if data_name not in all_data_names:
        all_data_names += [data_name]

assert len(all_data_names) == 10
# print('dataset sources')
# print(all_data_names)

# print()

# print('categories')
all_data_categories = all_vqa_dfs[('dsg', 'pali-17b', 'sd2dot1')]['item_id'].apply(get_data_category).unique().tolist()
# print(all_data_categories)


###############################
# Assign question categories
###############################
def get_tuple(item_id, question_id):

    id2tuples = dsg_id2tuple[item_id]

    tup_str = id2tuples[question_id]

    return tup_str

def get_question_category(tuple_str):

    if 'entity' in tuple_str:
        return 'entity'

    elif 'attribute' in tuple_str or 'other' in tuple_str:
        return 'attribute'

    elif 'relation' in tuple_str or 'action' in tuple_str:
        return 'relation'

    elif 'global' in tuple_str:
        return 'global'

    else:
        print('Exception:', tuple_str)

# print('question categories')

item_ids = all_vqa_dfs[('dsg', 'pali-17b', 'sd2dot1')]['item_id']
question_ids = all_vqa_dfs[('dsg', 'pali-17b', 'sd2dot1')]['question_id']

all_broad_q_categories = []
all_detail_q_categories = []
for item_id, question_id in zip(item_ids, question_ids):
    tuple_str = get_tuple(item_id, question_id)

    q_category = get_question_category(tuple_str)

    all_broad_q_categories += [q_category]
    all_detail_q_categories += [tuple_str]

# print(all_vqa_dfs[('dsg', 'pali-17b', 'sd2dot1')]['item_id'].apply(get_data_category).unique())

###############################
# Load VQA output
###############################

def load_vqa_output(item_df, qg_model=None, item_id=None):
    """Parse item-level VQA result df into question-level results
    
    outputs: [
        {
            "question_id": id,
            "question_type": qtype (tuple),
            "vqa_answer": answer,
            "vqa_score": score,
        }
    ]
    """
    # Exception handling
    if len(item_df) == 0:
        # if item_id in ['tifa160_134'] and :
        qa_datum = {
            "question_id": 1,
            "item_id": item_id,
            "qg_model": qg_model,
            "vqa_score": 0,
            "vqa_answer": "",
            "dep_valid": False,
            "question_type_broad": None,
            "question_type_detailed": None,
        }
        return [qa_datum]

    if item_id is None:
        item_id = item_df.item_id.unique()[0]

    outputs = []
    for row_i, row in enumerate(item_df.itertuples()):
        
        # multi-choice (tifa question only)
        if row.qg_model == 'tifa' and dsg_id_to_tifa160_question_data[item_id][row_i]['choices'] != ['yes', 'no']:
            gt_answer = dsg_id_to_tifa160_question_data[item_id][row_i]['answer']
            vqa_score = float(row.answer == gt_answer)
            choices = dsg_id_to_tifa160_question_data[item_id][row_i]['choices']

        # Binary
        else:
            vqa_score = float(row.answer == 'yes')
            gt_answer = 'yes'                        
            choices = ['yes', 'no']

        if str(row.answer) == 'nan':
            vqa_score = 0.0
            
        if 'question_id' in item_df.columns:
            question_id = int(row.question_id)
        else:
            question_id = row_i + 1

        if row.qg_model == 'dsg':
            tuple_str = get_tuple(item_id, question_id)
            q_category = get_question_category(tuple_str)
        else:
            tuple_str = None
            q_category = None

        qa_datum = {
            "item_id": item_id,
            "qg_model": row.qg_model,
            "question_id": question_id,
            "question": row.question,
            "question_type_broad": q_category,
            "question_type_detailed": tuple_str,
            "vqa_answer": row.answer,
            "answer_choices": choices,
            "gt_answer": gt_answer,
            "vqa_score": vqa_score,
            "dep_valid": None,
        }

        outputs += [qa_datum]

    return outputs

###############################
# Method to Calculate Correlation
###############################

from scipy.stats import kendalltau, spearmanr
def compute_correlation(metric1_scores, metric2_scores, verbose=False, filter_nan=True):
    assert len(metric1_scores) == len(metric2_scores)

    if verbose:
        print("N items:", len(metric1_scores))

    if filter_nan:
        new_metric1_scores = []
        new_metric2_scores = []
        for s1, s2 in zip(metric1_scores, metric2_scores):
            if np.isnan(s1):
                continue
            new_metric1_scores += [s1]
            new_metric2_scores += [s2]
        metric1_scores = new_metric1_scores
        metric2_scores = new_metric2_scores

        if verbose:
            print("N items (after filtering nan):", len(metric1_scores))
            assert len(metric1_scores) == len(metric2_scores)

            # print(metric1_scores[:10])
            # print(metric2_scores[:10])

    spearman_result = spearmanr(metric1_scores, metric2_scores)
    spearman = spearman_result.statistic
    spearman_pvalue = spearman_result.pvalue

    kendall_result = kendalltau(metric1_scores, metric2_scores)
    kendall = kendall_result.statistic
    kendall_pvalue = kendall_result.pvalue

    return {
        'spearman': spearman,
        'spearman_pvalue': spearman_pvalue,
        'kendall': kendall,
        'kendall_pvalue': kendall_pvalue,
        'n_samples': len(metric1_scores)
    }