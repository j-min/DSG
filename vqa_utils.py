

import os
from pathlib import Path
from PIL import Image
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

def load_image(item_id, model_type, image_dir='../mspice/images/image_v1/'):
    image_path = Path(image_dir) / f'{item_id}_{model_type}.jpg'
    if os.path.exists(image_path):
        return Image.open(image_path).convert('RGB')
    return False


def parse_data_type(src_line):
    return '_'.join(src_line.split('_')[:-1])


def format_question(question, choices):
    return f'Question: {question} Choices: {", ".join(choices)}. Answer:'


def calc_vqa_score(qid2answer, qid2dependency=None, qid2gtanswer=None) -> Dict[str, Any]:
    """Calculate VQA scores of questions and aggregate them into item-level score"""

    if qid2gtanswer is None:
        qid2gtanswer = {qid: 'yes' for qid in qid2answer.keys()}

    qid2scores = {}
    for qid, answer in qid2answer.items():
        gt_answer = qid2gtanswer[qid]
        qid2scores[qid] = float(answer == gt_answer)

    try:
        average_score_without_dep = sum(qid2scores.values()) / len(qid2scores)
    except ZeroDivisionError:
        average_score_without_dep = 0.0

    # zero-out scores from invalid questions 
    qid2validity = {}
    qid2scores_after_filtering = deepcopy(qid2scores)

    if qid2dependency is None:
        # no dependency - all questions are valid
        qid2dependency = {qid: [0] for qid in qid2answer.keys()}

    for qid, parent_ids in qid2dependency.items():
        # zero-out scores if parent questions are answered 'no'
        any_parent_answered_no = False
        for parent_id in parent_ids:
            if parent_id == 0:
                continue
            if qid2scores[parent_id] == 0:
                any_parent_answered_no = True
                break
        if any_parent_answered_no:
            qid2scores_after_filtering[qid] = 0.0
            qid2validity[qid] = False
        else:
            qid2validity[qid] = True

    try:
        average_score_with_dep = sum(qid2scores_after_filtering.values()) / len(qid2scores)
    except ZeroDivisionError:
        average_score_with_dep = 0.0
        
    return {
        # 'qid2tuple': qid2tuple,
        'qid2dependency': qid2dependency,
        # 'qid2question': qid2question,
        'qid2answer': qid2answer,
        'qid2scores': qid2scores,
        'qid2validity': qid2validity,
        'average_score_with_dependency': average_score_with_dep,
        'average_score_without_dependency': average_score_without_dep
    }





##### mPLUG-large #####

class MPLUG:
    def __init__(self, ckpt='damo/mplug_visual-question-answering_coco_large_en'):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt)

    def vqa(self, image, question):
        input_vqa = {'image': image, 'question': question}
        result = self.pipeline_vqa(input_vqa)
        return result['text']

##### InstructBLIP loading #####

class InstructBLIP:
    def __init__(self, ckpt='Salesforce/instructblip-vicuna-7b'):
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        self.processor = InstructBlipProcessor.from_pretrained(ckpt)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(ckpt)

    def vqa(self, image, question):
        device = next(self.model.parameters()).device
        inputs = self.processor(images=image,
                                text=question,
                                return_tensors="pt").to(device)
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()