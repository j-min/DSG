

import os
from pathlib import Path
from PIL import Image


def load_image(item_id, model_type, image_dir='../mspice/images/image_v1/'):
    image_path = Path(image_dir) / f'{item_id}_{model_type}.jpg'
    if os.path.exists(image_path):
        return Image.open(image_path).convert('RGB')
    return False


def parse_data_type(src_line):
    return '_'.join(src_line.split('_')[:-1])


def format_question(question, choices):
    return f'Question: {question} Choices: {", ".join(choices)}. Answer:'


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