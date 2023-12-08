
# Davidsonian Scene Graph (DSG) Code

> [!NOTE]
> For more info about DSG, please visit our [project page](https://google.github.io/dsg/).

This repository contains the code for **DSG**, a new framework for fine-grained text-to-image evaluation using Davidsonian semantics, as described in the paper:

**[Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation](https://arxiv.org/abs/2310.18235)**

by  [Jaemin Cho](https://j-min.io),
[Yushi Hu](https://yushi-hu.github.io/),
[Jason Baldridge](http://jasonbaldridge.com/),
[Roopal Garg](https://roopalgarg.com/),
[Peter Anderson](https://panderson.me/),
[Ranjay Krishna](https://ranjaykrishna.com/),
[Mohit Bansal](https://www.cs.unc.edu/~mbansal/),
[Jordi Pont-Tuset](https://jponttuset.cat/),
[Su Wang](https://research.google/people/107321/)

<br>

## Background - QG/A frameworks for text-to-image evaluation

<img src="./assets/qg-qa.png" width=1000px>

Evaluating text-to-image models is notoriously difficult. A strong recent approach for assessing text-image faithfulness is based on QG/A (question generation and answering), which uses pre-trained foundational models to automatically generate a set of questions and answers from the prompt, and output images are scored based on whether these answers extracted with a visual question answering model are consistent with the prompt-based answers. This kind of evaluation naturally depends on the quality of the underlying QG and QA models. We identify and address several reliability challenges in existing QG/A work: (a) QG questions should respect the prompt (avoiding hallucinations, duplications, and omissions), and (b) VQA answers should be consistent (not assert that there is no motorcycle in an image while also claiming the motorcycle is blue).

## Our solution - Davidsonian Scene Graph (DSG)

<img src="./assets/dsg_eval.png" width=1000px>

We address these issues with **Davidsonian Scene Graph (DSG)**, an empirically grounded evaluation framework inspired by formal semantics. DSG is an automatic, graph-based QG/A that is modularly implemented to be adaptable to any QG/A module. DSG produces atomic and unique questions organized in dependency graphs, which (i) ensure appropriate semantic coverage and (ii) sidestep inconsistent answers. With extensive experimentation and human evaluation on a range of model conﬁgurations (LLM, VQA, and T2I), we empirically demonstrate that DSG addresses the challenges noted above. Finally, we present DSG-1k, an open-sourced evaluation benchmark with 1,060 prompts, covering a wide range of ﬁne-grained semantic categories with a balanced distribution. We will release the DSG-1k prompts and the corresponding DSG questions.

<br>


## Example usage of DSG

Below is the pseudocode for evaluating text-to-image generation models with DSG.

Please check [./t2i_eval_example.ipynb](./t2i_eval_example.ipynb) for a full example using gpt-3.5-turbo 16k as an LLM and mPLUG-large as a VQA model. You can replace the LLM and VQA models with any other models.

```python
PROMPT_TUPLE = """Task: given input prompts,
describe each scene with skill-specific tuples ...
"""

PROMPT_DEPENDENCY = """Task: given input prompts and tuples,
describe the parent tuples of each tuple ...
"""

PROMPT_QUESTION = """Task: given input prompts and skill-specific tuples,
re-write tuple each in natural language question ...
"""

def generate_dsg(text, LLM):
    """generate DSG (tuples, dependency, and questions) from text"""
    # 1) generate atomic semantic tuples
    # output: dictionary of {tuple id: semantic tuple}
    id2tuples = LLM(text, PROMPT_TUPLE)
    # 2) generate dependency graph from the tuples
    # output: dictionary of {tuple id: ids of parent tuples}
    id2dependency = LLM(text, id2tuples, PROMPT_DEPENDENCY)
    # 3) generate questions from the tuples
    # output: dictionary of {tuple id: ids of generated questions}
    id2questions = LLM(text, id2tuples, PROMPT_QUESTION)
    return id2tuples, id2dependency, id2questions

def evaluate_image_dsg(text, generated_image, VQA, LLM):
    """evaluate a generated image with DSG"""
    # 1) generate DSG from text
    id2tuples, id2dependency, id2questions = generate_dsg(text, LLM)
    # 2) answer questions with the generated image
    id2scores = {}
    for id, question in id2questions.items():
        answer = VQA(generated_image, question)
        id2scores[id] = float(answer == 'yes')
    # 3) zero-out scores from invalid questions 
    for id, parent_ids in id2dependency.items():
        # zero-out scores if parent questions are answered 'no'
        any_parent_answered_no = False
        for parent_id in parent_ids:
            if id2scores[parent_id] == 0:
                any_parent_answered_no = True
                break
        if any_parent_answered_no:
            id2scores[id] = 0
    # 4) calculate the final score by averaging
    average_score = sum(id2scores.values()) / len(id2scores)
    return average_score
```




<br>

## Code Structure

```bash
# Generate DSG with arbitrary LLM
query_utils.py

# Parse generation results from LLM
parse_utils.py

# Example LLM call - GPT-3.5 turbo 16k
openai_utils.py

# Example VQA call - mPLUG-large / InstructBLIP
vqa_utils.py

# Example TIFA dev json
tifa160-dev-anns.json
```

## Setup

```bash
conda create -n dsg python=3.9
conda activate dsg
pip install -r requirements.txt

# (optional) if you want to use GPT models via OpenAI API for DSG-generating LLM
pip install openai

# (optional) if you want to use mPLUG-large / InstructBLIP for VQA models
pip install transformers==4.31.0 torch==2.0.1 "modelscope[multi-modal]" salesforce-lavis
```


## Citation

If you find our project useful in your research, please cite the following paper:

```bibtex
@article{Cho2023DSG,
  author    = {Jaemin Cho and Yushi Hu and Jason Baldridge and Roopal Garg and Peter Anderson and Ranjay Krishna and Mohit Bansal and Jordi Pont-Tuset and Su Wang},
  title     = {Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation},
  year      = {2023},
}
```
