# Load dataset for testing on the prompt performance
import dspy
from dspy.predict import Retry
from dspy.datasets import HotPotQA
import json

from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dsp.utils import EM, normalize_text
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

from utils import extract_text_by_citation, correct_citation_format, has_citations, citations_check

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=500)
dspy.settings.configure(lm=turbo, trace=[], temperature=0.7)

def get_dataset(json_file: str = 'datasets/Big-Bench-Hard/bbh/causal_judgement.json'):

    # Custom Reasoning Dataset
    json_file = 'datasets/Big-Bench-Hard/bbh/causal_judgement.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
    examples = data['examples']

    # Select indices for train & dev & eval || 80% train, 10% dev, 10% eval
    num_train = int(0.8 * len(examples))
    num_dev = int(0.1 * len(examples))
    num_eval = len(examples) - num_train - num_dev

    dataset = []
    for example in examples:
        question = example['input']
        answer = example['target']
        # like initializing a dictionary, but extended class separating inputs kesy and label keys
        dataset.append(dspy.Example(question=question, answer=answer).with_inputs('question'))

    # Split the data (Loaded)
    train = dataset[:num_train]
    dev = dataset[num_train:num_train+num_dev]
    eval = dataset[num_train+num_dev:]

    return train, dev, eval




def citation_faithfulness(example, pred, trace):
    paragraph, context = pred.paragraph, pred.context
    citation_dict = extract_text_by_citation(paragraph)
    if not citation_dict:
        return False, None
    context_dict = {str(i): context[i].split(' | ')[1] for i in range(len(context))}
    faithfulness_results = []
    unfaithful_citations = []
    check_citation_faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    for citation_num, texts in citation_dict.items():
        if citation_num not in context_dict:
            continue
        current_context = context_dict[citation_num]
        for text in texts:
            try:
                result = check_citation_faithfulness(context=current_context, text=text)
                is_faithful = result.faithfulness.lower() == 'true'
                faithfulness_results.append(is_faithful)
                if not is_faithful:
                    unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'context': current_context})
            except ValueError as e:
                faithfulness_results.append(False)
                unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'error': str(e)})
    final_faithfulness = all(faithfulness_results)
    if not faithfulness_results:
        return False, None
    return final_faithfulness, unfaithful_citations

# Essentially check the correctness of the answer between prediction and GT label
def check_correctness(example, pred):
    question, answer = example.question, example.answer
    pred_answer = pred.answer
    return pred_answer.lower() == answer.lower()


