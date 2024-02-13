# Load dataset for testing on the prompt performance
import dspy
from dspy.predict import Retry
from dspy.datasets import HotPotQA
import json

from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dsp.utils import EM, normalize_text
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

from utils import extract_text_by_citation, correct_citation_format, has_citations, citations_check


def get_dataset(json_file: str = 'datasets/Big-Bench-Hard/bbh/causal_judgement.json'):

    turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=500)
    dspy.settings.configure(lm=turbo, trace=[], temperature=0.7)


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
