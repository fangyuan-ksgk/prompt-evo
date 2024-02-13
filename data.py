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


class GenerateAnswer(dspy.Signature):
    """Provide an answer to a question."""
    context = dspy.InputField(desc="may contain relavent facts")
    question = dspy.InputField(desc="the question to be answered")
    answer = dspy.OutputField(desc="the answer to the question, should be either Yes or No")


class giveQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question, context):
        pred = self.gen_answer(context=context, question=question)
        pred = dspy.Prediction(answer = pred)
        return pred


# Essentially check the correctness of the answer between prediction and GT label
def check_correctness(example, pred):
    question, answer = example.question, example.answer
    pred_answer = pred.answer.lower().startswith('yes')
    gt_answer = answer.lower().startswith('yes')
    return pred_answer == gt_answer


