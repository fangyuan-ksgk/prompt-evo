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



# Custom Reasoning Dataset
json_file = 'datasets/Big-Bench-Hard/bbh/causal_judgement.json'
with open(json_file, 'r') as f:
    data = json.load(f)
print('Keys: ', data.keys())
examples = data['examples']
print('Number of examples: ', len(examples))
print('Example: ', examples[0])

# Select indices for train & dev & eval || 80% train, 10% dev, 10% eval
num_train = int(0.8 * len(examples))
num_dev = int(0.1 * len(examples))
num_eval = len(examples) - num_train - num_dev

dataset = []
for example in examples:
    question = example['input']
    answer = example['target']
    dataset.append(dspy.Example(question=question, answer=answer).with_inputs('question'))

# Split the data
train = dataset[:num_train]
dev = dataset[num_train:num_train+num_dev]
eval = dataset[num_train+num_dev:]





# train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in train]



# # Load the HotPotQA dataset
# dataset = HotPotQA(train_seed=1, train_size=300, eval_seed=2023, dev_size=300, test_size=0, keep_details=True)
# trainset = [x.with_inputs('question') for x in dataset.train]
# devset = [x.with_inputs('question') for x in dataset.dev]
