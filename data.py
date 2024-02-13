# Load dataset for testing on the prompt performance
import dspy
from dspy.predict import Retry
from dspy.datasets import HotPotQA

from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dsp.utils import EM, normalize_text
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

from utils import extract_text_by_citation, correct_citation_format, has_citations, citations_check

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=500)
dspy.settings.configure(lm=turbo, trace=[], temperature=0.7)

# Custom Reasoning Dataset
train = [('Who was the director of the 2009 movie featuring Peter Outerbridge as William Easton?', 'Kevin Greutert'),
         ('The heir to the Du Pont family fortune sponsored what wrestling team?', 'Foxcatcher'),
         ('In what year was the star of To Hell and Back born?', '1925'),
         ('Which award did the first book of Gary Zukav receive?', 'U.S. National Book Award'),
         ('What documentary about the Gilgo Beach Killer debuted on A&E?', 'The Killing Season'),
         ('Which author is English: John Braine or Studs Terkel?', 'John Braine'),
         ('Who produced the album that included a re-recording of "Lithium"?', 'Butch Vig')]

train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in train]



# Load the HotPotQA dataset
dataset = HotPotQA(train_seed=1, train_size=300, eval_seed=2023, dev_size=300, test_size=0, keep_details=True)
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]
