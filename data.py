# Load dataset for testing on the prompt performance
import dspy
from dspy.predict import Retry
from dspy.datasets import HotPotQA

from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dsp.utils import EM, normalize_text
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

from utils import extract_text_by_citation, correct_citation_format, has_citations, citations_check

