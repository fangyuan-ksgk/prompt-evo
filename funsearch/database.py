"""A programs database that implements the evolutionary algorithm."""
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any

from absl import logging
import numpy as np
import scipy

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]

def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
  if not np.all(np.isfinite(logits)):
    non_finites = set(logits[~np.isfinite(logits)])
    raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
  if not np.issubdtype(logits.dtype, np.floating):
    logits = np.array(logits, dtype=np.float32)

  result = scipy.special.softmax(logits / temperature, axis=-1)
  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
  """Reduces per-test scores into a single score."""
  return scores_per_test[list(scores_per_test.keys())[-1]]


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
  """Represents test scores as a canonical signature."""
  return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    """
    Prompt to conditon the LLM to produce outputs following certain distribution

    Attributes:
        persona: The persona to condition the LLM to produce high-quality conversations
        version_generated: The version of the prompt generated
        island_id: The id of the island that generated the prompt
    """
    persona: str
    version_generated: int
    island_id: int

# Agent shall have its system prompt composed of a fixed part (being customer of blah blah), and a variable part (being the persona)
# The persona is the part that is evolved | just like the code is the part that is evolved in the program database through funsearch

class AgentDatabase:
  """
  Pool of agents, organized as islands
  """

  def __init__(
      self,
      config,
      template,
      agent_to_evolve: str,
  ) -> None:
    self._config = config
    self._template = template
    self._agent_to_evolve = agent_to_evolve

    # Initialize empty islands
    self._islands = []
    for _ in range(config.num_islands):
      self._islands.append(
        Island(template, agent_to_evolve, config.agents_per_prompt,
               config.cluster_sampling_temperature_init,
               config.cluster_sampling_temperature_period))
    
    self._best_score_per_island: list[float] = (
      [-float('inf')] * config.num_islands)
    
    self._best_agent_per_island = (
      [None] * config.num_islands)
  
    self._best_scores_per_test_per_island = (
      [None] * config.num_islands)
    
    self._last_reset_time = time.time()

def get_prompt(self) -> Prompt:
  
  island_id = np.random.randint(len(self._islands))
  persona, version_generated = self._islands[island_id].get_prompt()
  return Prompt(persona, version_generated, island_id)

def _register_agent_in_island(
    self,
    agent,
    island_id: int,
    scores_per_test
):
    """Registers an agent in the island."""
    self._islands[island_id].register_program(agent, scores_per_test)
    score = _reduce_score(scores_per_test)
    if score > self._best_score_per_island[island_id]:
      self._best_agent_per_island[island_id] = agent
      self._best_scores_per_test_per_island[island_id] = scores_per_test # What is the difference between this and the next line?
      self._best_score_per_island[island_id] = score
      logging.ingo('Best score of island %d increased to %s', island_id, score)

def register_agent(
    self,
    agent,
    island_id: int | None,
    scores_per_test: ScoresPerTest,
):
  """Registers an agent in the database."""
  if island_id is NOne:
    for island_id in range(len(self._islands)):
      self._register_agent_in_island(agent, island_id, scores_per_test)
  else:
    self._register_agent_in_island(agent, island_id, scores_per_test)

# Check whether it is time to reset an island
if (time.time() - self._last_reset_time > self._config.reset_period):
  self._last_reset_time = time.time()
  self.reset_islands()

def reset_islands(self):
  """
  Reset the weaker half of islands
  """
  # Added random noise used to break ties
  indices_sorted_by_score: np.ndarray = np.argsort(
    self._best_score_per_island + 
    np.random.randn(len(self._best_score_per_island)) * 1e-6)
  num_islands_to_reset = self._config.num_islands // 2
  reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
  keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
  for island_id in reset_islands_ids:
    self._islands[island_id] = Island(
      self._template,
      self._agent_to_evolve,
      self._config.agents_per_prompt,
      self._config.cluster_sampling_temperature_init,
      self._config.cluster_sampling_temperature_period)
    self._best_score_per_island[island_id] = -float('inf')
    founder_island_id = np.random.choice(keep_islands_ids)
    founder = self._best_agent_per_island[founder_island_id]
    founder_scores = self._best_scores_per_test_per_island[founder_island_id]
    self._register_agent_in_island(founder, island_id, founder_scores)






class Island:
  """
  A sub-population of the prompts database
  """
  def __init__(
      self,
      template,
      agent_to_evolve: str,
      agents_per_prompt: int,
      cluster_sampling_temperature_init: float,
      cluster_sampling_temperature_period: int,
  ) -> None:
      self._template = template
      self._agent_to_evolve = agent_to_evolve
      self._agents_per_prompt = agents_per_prompt
      self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
      self._cluster_sampling_temperature_period = (
          cluster_sampling_temperature_period)

      self._clusters: dict[Signature, Cluster] = {}
      self._num_programs: int = 0

  def register_agent(
      self,
      agent,
      scores_per_test: ScoresPerTest
  ):
      """Stores an agent on this island, in its appropriate cluster."""
      signatures = list(self._clusters.keys())
      cluster_scores = np.array(
        [self._clusters[signature].score for signature in signatures])
      
      # Convert scores to probabilities using softmax with temperature schedule
      period = self._cluster_sampling_temperature_period
      temperature = self._cluster_sampling_temperature_init * (
        1 - (self.num_programs % period) / period)
      probabilities = _softmax(cluster_scores, temperature)

      # Beginning of an experiment when we have few clusters, place fewer prograns into prompt.
      agents_per_prompt = min(len(self._clusters), self._num_programs)

      idx = np.random.choice(
        len(signatures), agents_per_prompt, p=probabilities)
      chosen_signatures = [signatures[i] for i in idx]
      embodiments = []
      scores = []
      for signature in chosen_signatures:
        cluster = self._clusters[signature]
        embodiments.append(cluster.sample_agent())
        scores.append(cluster.score)

      indices = np.argsort(scores)
      sorted_embodiments = [embodiments[i] for i in indices]
      version_generated = len(sorted_embodiments) + 1
      return self._generate_prompt(sorted_embodiments), version_generated
  
  def _generate_prompt(
      self,
      embodiments: Sequence[Any]) -> str:
    embodiments = copy.deepcopy(embodiments)

    # Format the names and decstrings of functions to be included in the prompt
    versioned_agents: list[Any] = []
    for i, embodiment in enumerate(embodiments):
        new_agent_name = f'{self._agent_to_evolve}_{i}'
        embodiment.name = new_agent_name
        # Update the docstring for all subsequent agents after '_v0'
        if i > 0:
          embodiment.docstring = (
            f'Agent evolved from {self._agent_to_evolve}_{i - 1}')
          embodiment = 
