import ast
from collections import Iterator, MutableSet, Sequence
import dataclasses
import io
import tokenize
from absl import logging
from .template import agent_template

@dataclasses.dataclass
class Gene:
    name: str
    description: str

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __setattr__(self, name: str, value: str) -> None: # Controls when we are setting attribute of class instance
        if name == 'description':
            value = value.strip('\n')
        super().__setattr__(name, value)

def format_chromosome(genes):
    return "\n  ".join(genes)


@dataclasses.dataclass(frozen=True)
class Agent:
    """ Agent with gene prompts"""
    template: str = agent_template # Default to FWD insurance customer template
    genes: list[Gene]

    def __str__(self) -> str:
        return f"{self.template}\n{format_chromosome(self.genes)}"

    def find_gene_index(self, gene_name: str) -> int:
        """Returns the index of input gene name."""
        gene_names = [gene.name for gene in self.genes]
        count = gene_names.count(gene_name)
        if count == 0:
            raise ValueError(f"Gene {gene_name} not found in agent:\n{self.preface}")
        elif count > 1:
            raise ValueError(f'Multiple genes with name {gene_name} found in agent:\n'
                             f'{self.preface}')
        
        index = gene_names.index(gene_name)
        return index
    
    def get_gene(self, gene_name: str) -> Gene:
        """Returns the gene with input name."""
        index = self.find_gene_index(gene_name)
        return self.genes[index]
    

# At this point, I do not believe it is necessary to have text_to_agent, or text_to_gene in the codebase...





        

