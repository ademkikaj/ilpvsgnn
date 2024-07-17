import pyparsing as pp
from pyparsing import Word, alphas, nums
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


predicate = Word(alphas).setResultsName('predicate')

numeric_term = Word(nums + '.')
variable = Word(alphas)
variable_with_numeric = Word(alphas + nums)
term = (numeric_term | variable | variable_with_numeric).setResultsName('term')

terms = (pp.Suppress('(') + pp.delimitedList(term) + pp.Suppress(')')).setResultsName('terms')

fact  = (predicate + terms).setResultsName('facts', listAllMatches=True) + pp.Suppress('.')
facts = pp.OneOrMore(fact)

knowledge_base = []


test = 'triangle(2,o5).'

class ILPParser:

    def __init__(self):
        pass

    # parses the line and returns a list of the predicate and the terms
    # output format: [predicate, term1, term2, ...]
    def parseLine(self,line):
        # the input of the function shouldn't be an empry line
        # extract the predicate
        split_1 = line.split('(')
        predicate = split_1[0]
        # extract the terms
        terms = split_1[1].split(')')
        terms = terms[0].split(',')
        
        fact = [predicate] + terms

        # strip leading and trailing whitespaces
        fact = [x.strip() for x in fact]
        return fact


    # splite the knowledge base into respective predicate categories
    # output format: {predicate: [[predicate, term1, term2, ...], ...], ...}
    def splitPredicates(self,kb):
        predicates = []
        result = {}
        for li in kb:
            predicate = li[0]
            if predicate not in predicates:
                predicates.append(predicate)
                result[predicate] = [li[1:]]
            else:
                result[predicate].append(li)
        return result, predicates

    def toCSV(self,filename, cols):

        knowledge_base = []
        with open(filename,"r") as file:
            for line in file:
                if line.isspace():
                    continue
                else: 
                    fact = self.parseLine(line)
                    knowledge_base.append(fact)
        knowledge_base,predicates = self.splitPredicates(knowledge_base)

        for pred in predicates:
            cols = cols[pred]
            knowledge_base[pred] = pd.DataFrame(knowledge_base[pred])
            knowledge_base[pred].columns = cols

            knowledge_base[pred].to_csv("datasets/Bongard/Relational/" + pred + ".csv", index=False)


filePath = "datasets/Bongard/TILDE/bongard.kb"
cols = {"bongard": ["problemId", "class"], 
        "triangle": ["problemId", "objectId"],
        "square": ["problemId", "objectId"],
        "circle": ["problemId", "objectId"],
        "config": ["problemId", "objectId", "orient"],
        "in": ["problemId", "objectId1", "objectId2"]}
destinationPath = 'datasets/Bongard/Relational/'

