import random
import argparse
import codecs
import os
import numpy


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n ' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        with open(basename + ".trans", "r") as f:
            lines = f.readlines()  # read all lines from file into a list of strings
            for line in lines:
                state1, state2, prob = line.strip().split()  # split each line using whitespace as delimiter
                if state1 not in self.transitions:  # need to create new dictionary for new states
                    self.transitions[state1] = {}
                self.transitions[state1][state2] = float(prob)
        with open(basename + ".emit", "r") as f:
            lines = f.readlines()
            for line in lines:
                state, symbol, prob = line.strip().split()
                if state not in self.emissions:
                    self.emissions[state] = {}
                self.emissions[state][symbol] = float(prob)

    ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
