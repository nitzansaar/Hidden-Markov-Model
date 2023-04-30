import random
import argparse
import codecs
import os
import numpy
import numpy as np


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

            # Normalize transition probabilities
            for state1 in self.transitions:
                total_prob = sum(self.transitions[state1].values())
                for state2 in self.transitions[state1]:
                    self.transitions[state1][state2] /= total_prob

        with open(basename + ".emit", "r") as f:
            lines = f.readlines()
            for line in lines:
                state, symbol, prob = line.strip().split()
                if state not in self.emissions:
                    self.emissions[state] = {}
                self.emissions[state][symbol] = float(prob)

            # Normalize emission probabilities
            for state in self.emissions:
                total_prob = sum(self.emissions[state].values())
                for symbol in self.emissions[state]:
                    self.emissions[state][symbol] /= total_prob

    ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        state = '#'  # start with the initial state
        stateseq = []
        outputseq = []

        for _ in range(n):
            next_state = np.random.choice(list(self.transitions[state].keys()),
                                          p=list(self.transitions[state].values()))
            stateseq.append(next_state)

            output = np.random.choice(list(self.emissions[next_state].keys()),
                                      p=list(self.emissions[next_state].values()))
            outputseq.append(output)

            state = next_state  # update the current state

        return Observation(stateseq, outputseq)

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        num_states = len(self.transitions.keys()) - 1
        num_outputs = len(observation.outputseq)

        states = list(self.transitions.keys())
        states.remove("#")

        viterbi = np.zeros((num_states, num_outputs))

        backpointer = np.zeros((num_states, num_outputs), dtype=int)

        for i, state in enumerate(states):
            viterbi[i, 0] = self.transitions["#"][state] * self.emissions[state].get(observation.outputseq[0], 0)

        for t in range(1, num_outputs):
            for i, state in enumerate(states):
                max_prob = -1
                max_prob_idx = -1
                for j, prev_state in enumerate(states):
                    transition_prob = self.transitions[prev_state].get(state, 0)
                    prob = viterbi[j, t - 1] * transition_prob * self.emissions[state].get(observation.outputseq[t], 0)
                    if prob > max_prob:
                        max_prob = prob
                        max_prob_idx = j

                viterbi[i, t] = max_prob
                backpointer[i, t] = max_prob_idx

        best_path_pointer = np.argmax(viterbi[:, -1])
        best_path = [states[best_path_pointer]]

        for t in reversed(range(1, num_outputs)):
            best_path_pointer = backpointer[best_path_pointer, t]
            best_path.insert(0, states[best_path_pointer])

        return best_path


def main():
    parser = argparse.ArgumentParser(description="HMM model operations")
    parser.add_argument("basename", help="Base name for the HMM model files")
    parser.add_argument("--generate", type=int, help="Generate a sequence of the given length")
    parser.add_argument("--viterbi", type=str, help="Path to the file containing observations")

    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.basename)

    if args.generate:
        observation = hmm.generate(args.generate)
        print(f"Generated observation:\n{observation}")

    if args.viterbi:
        with open(args.viterbi, 'r') as f:
            for line in f.readlines():
                observation_output = line.strip().split()
                if len(observation_output) == 0:
                    continue
                observation = Observation([], observation_output)
                best_path = hmm.viterbi(observation)
                print(f"Observation: {' '.join(observation.outputseq)}")
                print(f"Most likely states: {' '.join(best_path)}\n")


if __name__ == "__main__":
    main()
