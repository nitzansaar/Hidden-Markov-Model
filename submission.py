from unittest import TestCase

from HMM import HMM


class TestHMM(TestCase):
    def test_load(self):
        model = HMM()
        model.load('two_english')
        print(model.emissions)
        print(model.transitions)

    def test_generate(self):
        model = HMM()
        model.load('partofspeech.browntags.trained')
        observation = model.generate(20)
        print(observation)

    def test_viterbi(self):
        model = HMM()
        model.load('partofspeech.browntags.trained')
        observation = model.generate(20)
        most_likely_states = model.viterbi(observation)
        print(f"Observation: {' '.join(observation.outputseq)}")
        print(f"True states: {' '.join(observation.stateseq)}")
        print(f"Most likely states: {' '.join(most_likely_states)}")
