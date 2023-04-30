from unittest import TestCase

from HMM import HMM


class TestHMM(TestCase):
    def test_load(self):
        model = HMM()
        model.load('two_english')
        print(model.emissions)
        print(model.transitions)
