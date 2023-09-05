# Hidden Markov Model (HMM) Implementation

This project involves the implementation of a Hidden Markov Model (HMM) in Python. The HMM is used to model sequences of observations with underlying hidden states, and it finds applications in various fields such as speech recognition, natural language processing, and bioinformatics.

## Approach Algorithm

The `Approach.py` script uses reinforcement learning to determine the optimal policy for a game called "Approach." The game involves two players rolling dice and deciding whether to "hold" or "roll" to get as close to a limit `n` without going over. The script implements the Q-learning algorithm to learn the best strategy for holding or rolling to maximize the chances of winning.

## HMM Model

The `HMM.py` script implements a Hidden Markov Model. The HMM consists of two main components: transition probabilities and emission probabilities. It reads transition and emission probabilities from external files, normalizes them, and can perform two operations:

- Generate: The script generates an observation sequence of a given length using the HMM model. It randomly samples hidden states and their corresponding outputs based on the learned probabilities.
- Viterbi: Given an observation sequence, the script uses the Viterbi algorithm to determine the most likely sequence of hidden states that generated the output sequence.

## Usage

### Approach Algorithm

To run the Approach algorithm, execute the following command:

python Approach.py

yaml
Copy code

The script will output the optimal strategy for holding or rolling at different sums.

### HMM Model

To run the HMM model, use the following commands:

For generating observations:
python HMM.py [basename] --generate [length]

csharp
Copy code

For performing the Viterbi algorithm on observation sequences:
python HMM.py [basename] --viterbi [path_to_observations_file]

css
Copy code

Replace `[basename]` with the base name of HMM model files, `[length]` with the desired observation sequence length, and `[path_to_observations_file]` with the path to the file containing observation sequences.

## What I Learned

- Reinforcement learning concepts and the Q-learning algorithm for optimizing strategies.
- Implementation of Hidden Markov Models (HMM) and their applications in modeling sequences.
- Utilizing external files to load and normalize transition and emission probabilities for HMM.
- Implementing the Viterbi algorithm for finding the most likely sequence of hidden states.
- Command-line argument parsing and usage in Python scripts.

This project has provided valuable experience in implementing machine learning algorithms and probabilistic models to solve real-world problems.