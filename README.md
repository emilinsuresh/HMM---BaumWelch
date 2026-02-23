Hidden Markov Model - Baum Welch Algorithm

Emilin Suresh
TCR24CS026

Description
This project implements the Baum–Welch Algorithm (Expectation–Maximization)
to train a Hidden Markov Model.

Inputs:
- Observation sequence
- Number of hidden states

Outputs:
- P(O | lambda)
- Transition Matrix (A)
- Emission Matrix (B)
- Initial Distribution (pi)

How to Run

1. Install Python 3
2. Install numpy:
   pip install numpy
3. Run the program:
   python hmm_baum_welch.py
4. Enter number of hidden states when prompted.

Example
Observations used in code:
Walk = 0
Shop = 1
Sequence = [0, 1, 1, 0, 1]
