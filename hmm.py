import numpy as np

class HMM:
    def __init__(self, n_states, n_observations):
        self.N = n_states
        self.M = n_observations

        # Random initialization
        self.A = self.normalize(np.random.rand(self.N, self.N))
        self.B = self.normalize(np.random.rand(self.N, self.M))
        self.pi = self.normalize(np.random.rand(self.N))

    def normalize(self, matrix):
        if len(matrix.shape) == 1:
            return matrix / matrix.sum()
        return matrix / matrix.sum(axis=1, keepdims=True)

    # Forward algorithm
    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))

        alpha[0] = self.pi * self.B[:, O[0]]

        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]

        return alpha

    # Backward algorithm
    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[T-1] = np.ones(self.N)

        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(
                    self.A[i, :] * self.B[:, O[t+1]] * beta[t+1]
                )

        return beta

    def baum_welch(self, O, iterations=10):
        T = len(O)

        for _ in range(iterations):
            alpha = self.forward(O)
            beta = self.backward(O)

            P_O = np.sum(alpha[-1])

            gamma = np.zeros((T, self.N))
            xi = np.zeros((T-1, self.N, self.N))

            for t in range(T):
                gamma[t] = (alpha[t] * beta[t]) / P_O

            for t in range(T-1):
                denom = P_O
                for i in range(self.N):
                    for j in range(self.N):
                        xi[t, i, j] = (
                            alpha[t, i]
                            * self.A[i, j]
                            * self.B[j, O[t+1]]
                            * beta[t+1, j]
                        ) / denom

            # Update pi
            self.pi = gamma[0]

            # Update A
            for i in range(self.N):
                for j in range(self.N):
                    self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

            # Update B
            for j in range(self.N):
                for k in range(self.M):
                    mask = (O == k)
                    self.B[j, k] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])

        return P_O


# =============================
# Example Run
# =============================

if __name__ == "__main__":
    # Example observation sequence
    # Walk=0, Shop=1
    observations = np.array([0, 1, 1, 0, 1])

    n_states = int(input("Enter number of hidden states: "))
    n_observations = len(set(observations))

    model = HMM(n_states, n_observations)

    P_O = model.baum_welch(observations, iterations=20)

    print("\nProbability P(O | lambda):")
    print(P_O)

    print("\nTransition Matrix A:")
    print(model.A)

    print("\nEmission Matrix B:")
    print(model.B)

    print("\nInitial Distribution pi:")
    print(model.pi)