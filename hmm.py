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

    # =========================
    # SCALED FORWARD
    # =========================
    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))
        scale = np.zeros(T)

        alpha[0] = self.pi * self.B[:, O[0]]
        scale[0] = np.sum(alpha[0])
        alpha[0] /= scale[0]

        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]
            scale[t] = np.sum(alpha[t])
            alpha[t] /= scale[t]

        return alpha, scale

    # =========================
    # SCALED BACKWARD
    # =========================
    def backward(self, O, scale):
        T = len(O)
        beta = np.zeros((T, self.N))

        beta[T-1] = 1 / scale[T-1]

        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(
                    self.A[i, :] * self.B[:, O[t+1]] * beta[t+1]
                )
            beta[t] /= scale[t]

        return beta

    # =========================
    # BAUM WELCH WITH SMOOTHING
    # =========================
    def baum_welch(self, O, iterations=20):
        T = len(O)
        epsilon = 1e-8   # smoothing factor

        for _ in range(iterations):
            alpha, scale = self.forward(O)
            beta = self.backward(O, scale)

            gamma = np.zeros((T, self.N))
            xi = np.zeros((T-1, self.N, self.N))

            for t in range(T):
                gamma[t] = alpha[t] * beta[t]
                gamma[t] /= np.sum(gamma[t])

            for t in range(T-1):
                denom = 0
                for i in range(self.N):
                    for j in range(self.N):
                        xi[t, i, j] = (
                            alpha[t, i]
                            * self.A[i, j]
                            * self.B[j, O[t+1]]
                            * beta[t+1, j]
                        )
                        denom += xi[t, i, j]
                xi[t] /= denom

            # Update pi
            self.pi = gamma[0] + epsilon
            self.pi = self.normalize(self.pi)

            # Update A
            for i in range(self.N):
                for j in range(self.N):
                    self.A[i, j] = np.sum(xi[:, i, j]) + epsilon
                self.A[i] = self.normalize(self.A[i])

            # Update B
            for j in range(self.N):
                for k in range(self.M):
                    mask = (O == k)
                    self.B[j, k] = np.sum(gamma[mask, j]) + epsilon
                self.B[j] = self.normalize(self.B[j])

        # Log-likelihood
        log_prob = -np.sum(np.log(scale))
        return log_prob


# =============================
# MAIN PROGRAM
# =============================
if __name__ == "__main__":
    # Example observation sequence
    # 0 and 1 are observation symbols
    observations = np.array([0, 1, 1, 0, 1])

    n_states = int(input("Enter number of hidden states: "))
    n_observations = len(set(observations))

    model = HMM(n_states, n_observations)

    log_prob = model.baum_welch(observations)

    # Clean readable output formatting
    np.set_printoptions(precision=4, suppress=True)

    print("\n==============================")
    print("   HMM Baum-Welch Results")
    print("==============================")

    print("\nLog Probability log P(O | lambda):")
    print(round(log_prob, 4))

    print("\nTransition Matrix A:")
    print(model.A)

    print("\nEmission Matrix B:")
    print(model.B)

    print("\nInitial Distribution pi:")
    print(model.pi)

    print("\n==============================")
