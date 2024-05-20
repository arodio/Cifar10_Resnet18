import time
import numpy as np


class ActivitySimulator:
    r"""Simulates clients activity

    The activity of each client follows a Bernoulli random variable


    Attributes
    ----------

    participation_matrix: 2-D array of size (`n_clients`, n_rounds)
        participation outcomes at every round per client,

    __rng: numpy.random._generator.Generator

    Methods
    -------

    get_active_clients

    """

    def __init__(self, n_clients, n_rounds, participation_probs, rng=None, deterministic_participation=False):
        """

        Parameters
        ----------

        n_clients:

        n_rounds:

        participation_probs : 1-D list (dtype=float)

        rng: numpy.random._generator.Generator

        deterministic_participation: bool, optional

        """
        self.__rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

        clients_per_prob = n_clients // len(participation_probs)

        if deterministic_participation:

            print("Deterministic participation")

            self.participation_matrix = np.zeros((n_clients, n_rounds), dtype=int)
            for i, participation_prob in enumerate(participation_probs):
                interval = max(1, int(1 / participation_prob))
                for client_id in range(i * clients_per_prob, (i + 1) * clients_per_prob):
                    self.participation_matrix[client_id, interval - 1::interval] = 1

        else:

            self.participation_matrix = np.concatenate(
                [self.__rng.binomial(1, participation_prob, size=(clients_per_prob, n_rounds))
                 for participation_prob in participation_probs]
            )

    def get_active_clients(self, c_round):
        """returns indices of active clients (i.e., with participation=1)

        Parameters
        ----------

        c_round:

        Returns
        -------
            * List[int]
        """
        return np.where(self.participation_matrix[:, c_round] == 1)[0].tolist()
