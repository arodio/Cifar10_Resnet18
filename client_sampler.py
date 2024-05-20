import time
import numpy as np

from abc import ABC, abstractmethod


class ClientsSampler(ABC):
    r"""Base class for clients sampler

    Attributes
    ----------

    clients_weights_dict: Dict[int: float]
        maps clients ids to their corresponding weight/importance in the objective function

    participation_dict: : Dict[int: float]
        maps clients ids to their corresponding participation probabilities

    activity_simulator: ActivitySimulator

    _time_step: int
        tracks the number of steps

    Methods
    ----------
    __init__

    sample_clients

    step

    """

    def __init__(
            self,
            clients,
            participation_probs,
            activity_simulator,
            *args,
            **kwargs
    ):
        """

        Parameters
        ----------
        activity_simulator: ActivitySimulator

        clients_weights_dict: Dict[int: float]

        """

        n_clients = len(clients)

        self.clients_weights_dict = self.get_client_weights_dict(clients)

        self.participation_dict = self.get_participation_dict(n_clients, participation_probs)

        self.activity_simulator = activity_simulator

        self._time_step = -1

    @staticmethod
    def get_client_weights_dict(clients):
        """compute client weights as a proportion of training samples

        Parameters
        ----------
        clients : list

        Returns
        -------
        dict : key is client_id and value is client_weight.
        """
        clients_weights = np.array([client.n_train_samples for client in clients])
        clients_weights = clients_weights / clients_weights.sum()

        return {client.id: weight for client, weight in zip(clients, clients_weights)}

    @staticmethod
    def get_participation_dict(n_clients, participation_probs):
        """return a dictionary mapping client_id to participation_prob

        Parameters
        ----------
        n_clients : int
        participation_probs : list

        Returns
        -------
        dict : key is client_id and value is participation_prob
        """
        client_probs = np.repeat(participation_probs, n_clients // len(participation_probs))
        return dict(enumerate(np.tile(client_probs, n_clients // len(client_probs))))

    def get_active_clients(self, c_round):
        """receive the list of active clients

        Parameters
        ----------

        c_round:

        Returns
        -------
            * List[int]
        """
        return self.activity_simulator.get_active_clients(c_round)

    def step(self):
        """update the internal step of the clients sampler

        Parameters
        ----------

        Returns
        -------
            None
        """

        self._time_step += 1

    @abstractmethod
    def sample(self, active_clients):
        """sample clients

        Parameters
        ----------
        active_clients: List[int]

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * List[float]: weights to be associated to the sampled clients_dict
        """
        pass


class UnbiasedClientsSampler(ClientsSampler):
    """
    Samples all active clients with aggregation weight inversely proportional to their participation prob
    """

    def sample(self, active_clients):
        """implementation of the abstract method ClientSampler.sample for the UnbiasedClientSampler

        Parameters
        ----------
        active_clients: List[int]

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * List[float]: weights to be associated to the sampled clients_dict
        """
        sampled_clients_ids, sampled_clients_weights = [], []

        for client_id in active_clients:
            sampled_clients_ids.append(client_id)

            sampled_clients_weights.append(
                self.clients_weights_dict[client_id] / self.participation_dict[client_id]
            )

        self.step()

        return sampled_clients_ids, sampled_clients_weights
