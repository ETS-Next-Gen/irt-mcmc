import logging
import nirt.simulate.simulate_data as sim
import numpy as np
import unittest


class TestSimulatedData(unittest.TestCase):
    def setUp(self) -> None:
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.WARN, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")

    def test_generate_dichotomous_responses(self):
        np.random.seed(0)

        # Number of persons.
        P = 100
        # Number of items.
        I = 20
        # Number of latent ability dimensions (sub-scales).
        C = 1
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        asym = 0  # 0.25
        discrimination = 1

        x, theta, b, c = sim.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination)

        assert x.shape == (P, I)
        assert theta.shape == (P, C)
        assert b.shape == (I, )
        assert c.shape == (I, )


    def test_generate_continuous_responses(self):
        np.random.seed(0)

        # Number of persons.
        P = 100
        # Number of items.
        I = 20
        # Number of latent ability dimensions (sub-scales).
        C = 1
        # Using 2-PL model with fixed discrimination and no asymptote for all items.
        asym = 0  # 0.25
        discrimination = 1

        x, theta, b, c = sim.generate_dichotomous_responses(P, I, C, asymptote=asym, discrimination=discrimination,
                                                            dichotomous=False)

        assert x.shape == (P, I)
        assert theta.shape == (P, C)
        assert b.shape == (I, )
        assert c.shape == (I, )

    def test_generate_response_times(self):
        np.random.seed(0)

        # Number of persons.
        P = 100
        # Number of items.
        I = 20
        t = sim.generate_response_times(P, I)

        assert t.shape == (P, I)
        assert np.all(t > 0)
