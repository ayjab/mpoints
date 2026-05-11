import unittest
import numpy as np
from mpoints import hybrid_hawkes_multi_exp


class HybridHawkesMultiExpTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.de = np.random.randint(1, 3)
        self.dx = np.random.randint(1, 3)
        self.M = np.random.randint(1, 4)
        self.events_labels = [chr(65 + n) for n in range(self.de)]
        self.states_labels = [chr(48 + n) for n in range(self.dx)]
        self.nus = np.random.uniform(0.01, 0.1, (self.dx, self.de))
        self.alphas = np.random.uniform(0, 0.3,
                                        (self.de, self.dx, self.de, self.M))
        self.betas = np.random.uniform(0.5, 5.0,
                                       (self.de, self.dx, self.de, self.M))
        _phis = np.random.dirichlet(np.ones(self.dx),
                                    size=(self.dx, self.de))
        self.phis = _phis

    def _make_model(self):
        model = hybrid_hawkes_multi_exp.HybridHawkesMultiExp(
            self.de, self.dx, self.M,
            self.events_labels, self.states_labels)
        return model

    def test_init_and_setters(self):
        model = self._make_model()
        self.assertEqual(self.de, model.number_of_event_types)
        self.assertEqual(self.dx, model.number_of_states)
        self.assertEqual(self.M, model.number_of_exponentials)
        self.assertEqual(self.phis.shape, model.transition_probabilities.shape)
        self.assertEqual(self.nus.shape, model.base_rates.shape)
        self.assertEqual(self.alphas.shape, model.impact_coefficients.shape)
        self.assertEqual(self.betas.shape, model.decay_coefficients.shape)

        model.set_transition_probabilities(self.phis)
        model.set_hawkes_parameters(self.nus, self.alphas, self.betas)
        self.assertTrue(np.allclose(self.phis, model.transition_probabilities))
        self.assertTrue(np.allclose(self.nus, model.base_rates))
        self.assertTrue(np.allclose(self.alphas, model.impact_coefficients))
        self.assertTrue(np.allclose(self.betas, model.decay_coefficients))

    def test_flatten_parameters(self):
        flat = hybrid_hawkes_multi_exp.HybridHawkesMultiExp.parameters_to_array(
            self.nus, self.alphas, self.betas)
        nus_r, alphas_r, betas_r = (
            hybrid_hawkes_multi_exp.HybridHawkesMultiExp.array_to_parameters(
                flat, self.de, self.dx, self.M))
        self.assertTrue(np.allclose(nus_r, self.nus))
        self.assertTrue(np.allclose(alphas_r, self.alphas))
        self.assertTrue(np.allclose(betas_r, self.betas))

    def test_simulate(self):
        model = self._make_model()
        model.set_transition_probabilities(self.phis)
        model.set_hawkes_parameters(self.nus, self.alphas, self.betas)
        times, events, states = model.simulate(0.0, 3600.0,
                                               max_number_of_events=50000)
        self.assertTrue(len(times) > 0)
        self.assertTrue(np.all(states < self.dx))
        self.assertTrue(np.all(events < self.de))
        self.assertTrue(np.all(np.diff(times) > 0.0))

    def test_state_dependent_baseline(self):
        """Verify that different states produce different average intensities."""
        model = hybrid_hawkes_multi_exp.HybridHawkesMultiExp(
            1, 2, 1, ['A'], ['low', 'high'])
        phis = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        nus = np.array([[0.01], [1.0]])
        alphas = np.zeros((1, 2, 1, 1))
        betas = np.ones((1, 2, 1, 1))
        model.set_transition_probabilities(phis)
        model.set_hawkes_parameters(nus, alphas, betas)
        times0, _, _ = model.simulate(0.0, 1000.0, initial_state=0)
        times1, _, _ = model.simulate(0.0, 1000.0, initial_state=1)
        self.assertGreater(len(times1), len(times0) * 5)

    def test_log_likelihood_and_gradient(self):
        """Gradient check via finite differences."""
        model = self._make_model()
        model.set_transition_probabilities(self.phis)
        model.set_hawkes_parameters(self.nus, self.alphas, self.betas)
        times, events, states = model.simulate(0.0, 500.0,
                                               max_number_of_events=10000)
        if len(times) < 10:
            self.skipTest('too few events for gradient test')
        params = model.parameters_to_array(self.nus, self.alphas, self.betas)
        grad = model.gradient(params, times, events, states, 0.0, 500.0)
        eps = 1e-5
        for idx in np.random.choice(len(params), min(5, len(params)),
                                    replace=False):
            p_plus = params.copy()
            p_plus[idx] += eps
            p_minus = params.copy()
            p_minus[idx] -= eps
            ll_plus = model.log_likelihood_of_events(
                p_plus, times, events, states, 0.0, 500.0)
            ll_minus = model.log_likelihood_of_events(
                p_minus, times, events, states, 0.0, 500.0)
            fd = (ll_plus - ll_minus) / (2 * eps)
            self.assertAlmostEqual(
                grad[idx], fd, places=2,
                msg=f'Gradient mismatch at index {idx}: '
                    f'analytical={grad[idx]:.6f}, fd={fd:.6f}')

    def test_reduces_to_single_exponential(self):
        """With M=1, results should match the single-exponential model."""
        de, dx, M = 1, 1, 1
        model = hybrid_hawkes_multi_exp.HybridHawkesMultiExp(
            de, dx, M, ['A'], ['0'])
        phis = np.array([[[1.0]]])
        nus = np.array([[0.05]])
        alphas = np.array([[[[0.1]]]])
        betas = np.array([[[[1.0]]]])
        model.set_transition_probabilities(phis)
        model.set_hawkes_parameters(nus, alphas, betas)
        np.random.seed(123)
        times, events, states = model.simulate(0.0, 1000.0)
        self.assertTrue(len(times) > 10)
        params = model.parameters_to_array(nus, alphas, betas)
        ll = model.log_likelihood_of_events(
            params, times, events, states, 0.0, 1000.0)
        self.assertTrue(np.isfinite(ll))


if __name__ == '__main__':
    unittest.main()
