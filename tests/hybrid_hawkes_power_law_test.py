import unittest
import numpy as np
from mpoints import hybrid_hawkes_power_law


class HybridHawkesPowerLawTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(99)
        self.de = np.random.randint(1, 3)
        self.dx = np.random.randint(1, 3)
        self.events_labels = [chr(65 + n) for n in range(self.de)]
        self.states_labels = [chr(48 + n) for n in range(self.dx)]
        self.nus = np.random.uniform(0.01, 0.1, (self.dx, self.de))
        self.kappa = np.random.uniform(0.01, 0.3, (self.de, self.dx, self.de))
        self.cutoff = np.random.uniform(0.1, 1.0, (self.de, self.dx, self.de))
        self.exponent = np.random.uniform(1.5, 3.0,
                                          (self.de, self.dx, self.de))
        _phis = np.random.dirichlet(np.ones(self.dx),
                                    size=(self.dx, self.de))
        self.phis = _phis

    def _make_model(self):
        model = hybrid_hawkes_power_law.HybridHawkesPowerLaw(
            self.de, self.dx,
            self.events_labels, self.states_labels)
        return model

    def test_init_and_setters(self):
        model = self._make_model()
        self.assertEqual(self.de, model.number_of_event_types)
        self.assertEqual(self.dx, model.number_of_states)
        self.assertEqual(self.phis.shape, model.transition_probabilities.shape)
        self.assertEqual(self.nus.shape, model.base_rates.shape)
        self.assertEqual(self.kappa.shape, model.kappa.shape)
        self.assertEqual(self.cutoff.shape, model.cutoff.shape)
        self.assertEqual(self.exponent.shape, model.exponent.shape)

        model.set_transition_probabilities(self.phis)
        model.set_hawkes_parameters(self.nus, self.kappa,
                                    self.cutoff, self.exponent)
        self.assertTrue(np.allclose(self.nus, model.base_rates))
        self.assertTrue(np.allclose(self.kappa, model.kappa))
        self.assertTrue(np.allclose(self.cutoff, model.cutoff))
        self.assertTrue(np.allclose(self.exponent, model.exponent))

    def test_exponent_validation(self):
        model = self._make_model()
        bad_exp = np.ones((self.de, self.dx, self.de)) * 0.5
        with self.assertRaises(ValueError):
            model.set_hawkes_parameters(self.nus, self.kappa,
                                        self.cutoff, bad_exp)

    def test_flatten_parameters(self):
        flat = hybrid_hawkes_power_law.HybridHawkesPowerLaw.parameters_to_array(
            self.nus, self.kappa, self.cutoff, self.exponent)
        nus_r, kappa_r, cutoff_r, exp_r = (
            hybrid_hawkes_power_law.HybridHawkesPowerLaw.array_to_parameters(
                flat, self.de, self.dx))
        self.assertTrue(np.allclose(nus_r, self.nus))
        self.assertTrue(np.allclose(kappa_r, self.kappa))
        self.assertTrue(np.allclose(cutoff_r, self.cutoff))
        self.assertTrue(np.allclose(exp_r, self.exponent))

    def test_kernel_at_time(self):
        k = hybrid_hawkes_power_law.HybridHawkesPowerLaw.kernel_at_time
        self.assertAlmostEqual(k(0.0, 1.0, 1.0, 2.0), 1.0)
        self.assertAlmostEqual(k(1.0, 1.0, 1.0, 2.0), 0.25)

    def test_kernel_integral(self):
        ki = hybrid_hawkes_power_law.HybridHawkesPowerLaw.kernel_integral
        val = ki(10.0, 1.0, 1.0, 2.0)
        expected = (1.0 / 1.0) * (1.0 ** (-1.0) - 11.0 ** (-1.0))
        self.assertAlmostEqual(val, expected, places=10)

    def test_simulate(self):
        model = self._make_model()
        model.set_transition_probabilities(self.phis)
        model.set_hawkes_parameters(self.nus, self.kappa,
                                    self.cutoff, self.exponent)
        times, events, states = model.simulate(0.0, 1000.0,
                                               max_number_of_events=50000)
        self.assertTrue(len(times) > 0)
        self.assertTrue(np.all(states < self.dx))
        self.assertTrue(np.all(events < self.de))
        self.assertTrue(np.all(np.diff(times) > 0.0))

    def test_state_dependent_baseline(self):
        """Verify that different states produce different average intensities."""
        model = hybrid_hawkes_power_law.HybridHawkesPowerLaw(
            1, 2, ['A'], ['low', 'high'])
        phis = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        nus = np.array([[0.01], [1.0]])
        kappa = np.zeros((1, 2, 1))
        cutoff = np.ones((1, 2, 1))
        exponent = 2.0 * np.ones((1, 2, 1))
        model.set_transition_probabilities(phis)
        model.set_hawkes_parameters(nus, kappa, cutoff, exponent)
        times0, _, _ = model.simulate(0.0, 1000.0, initial_state=0)
        times1, _, _ = model.simulate(0.0, 1000.0, initial_state=1)
        self.assertGreater(len(times1), len(times0) * 5)

    def test_log_likelihood_and_gradient(self):
        """Gradient check via finite differences."""
        model = self._make_model()
        model.set_transition_probabilities(self.phis)
        model.set_hawkes_parameters(self.nus, self.kappa,
                                    self.cutoff, self.exponent)
        times, events, states = model.simulate(0.0, 200.0,
                                               max_number_of_events=5000)
        if len(times) < 10:
            self.skipTest('too few events for gradient test')
        params = model.parameters_to_array(self.nus, self.kappa,
                                           self.cutoff, self.exponent)
        grad = model.gradient(params, times, events, states, 0.0, 200.0)
        eps = 1e-5
        for idx in np.random.choice(len(params), min(5, len(params)),
                                    replace=False):
            p_plus = params.copy()
            p_plus[idx] += eps
            p_minus = params.copy()
            p_minus[idx] -= eps
            ll_plus = model.log_likelihood_of_events(
                p_plus, times, events, states, 0.0, 200.0)
            ll_minus = model.log_likelihood_of_events(
                p_minus, times, events, states, 0.0, 200.0)
            fd = (ll_plus - ll_minus) / (2 * eps)
            self.assertAlmostEqual(
                grad[idx], fd, places=2,
                msg=f'Gradient mismatch at index {idx}: '
                    f'analytical={grad[idx]:.6f}, fd={fd:.6f}')


if __name__ == '__main__':
    unittest.main()
