from typing import Optional, Sequence
import numpy as np
import math
import copy
import bisect
import scipy.optimize as opt


class HybridHawkesMultiExp:
    r"""
    State-dependent Hawkes process with **multi-exponential** kernels.

    The intensity of event type :math:`e` is

    .. math::

        \lambda_e(t) = \nu_{x(t^-),\,e}
        + \sum_{t_n < t} \sum_{m=1}^{M}
          \alpha_{e_n\, x_n\, e,\, m}\,
          \exp\!\bigl(-\beta_{e_n\, x_n\, e,\, m}\,(t - t_n)\bigr),

    where :math:`M` is the number of exponential components per kernel,
    :math:`x(t^-)` is the state just before time :math:`t`, and each past
    event :math:`(t_n, e_n, x_n)` contributes through every component
    :math:`m`.

    The base rates :math:`\nu_{x,e}` are **state-dependent**: each
    (state, event-type) pair has its own baseline arrival rate.

    :param number_of_event_types: :math:`d_e`, the number of event types.
    :param number_of_states: :math:`d_x`, the number of possible states.
    :param number_of_exponentials: :math:`M`, number of exponential
        components per kernel.
    :param events_labels: human-readable names for each event type.
    :param states_labels: human-readable names for each state.
    """

    def __init__(self, number_of_event_types, number_of_states,
                 number_of_exponentials, events_labels, states_labels):
        self.number_of_event_types = number_of_event_types
        self.number_of_states = number_of_states
        self.number_of_exponentials = number_of_exponentials
        self.events_labels = events_labels
        self.states_labels = states_labels
        de = number_of_event_types
        dx = number_of_states
        M = number_of_exponentials
        self.transition_probabilities = np.zeros((dx, de, dx))
        self.base_rates = np.zeros((dx, de))
        self.impact_coefficients = np.zeros((de, dx, de, M))
        self.decay_coefficients = np.ones((de, dx, de, M))

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_transition_probabilities(self, transition_probabilities):
        r"""
        Set the transition probabilities :math:`\phi`.

        :param transition_probabilities: shape :math:`(d_x, d_e, d_x)`.
        """
        expected = (self.number_of_states, self.number_of_event_types,
                    self.number_of_states)
        if np.shape(transition_probabilities) != expected:
            raise ValueError('transition probabilities have incorrect shape')
        self.transition_probabilities = copy.copy(transition_probabilities)

    def set_hawkes_parameters(self, base_rates, impact_coefficients,
                              decay_coefficients):
        r"""
        Set :math:`(\nu, \alpha, \beta)`.

        :param base_rates: shape :math:`(d_x, d_e)`.
        :param impact_coefficients: shape :math:`(d_e, d_x, d_e, M)`.
        :param decay_coefficients: shape :math:`(d_e, d_x, d_e, M)`.
        """
        de = self.number_of_event_types
        dx = self.number_of_states
        M = self.number_of_exponentials
        if np.shape(base_rates) != (dx, de):
            raise ValueError('base rates have incorrect shape')
        if np.shape(impact_coefficients) != (de, dx, de, M):
            raise ValueError('impact coefficients have incorrect shape')
        if np.shape(decay_coefficients) != (de, dx, de, M):
            raise ValueError('decay coefficients have incorrect shape')
        self.base_rates = copy.copy(base_rates)
        self.impact_coefficients = copy.copy(impact_coefficients)
        self.decay_coefficients = copy.copy(decay_coefficients)

    # ------------------------------------------------------------------
    # Simulation (Ogata thinning)
    # ------------------------------------------------------------------

    def simulate(self,
                 time_start: float,
                 time_end: float,
                 initial_condition_times: Optional[Sequence] = None,
                 initial_condition_events: Optional[Sequence] = None,
                 initial_condition_states: Optional[Sequence] = None,
                 initial_state: int = 0,
                 max_number_of_events: int = 10 ** 6):
        r"""
        Simulate a sample path via Ogata thinning.

        :returns: (times, events, states) arrays including initial condition.
        """
        de = self.number_of_event_types
        dx = self.number_of_states
        M = self.number_of_exponentials
        alphas = self.impact_coefficients
        betas = self.decay_coefficients

        if initial_condition_times is None:
            initial_condition_times = np.array([], dtype=float)
        else:
            initial_condition_times = np.asarray(
                initial_condition_times, dtype=float)
        if initial_condition_events is None:
            initial_condition_events = np.array([], dtype=int)
        else:
            initial_condition_events = np.asarray(
                initial_condition_events, dtype=int)
        if initial_condition_states is None:
            initial_condition_states = np.array([], dtype=int)
        else:
            initial_condition_states = np.asarray(
                initial_condition_states, dtype=int)

        n_init = len(initial_condition_times)
        partial_sums = np.zeros((de, dx, de, M))
        for n in range(n_init):
            t_n = initial_condition_times[n]
            e_n = initial_condition_events[n]
            x_n = initial_condition_states[n]
            dt = time_start - t_n
            for e2 in range(de):
                for m in range(M):
                    partial_sums[e_n, x_n, e2, m] += (
                        alphas[e_n, x_n, e2, m]
                        * math.exp(-betas[e_n, x_n, e2, m] * dt))

        state = (initial_condition_states[-1] if n_init > 0
                 else initial_state)

        intensities = np.zeros(de)
        for e2 in range(de):
            intensities[e2] = self.base_rates[state, e2]
            for e1 in range(de):
                for x in range(dx):
                    for m in range(M):
                        intensities[e2] += partial_sums[e1, x, e2, m]
        intensity_max = intensities.sum()

        max_size = n_init + max_number_of_events
        result_times = np.zeros(max_size)
        result_events = np.zeros(max_size, dtype=int)
        result_states = np.zeros(max_size, dtype=int)
        result_times[:n_init] = initial_condition_times
        result_events[:n_init] = initial_condition_events
        result_states[:n_init] = initial_condition_states

        time = time_start
        n = n_init
        while time < time_end and n < max_size:
            dt = np.random.exponential(1.0 / intensity_max)
            time += dt
            if time > time_end:
                break
            for e1 in range(de):
                for x in range(dx):
                    for e2 in range(de):
                        for m in range(M):
                            partial_sums[e1, x, e2, m] *= math.exp(
                                -betas[e1, x, e2, m] * dt)
            intensity_total = 0.0
            for e2 in range(de):
                intensities[e2] = self.base_rates[state, e2]
                for e1 in range(de):
                    for x in range(dx):
                        for m in range(M):
                            intensities[e2] += partial_sums[e1, x, e2, m]
                intensity_total += intensities[e2]

            u = np.random.uniform(0, intensity_max)
            if u < intensity_total:
                event = _random_choice(intensities)
                probs = self.transition_probabilities[state, event, :]
                state = _random_choice(probs)
                result_times[n] = time
                result_events[n] = event
                result_states[n] = state
                n += 1
                for e2 in range(de):
                    for m in range(M):
                        partial_sums[event, state, e2, m] += (
                            alphas[event, state, e2, m])
                intensity_total = 0.0
                for e2 in range(de):
                    intensities[e2] = self.base_rates[state, e2]
                    for e1 in range(de):
                        for x in range(dx):
                            for m in range(M):
                                intensities[e2] += (
                                    partial_sums[e1, x, e2, m])
                    intensity_total += intensities[e2]
            intensity_max = intensity_total

        return result_times[:n], result_events[:n], result_states[:n]

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def log_likelihood_of_events(self, parameters, times, events, states,
                                 time_start, time_end):
        r"""
        Compute the log-likelihood

        .. math::

            l = \sum_{n:\, t_0 < t_n \le T} \log\lambda_{e_n}(t_n)
                - \sum_e \int_{t_0}^{T} \lambda_e(t)\,dt.

        :param parameters: 1-D array produced by :meth:`parameters_to_array`.
        :returns: scalar log-likelihood.
        """
        de = self.number_of_event_types
        dx = self.number_of_states
        M = self.number_of_exponentials
        base_rates, alphas, betas = self.array_to_parameters(
            parameters, de, dx, M)
        ratios = np.where(betas > 0, alphas / betas, 0.0)

        index_start = bisect.bisect_right(times, time_start)

        partial_sums = np.zeros((de, dx, de, M))
        ll = 0.0
        for n in range(index_start):
            t_n = times[n]
            e_n = events[n]
            x_n = states[n]
            dt0 = time_start - t_n
            dt1 = time_end - t_n
            for e2 in range(de):
                for m in range(M):
                    b = betas[e_n, x_n, e2, m]
                    r = ratios[e_n, x_n, e2, m]
                    partial_sums[e_n, x_n, e2, m] += math.exp(-b * dt0)
                    ll -= r * (math.exp(-b * dt0) - math.exp(-b * dt1))

        for e1 in range(de):
            for x in range(dx):
                for e2 in range(de):
                    for m in range(M):
                        partial_sums[e1, x, e2, m] *= (
                            alphas[e1, x, e2, m])

        base_rate_sums = np.zeros(dx)
        for x in range(dx):
            for e in range(de):
                base_rate_sums[x] += base_rates[x, e]

        if len(times) == 0:
            current_state = 0
        elif index_start > 0:
            current_state = states[index_start - 1]
        elif index_start < len(states):
            current_state = states[index_start]
        else:
            current_state = 0
        if current_state < 0 or current_state >= dx:
            current_state = 0

        previous_time = time_start
        for n in range(index_start, len(times)):
            t_n = times[n]
            e_n = events[n]
            x_n = states[n]
            dt = t_n - previous_time
            ll -= base_rate_sums[current_state] * dt
            for e1 in range(de):
                for x in range(dx):
                    for e2 in range(de):
                        for m in range(M):
                            b = betas[e1, x, e2, m]
                            partial_sums[e1, x, e2, m] *= math.exp(-b * dt)
            intensity = base_rates[current_state, e_n]
            for e1 in range(de):
                for x in range(dx):
                    for m in range(M):
                        intensity += partial_sums[e1, x, e_n, m]
            ll += math.log(intensity)
            for e2 in range(de):
                for m in range(M):
                    partial_sums[e_n, x_n, e2, m] += alphas[e_n, x_n, e2, m]
            previous_time = t_n
            current_state = x_n
            dt_end = time_end - t_n
            for e2 in range(de):
                for m in range(M):
                    b = betas[e_n, x_n, e2, m]
                    r = ratios[e_n, x_n, e2, m]
                    ll -= r * (1.0 - math.exp(-b * dt_end))

        if time_end > previous_time:
            ll -= base_rate_sums[current_state] * (time_end - previous_time)
        return ll

    # ------------------------------------------------------------------
    # Gradient (analytical)
    # ------------------------------------------------------------------

    def gradient(self, parameters, times, events, states,
                 time_start, time_end):
        r"""
        Gradient of the log-likelihood w.r.t.
        :math:`(\nu, \alpha, \beta)`.

        :returns: 1-D array with same layout as *parameters*.
        """
        de = self.number_of_event_types
        dx = self.number_of_states
        M = self.number_of_exponentials
        base_rates, alphas, betas = self.array_to_parameters(
            parameters, de, dx, M)
        ratios = np.where(betas > 0, alphas / betas, 0.0)
        index_start = bisect.bisect_right(times, time_start)

        g_nu = np.zeros((dx, de))
        g_alpha = np.zeros((de, dx, de, M))
        g_beta = np.zeros((de, dx, de, M))
        state_time = np.zeros(dx)

        partial_sums = np.zeros((de, dx, de, M))
        partial_sums_1 = np.zeros((de, dx, de, M))

        for n in range(index_start):
            t_n = times[n]
            e_n = events[n]
            x_n = states[n]
            dt0 = time_start - t_n
            dt1 = time_end - t_n
            for e2 in range(de):
                for m in range(M):
                    b = betas[e_n, x_n, e2, m]
                    r = ratios[e_n, x_n, e2, m]
                    a_val = math.exp(-b * dt0)
                    b_val = math.exp(-b * dt1)
                    partial_sums[e_n, x_n, e2, m] += a_val
                    partial_sums_1[e_n, x_n, e2, m] += a_val * dt0
                    g_alpha[e_n, x_n, e2, m] -= (a_val - b_val) / b
                    g_beta[e_n, x_n, e2, m] -= (
                        r * (dt1 * b_val - dt0 * a_val)
                        - r * (a_val - b_val) / b)

        for e1 in range(de):
            for x in range(dx):
                for e2 in range(de):
                    for m in range(M):
                        a = alphas[e1, x, e2, m]
                        partial_sums[e1, x, e2, m] *= a
                        partial_sums_1[e1, x, e2, m] *= a

        if len(times) == 0:
            current_state = 0
        elif index_start > 0:
            current_state = states[index_start - 1]
        elif index_start < len(states):
            current_state = states[index_start]
        else:
            current_state = 0
        if current_state < 0 or current_state >= dx:
            current_state = 0

        previous_time = time_start
        for n in range(index_start, len(times)):
            t_n = times[n]
            e_n = events[n]
            x_n = states[n]
            dt = t_n - previous_time
            state_time[current_state] += dt
            for e1 in range(de):
                for x in range(dx):
                    for e2 in range(de):
                        for m in range(M):
                            b = betas[e1, x, e2, m]
                            partial_sums_1[e1, x, e2, m] += (
                                dt * partial_sums[e1, x, e2, m])
                            decay = math.exp(-b * dt)
                            partial_sums_1[e1, x, e2, m] *= decay
                            partial_sums[e1, x, e2, m] *= decay

            intensity = base_rates[current_state, e_n]
            for e1 in range(de):
                for x in range(dx):
                    for m in range(M):
                        intensity += partial_sums[e1, x, e_n, m]
            g_nu[current_state, e_n] += 1.0 / intensity
            for e1 in range(de):
                for x in range(dx):
                    for m in range(M):
                        a = alphas[e1, x, e_n, m]
                        if a > 0:
                            g_alpha[e1, x, e_n, m] += (
                                partial_sums[e1, x, e_n, m] / a
                            ) / intensity
                        g_beta[e1, x, e_n, m] -= (
                            partial_sums_1[e1, x, e_n, m] / intensity)

            for e2 in range(de):
                for m in range(M):
                    a = alphas[e_n, x_n, e2, m]
                    partial_sums[e_n, x_n, e2, m] += a

            previous_time = t_n
            current_state = x_n
            dt_end = time_end - t_n
            for e2 in range(de):
                for m in range(M):
                    b = betas[e_n, x_n, e2, m]
                    r = ratios[e_n, x_n, e2, m]
                    c = 1.0 - math.exp(-b * dt_end)
                    g_alpha[e_n, x_n, e2, m] -= c / b
                    g_beta[e_n, x_n, e2, m] -= (
                        r * dt_end * (1.0 - c)
                        - r * c / b)

        if time_end > previous_time:
            state_time[current_state] += time_end - previous_time
        for x in range(dx):
            for e in range(de):
                g_nu[x, e] -= state_time[x]

        return self.parameters_to_array(g_nu, g_alpha, g_beta)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate_hawkes_parameters(self, times, events, states,
                                   time_start, time_end,
                                   maximum_number_of_iterations=2000,
                                   method='TNC',
                                   parameters_lower_bound=1e-6,
                                   parameters_upper_bound=None,
                                   given_guesses=None,
                                   number_of_random_guesses=1,
                                   min_decay_coefficient=0.5,
                                   max_decay_coefficient=100):
        r"""
        Maximum-likelihood estimation of :math:`(\nu, \alpha, \beta)`.

        :returns: (OptimizeResult, best_initial_guess, guess_kind).
        """
        if given_guesses is None:
            given_guesses = []
        de = self.number_of_event_types
        dx = self.number_of_states
        M = self.number_of_exponentials
        guesses = list(given_guesses)

        avg_intensities = np.zeros(de)
        for n in range(len(times)):
            avg_intensities[events[n]] += 1
        avg_intensities /= (time_end - time_start)

        for _ in range(number_of_random_guesses):
            g_nu = np.zeros((dx, de))
            for e in range(de):
                g_nu[:, e] = avg_intensities[e] / 2
            g_beta = np.zeros((de, dx, de, M))
            for e1 in range(de):
                for x in range(dx):
                    for e2 in range(de):
                        for m in range(M):
                            u = np.random.uniform(
                                math.log10(min_decay_coefficient),
                                math.log10(max_decay_coefficient))
                            g_beta[e1, x, e2, m] = 10 ** u
            g_alpha = np.zeros((de, dx, de, M))
            for e1 in range(de):
                for x in range(dx):
                    for e2 in range(de):
                        for m in range(M):
                            g_alpha[e1, x, e2, m] = (
                                np.random.uniform(0, 1)
                                * g_beta[e1, x, e2, m] / M)
            guesses.append(self.parameters_to_array(g_nu, g_alpha, g_beta))

        dim = dx * de + 2 * de * dx * de * M
        bounds = [(parameters_lower_bound, parameters_upper_bound)] * dim

        def neg_ll(params):
            return -self.log_likelihood_of_events(
                params, times, events, states, time_start, time_end)

        def neg_grad(params):
            return -self.gradient(
                params, times, events, states, time_start, time_end)

        results = []
        for g in guesses:
            o = opt.minimize(neg_ll, g, method=method, bounds=bounds,
                             jac=neg_grad,
                             options={'maxiter': maximum_number_of_iterations})
            results.append(o)

        best_idx = int(np.argmin([r.fun for r in results]))
        kind = ('given' if best_idx < len(given_guesses) else 'random')
        return results[best_idx], guesses[best_idx], kind

    # ------------------------------------------------------------------
    # Transition-probability estimation
    # ------------------------------------------------------------------

    def estimate_transition_probabilities(self, events, states):
        r"""
        Empirical (MLE) estimate of the transition probabilities.
        """
        de = self.number_of_event_types
        dx = self.number_of_states
        result = np.zeros((dx, de, dx))
        counts = np.zeros((dx, de))
        for n in range(1, len(events)):
            e = events[n]
            x_before = states[n - 1]
            x_after = states[n]
            counts[x_before, e] += 1
            result[x_before, e, x_after] += 1
        for x1 in range(dx):
            for e in range(de):
                if counts[x1, e] > 0:
                    result[x1, e, :] /= counts[x1, e]
        return result

    # ------------------------------------------------------------------
    # Parameter (de-)serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def parameters_to_array(base_rates, impact_coefficients,
                            decay_coefficients):
        r"""
        Flatten :math:`(\nu, \alpha, \beta)` into a 1-D array.
        """
        return np.concatenate([
            base_rates.ravel(),
            impact_coefficients.ravel(),
            decay_coefficients.ravel(),
        ])

    @staticmethod
    def array_to_parameters(array, number_of_event_types, number_of_states,
                            number_of_exponentials):
        r"""
        Recover :math:`(\nu, \alpha, \beta)` from a 1-D array.
        """
        de = number_of_event_types
        dx = number_of_states
        M = number_of_exponentials
        n_nu = dx * de
        n_coeff = de * dx * de * M
        base_rates = array[:n_nu].reshape(dx, de)
        alphas = array[n_nu:n_nu + n_coeff].reshape(de, dx, de, M)
        betas = array[n_nu + n_coeff:n_nu + 2 * n_coeff].reshape(
            de, dx, de, M)
        return base_rates, alphas, betas


def _random_choice(weights):
    """Weighted random choice returning the selected index."""
    total = weights.sum()
    u = np.random.uniform(0, total)
    cumsum = 0.0
    for i in range(len(weights)):
        cumsum += weights[i]
        if u <= cumsum:
            return i
    return len(weights) - 1
