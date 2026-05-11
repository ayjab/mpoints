from typing import Optional, Sequence
import numpy as np
import math
import copy
import bisect
import scipy.optimize as opt


class HybridHawkesPowerLaw:
    r"""
    State-dependent Hawkes process with **power-law** kernels.

    The intensity of event type :math:`e` is

    .. math::

        \lambda_e(t) = \nu_{x(t^-),\,e}
        + \sum_{t_n < t}
          \kappa_{e_n\, x_n\, e}\,
          \bigl(c_{e_n\, x_n\, e} + t - t_n\bigr)^{-p_{e_n\, x_n\, e}},

    where :math:`\kappa > 0` controls the amplitude, :math:`c > 0` is a
    smoothing constant that prevents the singularity at :math:`t = t_n`,
    and :math:`p > 1` ensures the kernel is integrable.

    The base rates :math:`\nu_{x,e}` are **state-dependent**: each
    (state, event-type) pair has its own baseline arrival rate.

    The integral of the kernel over :math:`[0, T]` is

    .. math::

        \int_0^T \kappa\,(c + s)^{-p}\,ds
        = \frac{\kappa}{p - 1}
          \bigl[c^{-(p-1)} - (c + T)^{-(p-1)}\bigr].

    :param number_of_event_types: :math:`d_e`, the number of event types.
    :param number_of_states: :math:`d_x`, the number of possible states.
    :param events_labels: human-readable names for each event type.
    :param states_labels: human-readable names for each state.
    """

    def __init__(self, number_of_event_types, number_of_states,
                 events_labels, states_labels):
        self.number_of_event_types = number_of_event_types
        self.number_of_states = number_of_states
        self.events_labels = events_labels
        self.states_labels = states_labels
        de = number_of_event_types
        dx = number_of_states
        self.transition_probabilities = np.zeros((dx, de, dx))
        self.base_rates = np.zeros((dx, de))
        self.kappa = np.zeros((de, dx, de))
        self.cutoff = np.ones((de, dx, de))
        self.exponent = 2.0 * np.ones((de, dx, de))

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

    def set_hawkes_parameters(self, base_rates, kappa, cutoff, exponent):
        r"""
        Set :math:`(\nu, \kappa, c, p)`.

        :param base_rates: shape :math:`(d_x, d_e)`.
        :param kappa: amplitude, shape :math:`(d_e, d_x, d_e)`.
        :param cutoff: smoothing constant :math:`c`, shape
            :math:`(d_e, d_x, d_e)`.
        :param exponent: power-law exponent :math:`p > 1`, shape
            :math:`(d_e, d_x, d_e)`.
        """
        de = self.number_of_event_types
        dx = self.number_of_states
        if np.shape(base_rates) != (dx, de):
            raise ValueError('base rates have incorrect shape')
        if np.shape(kappa) != (de, dx, de):
            raise ValueError('kappa has incorrect shape')
        if np.shape(cutoff) != (de, dx, de):
            raise ValueError('cutoff has incorrect shape')
        if np.shape(exponent) != (de, dx, de):
            raise ValueError('exponent has incorrect shape')
        if np.any(exponent <= 1.0):
            raise ValueError('exponent must be > 1 for integrability')
        self.base_rates = copy.copy(base_rates)
        self.kappa = copy.copy(kappa)
        self.cutoff = copy.copy(cutoff)
        self.exponent = copy.copy(exponent)

    # ------------------------------------------------------------------
    # Kernel helpers
    # ------------------------------------------------------------------

    @staticmethod
    def kernel_at_time(dt, kappa_val, c_val, p_val):
        r"""
        Evaluate :math:`\kappa\,(c + \Delta t)^{-p}`.
        """
        return kappa_val * (c_val + dt) ** (-p_val)

    @staticmethod
    def kernel_integral(dt, kappa_val, c_val, p_val):
        r"""
        Evaluate :math:`\int_0^{\Delta t} \kappa\,(c + s)^{-p}\,ds`.
        """
        pm1 = p_val - 1.0
        return (kappa_val / pm1) * (c_val ** (-pm1) - (c_val + dt) ** (-pm1))

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

        Because the power-law kernel is not recursive, each intensity
        evaluation sums over all past events.

        :returns: (times, events, states) arrays including initial condition.
        """
        de = self.number_of_event_types

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
        state = (initial_condition_states[-1] if n_init > 0
                 else initial_state)

        max_size = n_init + max_number_of_events
        result_times = np.zeros(max_size)
        result_events = np.zeros(max_size, dtype=int)
        result_states = np.zeros(max_size, dtype=int)
        result_times[:n_init] = initial_condition_times
        result_events[:n_init] = initial_condition_events
        result_states[:n_init] = initial_condition_states

        def _compute_intensities(t, n_events):
            """Compute intensities at time t given n_events recorded."""
            intensities = np.array(self.base_rates[state])
            for k in range(n_events):
                dt = t - result_times[k]
                if dt <= 0:
                    continue
                e_k = result_events[k]
                x_k = result_states[k]
                for e2 in range(de):
                    intensities[e2] += self.kernel_at_time(
                        dt, self.kappa[e_k, x_k, e2],
                        self.cutoff[e_k, x_k, e2],
                        self.exponent[e_k, x_k, e2])
            return intensities

        time = time_start
        n = n_init
        intensities = _compute_intensities(time, n)
        intensity_max = max(intensities.sum(), 1e-10)

        while time < time_end and n < max_size:
            dt = np.random.exponential(1.0 / intensity_max)
            time += dt
            if time > time_end:
                break

            intensities = _compute_intensities(time, n)
            intensity_total = intensities.sum()

            u = np.random.uniform(0, intensity_max)
            if u < intensity_total:
                event = _random_choice(intensities)
                probs = self.transition_probabilities[state, event, :]
                state = _random_choice(probs)
                result_times[n] = time
                result_events[n] = event
                result_states[n] = state
                n += 1
                intensities = _compute_intensities(time, n)
                intensity_total = intensities.sum()

            intensity_max = max(intensity_total, 1e-10)

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
        base_rates, kappa, cutoff, exponent = self.array_to_parameters(
            parameters, de, dx)

        index_start = bisect.bisect_right(times, time_start)
        N = len(times)

        # Determine starting state
        if N == 0:
            current_state = 0
        elif index_start > 0:
            current_state = states[index_start - 1]
        elif index_start < N:
            current_state = states[index_start]
        else:
            current_state = 0
        if current_state < 0 or current_state >= dx:
            current_state = 0

        # Term 1: sum of log-intensities at event times after time_start
        ll = 0.0
        previous_time = time_start
        for n in range(index_start, N):
            t_n = times[n]
            e_n = events[n]
            x_n = states[n]

            ll -= self._sum_base_rate(
                base_rates, current_state, de) * (t_n - previous_time)

            intensity = base_rates[current_state, e_n]
            for k in range(n):
                dt = t_n - times[k]
                e_k = events[k]
                x_k = states[k]
                intensity += self.kernel_at_time(
                    dt, kappa[e_k, x_k, e_n],
                    cutoff[e_k, x_k, e_n], exponent[e_k, x_k, e_n])
            ll += math.log(intensity)
            previous_time = t_n
            current_state = x_n

        if time_end > previous_time:
            ll -= self._sum_base_rate(
                base_rates, current_state, de) * (time_end - previous_time)

        # Term 2: integral of each kernel contribution
        for n in range(N):
            t_n = times[n]
            e_n = events[n]
            x_n = states[n]
            if t_n <= time_start:
                T_lower = time_start - t_n
                T_upper = time_end - t_n
            else:
                T_lower = 0.0
                T_upper = time_end - t_n
            for e2 in range(de):
                k_val = kappa[e_n, x_n, e2]
                c_val = cutoff[e_n, x_n, e2]
                p_val = exponent[e_n, x_n, e2]
                integral_upper = self.kernel_integral(
                    T_upper, k_val, c_val, p_val)
                integral_lower = self.kernel_integral(
                    T_lower, k_val, c_val, p_val)
                ll -= (integral_upper - integral_lower)

        return ll

    @staticmethod
    def _sum_base_rate(base_rates, state, de):
        s = 0.0
        for e in range(de):
            s += base_rates[state, e]
        return s

    # ------------------------------------------------------------------
    # Gradient (analytical)
    # ------------------------------------------------------------------

    def gradient(self, parameters, times, events, states,
                 time_start, time_end):
        r"""
        Gradient of the log-likelihood w.r.t.
        :math:`(\nu, \kappa, c, p)`.

        :returns: 1-D array with same layout as *parameters*.
        """
        de = self.number_of_event_types
        dx = self.number_of_states
        base_rates, kappa, cutoff, exponent = self.array_to_parameters(
            parameters, de, dx)

        index_start = bisect.bisect_right(times, time_start)
        N = len(times)

        g_nu = np.zeros((dx, de))
        g_kappa = np.zeros((de, dx, de))
        g_cutoff = np.zeros((de, dx, de))
        g_exponent = np.zeros((de, dx, de))

        if N == 0:
            current_state = 0
        elif index_start > 0:
            current_state = states[index_start - 1]
        elif index_start < N:
            current_state = states[index_start]
        else:
            current_state = 0
        if current_state < 0 or current_state >= dx:
            current_state = 0

        # Gradient of log-intensity terms
        state_time = np.zeros(dx)
        previous_time = time_start
        for n in range(index_start, N):
            t_n = times[n]
            e_n = events[n]
            x_n = states[n]
            state_time[current_state] += (t_n - previous_time)

            intensity = base_rates[current_state, e_n]
            for k in range(n):
                dt = t_n - times[k]
                e_k = events[k]
                x_k = states[k]
                intensity += self.kernel_at_time(
                    dt, kappa[e_k, x_k, e_n],
                    cutoff[e_k, x_k, e_n], exponent[e_k, x_k, e_n])

            inv_lam = 1.0 / intensity
            g_nu[current_state, e_n] += inv_lam

            for k in range(n):
                dt = t_n - times[k]
                e_k = events[k]
                x_k = states[k]
                k_v = kappa[e_k, x_k, e_n]
                c_v = cutoff[e_k, x_k, e_n]
                p_v = exponent[e_k, x_k, e_n]
                cpdt = c_v + dt
                cpdt_neg_p = cpdt ** (-p_v)
                g_kappa[e_k, x_k, e_n] += cpdt_neg_p * inv_lam
                g_cutoff[e_k, x_k, e_n] -= (
                    k_v * p_v * cpdt ** (-(p_v + 1))) * inv_lam
                g_exponent[e_k, x_k, e_n] -= (
                    k_v * cpdt_neg_p * math.log(cpdt)) * inv_lam

            previous_time = t_n
            current_state = x_n

        if time_end > previous_time:
            state_time[current_state] += (time_end - previous_time)
        for x in range(dx):
            for e in range(de):
                g_nu[x, e] -= state_time[x]

        # Gradient of integral terms
        for n in range(N):
            t_n = times[n]
            e_n = events[n]
            x_n = states[n]
            if t_n <= time_start:
                T_lower = time_start - t_n
                T_upper = time_end - t_n
            else:
                T_lower = 0.0
                T_upper = time_end - t_n

            for e2 in range(de):
                k_v = kappa[e_n, x_n, e2]
                c_v = cutoff[e_n, x_n, e2]
                p_v = exponent[e_n, x_n, e2]
                pm1 = p_v - 1.0

                cL = c_v + T_lower
                cU = c_v + T_upper
                cL_neg = cL ** (-pm1)
                cU_neg = cU ** (-pm1)

                # d/d kappa of integral
                integral_norm = (cL_neg - cU_neg) / pm1
                g_kappa[e_n, x_n, e2] -= integral_norm

                # d/d c of integral
                g_cutoff[e_n, x_n, e2] -= k_v * (
                    cU ** (-p_v) - cL ** (-p_v))

                # d/d p of integral
                g_exponent[e_n, x_n, e2] -= k_v * (
                    (cU_neg * math.log(cU) - cL_neg * math.log(cL)) / pm1
                    - (cL_neg - cU_neg) / (pm1 ** 2))

        return self.parameters_to_array(g_nu, g_kappa, g_cutoff, g_exponent)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate_hawkes_parameters(self, times, events, states,
                                   time_start, time_end,
                                   maximum_number_of_iterations=2000,
                                   method='L-BFGS-B',
                                   parameters_lower_bound=1e-6,
                                   given_guesses=None,
                                   number_of_random_guesses=1):
        r"""
        Maximum-likelihood estimation of
        :math:`(\nu, \kappa, c, p)`.

        :returns: (OptimizeResult, best_initial_guess, guess_kind).
        """
        if given_guesses is None:
            given_guesses = []
        de = self.number_of_event_types
        dx = self.number_of_states
        guesses = list(given_guesses)

        avg_intensities = np.zeros(de)
        for n in range(len(times)):
            avg_intensities[events[n]] += 1
        avg_intensities /= (time_end - time_start)

        for _ in range(number_of_random_guesses):
            g_nu = np.zeros((dx, de))
            for e in range(de):
                g_nu[:, e] = avg_intensities[e] / 2
            g_kappa = np.random.uniform(0.01, 0.5, (de, dx, de))
            g_cutoff = np.random.uniform(0.01, 1.0, (de, dx, de))
            g_exponent = np.random.uniform(1.5, 3.0, (de, dx, de))
            guesses.append(
                self.parameters_to_array(g_nu, g_kappa, g_cutoff, g_exponent))

        n_nu = dx * de
        n_coeff = de * dx * de
        bounds = []
        for i in range(n_nu):
            bounds.append((parameters_lower_bound, None))
        for i in range(n_coeff):
            bounds.append((parameters_lower_bound, None))
        for i in range(n_coeff):
            bounds.append((parameters_lower_bound, None))
        for i in range(n_coeff):
            bounds.append((1.0 + parameters_lower_bound, None))

        def neg_ll(params):
            return -self.log_likelihood_of_events(
                params, times, events, states, time_start, time_end)

        def neg_grad(params):
            return -self.gradient(
                params, times, events, states, time_start, time_end)

        results = []
        for g in guesses:
            try:
                o = opt.minimize(
                    neg_ll, g, method=method, bounds=bounds, jac=neg_grad,
                    options={'maxiter': maximum_number_of_iterations})
                results.append(o)
            except (ValueError, FloatingPointError):
                pass

        if not results:
            raise RuntimeError('All optimisation attempts failed')
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
    def parameters_to_array(base_rates, kappa, cutoff, exponent):
        r"""
        Flatten :math:`(\nu, \kappa, c, p)` into a 1-D array.
        """
        return np.concatenate([
            base_rates.ravel(),
            kappa.ravel(),
            cutoff.ravel(),
            exponent.ravel(),
        ])

    @staticmethod
    def array_to_parameters(array, number_of_event_types, number_of_states):
        r"""
        Recover :math:`(\nu, \kappa, c, p)` from a 1-D array.
        """
        de = number_of_event_types
        dx = number_of_states
        n_nu = dx * de
        n_coeff = de * dx * de
        base_rates = array[:n_nu].reshape(dx, de)
        kappa = array[n_nu:n_nu + n_coeff].reshape(de, dx, de)
        cutoff = array[n_nu + n_coeff:n_nu + 2 * n_coeff].reshape(de, dx, de)
        exponent = array[n_nu + 2 * n_coeff:n_nu + 3 * n_coeff].reshape(
            de, dx, de)
        return base_rates, kappa, cutoff, exponent


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
