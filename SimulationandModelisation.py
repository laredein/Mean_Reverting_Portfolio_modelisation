from Functions import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn

# I have taken that as it seems to be common choice for SDE optimisation
def ornstein_uhlenback_mle(params, Series):
    alpha, k, sigma = params
    num_steps = len(Series)
    dt = 1 / (num_steps - 1)
    log_likelihood = -num_steps / 2 * np.log(2 * np.pi * sigma ** 2 * dt)

    for i in range(1, num_steps):
        diff = Series[i] - Series[i - 1] - alpha * (k - Series[i - 1]) * dt
        log_likelihood -= (diff ** 2) / (2 * sigma ** 2 * dt)

    return -log_likelihood


# I have used Oiler–Maruyama method as it is reasonably simple
# and the step size is small enough to afford it(yet It is just my intuition)
# even though it has square root rate of convergence(which is not the best option)
# I also read about Milstein method, but given that sigma is constant and its
# derivative is 0 - they are same for our case
def ornstein_uhlenback_simulation(initial_value, num_periods, params, show_graph = True):
    alpha, k, sigma = params
    dt = 1 / (num_periods - 1)
    path = np.zeros(num_periods) + initial_value
    dW = np.random.normal(0, np.sqrt(dt), num_periods)
    for i in range(1, num_periods):
            path[i] = path[i - 1] + alpha * (k - path[i - 1]) * dt + sigma * dW[i - 1]
    if show_graph:
        y = np.linspace(0, 1000, len(path))
        plt.plot(y, path, "blue")
        plt.show()
    return path
# I also found the library to generate SDE
# https://sdepy.readthedocs.io/en/v1.1.2/generated/sdepy.ornstein_uhlenbeck_process.html
# But decided not to proceed with it
# As it seemed a bit unfair to use it


# taken from the internet
def custom_mle(params, Series):
    alpha, k, sigma, power = params
    num_steps = len(Series)
    dt = 1 / (num_steps - 1)
    log_likelihood = -num_steps / 2 * np.log(2 * np.pi * sigma ** 2 * dt)

    for i in range(1, num_steps):
        diff = Series[i] - Series[i - 1] - alpha * (k - Series[i - 1]) * dt
        log_likelihood -= (diff ** 2) / (2 * sigma ** 2 * dt * abs(k - Series[i - 1]) ** (2 * power))

    return -log_likelihood


def custom_simulation(initial_value, num_periods, params, show_graph = True):
    alpha, k, sigma, power = params
    dt = 1 / (num_periods - 1)
    path = np.zeros(num_periods) + initial_value
    dW = np.random.normal(0, np.sqrt(dt), num_periods)
    for i in range(1, num_periods):
            path[i] = path[i - 1] + alpha * (k - path[i - 1]) * dt + abs(k - path[i - 1]) ** power * sigma * dW[i - 1]

    if show_graph:
        y = np.linspace(0, 1000, len(path))
        plt.plot(y, path, "blue")
        plt.show()
    return path

# for num_steps
def terminal_value_density(initial_value, num_periods, threshold,
                           simulation_function, simulation_data = (),
                           number_of_simulations = 1000, show_graph = True):
    terminal_values = []
    threshold_reached = 0
    for simulation_index in range(number_of_simulations):
        terminal_value = simulation_function(initial_value, num_periods, simulation_data, show_graph = False)[num_periods - 1]
        terminal_values.append(terminal_value)
        if terminal_value >= threshold:
            threshold_reached += 1

    if show_graph:
        seaborn.histplot(terminal_values, kde = True, color = 'skyblue', bins = 25)
        plt.show()

    return threshold_reached / number_of_simulations, np.array(terminal_values)


# Question is how to handle infinities in given time horizon
def expected_threshold_hit(initial_value, num_periods, threshold,
                          simulation_function, simulation_data = (),
                          number_of_simulations = 1000, show_graph = True):
    stopping_times = []
    for simulation_index in range(number_of_simulations):
        path = simulation_function(initial_value, num_periods, simulation_data, show_graph = False)
        # if there is no such time than argmax return 0 so we need to check for that
        stopping_time = np.argmax(path >= threshold)
        if path[stopping_time] >= threshold:
            stopping_times.append(stopping_time)
        else:
            stopping_times.append(np.inf)
    stopping_times = np.array(stopping_times)

    # counting paths which haven't reached threshold at all
    count_inf = np.sum(stopping_times == np.inf)

    # mask to skip infinite elements
    finite_stopping_times_mask = np.isfinite(stopping_times)
    finite_stopping_times = stopping_times[finite_stopping_times_mask]

    if show_graph:
        seaborn.histplot(finite_stopping_times, kde = True, color = 'skyblue', bins = 25)
        plt.show()
    return np.sum(finite_stopping_times) / len(finite_stopping_times), 1 - count_inf / len(stopping_times), stopping_times