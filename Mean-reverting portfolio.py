from Functions import *
from SimulationandModelisation import *
import hurst

# 0
# Initialisation of data
SpreadSeries = spread_From_Excel("spread.xlsx")
show_graphs = True
# 1

# As we may see after changing coefficients in the following initial state -
# our final results doesn't change
# that good, yet, not sufficient, indicator
initial_state = np.array([0.5, np.mean(SpreadSeries) + 3, np.std(SpreadSeries) + 2])
# Minimize negative log-likelihood
result = minimize(ornstein_uhlenback_mle, initial_state, args = SpreadSeries)
alpha, k, sigma = result.x
# https://www.sciencedirect.com/science/article/pii/S030440760800208X
# by the following link we may find the explicit formula,
# but I decided to go with numerical approach as I remain unsure about validity of provided formulas
# and their proof seem non-obvious and computation heavy
print(result.x)

# 2
# We simulate using  method
OE_simulation = ornstein_uhlenback_simulation(SpreadSeries[0], len(SpreadSeries), (alpha, k, sigma), show_graph = show_graphs)

"""
y = np.linspace(0, 1000, len(SpreadSeries))
plt.plot(y, SpreadSeries, "blue")
plt.show()
"""

# 3
# As we may see - first two order moments match

# Though it wasn't asked I also checked hurst exponent to ensure both processes are mean-reverting
# Values in both cases are indeed under 0.5

print("hurst exponent of given series/simulation " + str(hurst.compute_Hc(SpreadSeries)[0]) +
      "/" + str(str(hurst.compute_Hc(OE_simulation)[0])))
print()

# As we may see - first 2 moments are almost identical
print("mean of given series/simulation " + str(series_Moments(SpreadSeries)[0]) +
      "/" + str(series_Moments(OE_simulation)[0]))
print()
print("variance of given series/simulation " + str(series_Moments(SpreadSeries)[1]) +
      "/" + str(series_Moments(OE_simulation)[1]))
print()
print("excess kurtosis of given series/simulation " + str(series_Moments(SpreadSeries)[2]) +
      "/" + str(series_Moments(OE_simulation)[2]))
print()
print("skewness of given series/simulation " + str(series_Moments(SpreadSeries)[3]) +
      "/" + str(series_Moments(OE_simulation)[3]))
print()


# 4
print("mean of given increment series/simulation " + str(series_Increment_Moments(SpreadSeries)[0]) +
      "/" + str(series_Increment_Moments(OE_simulation)[0]))
print()
print("variance of given increment series/simulation " + str(series_Increment_Moments(SpreadSeries)[1]) +
      "/" + str(series_Increment_Moments(OE_simulation)[1]))
print()

# 5

# I decided to consider different volatility functions and try to optimise over them,
# I considered polynomial one as exponential seems to be uncommon for mean-reverting processes
# according to the few sources I found over internet
bounds = ((None, None), (None, None), (sigma - 5, sigma + 5), (None, None))

# It converges to the power near to 1
initial_state = np.array([alpha, k, sigma, 1])
# Minimize negative log-likelihood
result = minimize(custom_mle, initial_state, args = SpreadSeries, bounds = bounds)
alpha, k, sigma, power = result.x
print(result.x)

# alpha, k, sigma

Custom_simulation = custom_simulation(SpreadSeries[0], len(SpreadSeries), (alpha, k, sigma, power), show_graph = show_graphs)

# 7

print("hurst exponent of given custom/ue " + str(hurst.compute_Hc(Custom_simulation)[0]) +
      "/" + str(str(hurst.compute_Hc(OE_simulation)[0])))
print()

# As we may see - first 2 moments are almost identical
print("mean of given custom/ue " + str(series_Moments(Custom_simulation)[0]) +
      "/" + str(series_Moments(OE_simulation)[0]))
print()
print("variance of given custom/ue " + str(series_Moments(Custom_simulation)[1]) +
      "/" + str(series_Moments(OE_simulation)[1]))
print()
print("mean of given increment custom/ue " + str(series_Increment_Moments(Custom_simulation)[0]) +
      "/" + str(series_Increment_Moments(OE_simulation)[0]))
print()
print("variance of given increment custom/ue " + str(series_Increment_Moments(Custom_simulation)[1]) +
      "/" + str(series_Increment_Moments(OE_simulation)[1]))


# 8
print("Probability of reaching terminal value: " +
      str(terminal_value_density(SpreadSeries[0], len(SpreadSeries),
                                 23, ornstein_uhlenback_simulation, (alpha, k, sigma),
      number_of_simulations = 500, show_graph = show_graphs)[0]))
print()
# 9

E, prob, _ = expected_threshold_hit(SpreadSeries[0], len(SpreadSeries),
                                 22.5, ornstein_uhlenback_simulation, (alpha, k, sigma),
      number_of_simulations = 500, show_graph = show_graphs)
print("Probability to reach the value: " + str(prob))
print("Expected time to hit the value given that it was reached at some point: " + str(E))
print()

