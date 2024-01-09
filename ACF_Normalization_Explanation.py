#!/usr/bin/env python
# coding: utf-8

# ***WHY DO WE NORMALIZE AN AUTOCORRELATION FUNCTION IN TIME SERIES ANALYSIS***

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# Generate two time series with different scales
np.random.seed(42)
time_series_1 = np.cumsum(np.random.normal(size=100))
time_series_2 = 10 * np.cumsum(np.random.normal(size=100))

# Calculate autocorrelation without normalization
acf_unnormalized_1 = np.correlate(time_series_1, time_series_1, mode='full')
acf_unnormalized_2 = np.correlate(time_series_2, time_series_2, mode='full')

# Calculate autocorrelation with normalization
acf_normalized_1 = np.correlate(time_series_1, time_series_1, mode='full') / (np.std(time_series_1) ** 2)
acf_normalized_2 = np.correlate(time_series_2, time_series_2, mode='full') / (np.std(time_series_2) ** 2)

# Plot the results
lags = np.arange(-99, 100)
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(lags, acf_unnormalized_1)
plt.title('ACF Without Normalization (Series 1)')
plt.xlabel('Lag')

plt.subplot(2, 2, 2)
plt.plot(lags, acf_normalized_1)
plt.title('ACF With Normalization (Series 1)')
plt.xlabel('Lag')

plt.subplot(2, 2, 3)
plt.plot(lags, acf_unnormalized_2)
plt.title('ACF Without Normalization (Series 2)')
plt.xlabel('Lag')

plt.subplot(2, 2, 4)
plt.plot(lags, acf_normalized_2)
plt.title('ACF With Normalization (Series 2)')
plt.xlabel('Lag')

plt.tight_layout()
plt.show()


# In this example:
# 
# time_series_1 has a standard normal distribution, and time_series_2 has the same distribution but scaled by a factor of 10.
# We calculate the autocorrelation without normalization (acf_unnormalized) and with normalization (acf_normalized) for both time series.
# In the plots, you'll observe the impact of normalization on the scale of the autocorrelation values. The ACF with normalization is consistent and comparable across different scales, whereas the ACF without normalization reflects the impact of the scale of the time series.
# 
# ***INTERPRETATION OF GRAPH***
# 
# Yes, in the context of time series analysis, a lag can be negative. The concept of lag refers to the displacement or shift of a time series in relation to itself. A positive lag means shifting the series forward in time, and a negative lag means shifting the series backward in time.
# 
# Here's how it works:
# 
# Positive Lag (k>0):
# 
# A positive lag of k means that you are looking at the correlation between the original time series and its values k time units into the future.
# 
# Negative Lag (k<0):
# 
# A negative lag of k means that you are looking at the correlation between the original time series and its values ∣k∣ time units into the past.
# For example:
# 
# A positive lag of 1 (k=1) means you are correlating the time series with its values one time unit into the future.
# A negative lag of -1 (k=−1) means you are correlating the time series with its values one time unit into the past.
# In the autocorrelation function (ACF), both positive and negative lags are considered to examine how current values in the time series are correlated with past and future values. It helps in understanding the temporal dependencies and patterns within the data.
# 
# The autocorrelation function is typically symmetrical around lag 0, where the series is correlated with itself at the same time point. Positive lags represent future correlations, and negative lags represent past correlations. The magnitude of the autocorrelation at each lag indicates the strength and direction of the relationship.
# The symmetry of the autocorrelation function (ACF) around lag 0 is a result of the inherent time-reversibility (Time reversibility is a concept in physics and mathematics that refers to the ability of a system or process to behave in the same way whether time is moving forward or backward. In the context of time series analysis, time reversibility implies that the statistical properties and patterns of a process are the same when observed in the reverse direction of time.) of the process being analyzed. This symmetry is a characteristic of stationary time series, where statistical properties do not change over time.

# ***RELATIONSHIP BETWEEN MARKOV PROPERTY AND TIME-SERIE***
# 
# The Markov property states that the future state of a system depends only on its current state, given its entire past history. In other words, the future is conditionally independent of the past, given the present. This means that, in a Markov process, knowing the entire past history doesn't provide any additional information about the future beyond what is already known from the current state. The system "forgets" its past and depends only on its current state to determine future states.
# 
# In the context of time series, the Markov property allows for simplifications in modeling and analysis, as predictions about future states can be made based solely on the current state. Many common time series models, including autoregressive (AR) models and certain state-space models, assume or approximate the Markov property.
# 
# The Markov property is particularly useful when modeling processes where the future behavior depends primarily on the current state and is relatively independent of the past, given that current state. This assumption simplifies the modeling process and allows for the development of computationally efficient models.
# 
# Let's break it down here:
# 
# Markov Property:
# 
# Imagine you have a sequence of events, like flipping a coin several times.
# The Markov property says that the outcome of the next flip depends only on the current flip, not on all the flips that came before it.
# In Time Series:
# 
# In time series analysis, we're looking at sequences of data over time (like stock prices or weather readings).
# The Markov property simplifies things. It says that to predict what happens next, you only need to know what's happening right now. You don't need to remember everything that happened before.
# Example:
# 
# Suppose you're predicting tomorrow's weather.
# If the process follows the Markov property, your prediction depends only on today's weather, not on the entire history of past weather.
# Why It Matters:
# 
# Assuming the Markov property makes modeling and predictions simpler. Many common models, like autoregressive models, assume or approximate this property.
# However, not all processes follow this rule. Some time series have long-term patterns that the Markov property doesn't capture well.
# Summary:
# 
# The Markov property is like saying, "What happens next depends mainly on what's happening right now, not on everything that's happened in the past." It's a simplifying assumption that's often useful but may not always fit every situation.

# *Consistent Interpretation:*
# 
# With normalization, the interpretation of the autocorrelation values is consistent across different scales. A normalized autocorrelation value of 1 indicates a perfect positive autocorrelation, -1 indicates a perfect negative autocorrelation, and 0 indicates no autocorrelation. This consistent interpretation simplifies the analysis and communication of results.
# 
# *Comparable Patterns Across Time Series:*
# 
# Normalization ensures that autocorrelation patterns are comparable across different time series. Without normalization, differences in scale could make it challenging to assess whether similar autocorrelation patterns exist at different lags. Normalization allows for a meaningful comparison of the strength and direction of autocorrelation.
# 
# *Statistical Significance Testing:*
# 
# When performing statistical tests related to autocorrelation, normalization is crucial for obtaining reliable results. Normalized autocorrelation values provide a standardized basis for hypothesis testing, making it easier to assess whether autocorrelation at specific lags is statistically significant.
# 
# *Ease of Communication:*
# 
# Normalization simplifies the communication of autocorrelation results. Normalized values are on a standardized scale, making it easier to convey the strength and direction of autocorrelation to others, even if they are not familiar with the specific scales of the original time series.
# 
# *Consistency Across Lags:*
# 
# Normalization ensures that autocorrelation values at different lags are consistent and comparable. This is particularly important when analyzing the persistence of patterns over time. Without normalization, the magnitude of autocorrelation might vary simply due to differences in the variances of the time series at different lags.
# In summary, normalization of the autocorrelation function provides a standardized and consistent framework for interpreting and comparing autocorrelation results. It addresses issues related to differences in scale, facilitating a more meaningful analysis of the temporal relationships within time series data.
# 
# **Normalization of the autocorrelation function (ACF) is not intended to introduce biases; rather, it is a standard practice aimed at removing the influence of scale and ensuring consistent interpretation. However, it's essential to understand the potential sources of bias in statistical analyses, including those related to autocorrelation:**
# 
# *Scale-Dependent Bias:*
# 
# Autocorrelation values are sensitive to the scale of the time series. If you have time series data with varying scales, unnormalized autocorrelation values may reflect differences in scale rather than true underlying relationships. Normalization helps mitigate this scale-dependent bias.
# 
# *Sample Size Bias:*
# 
# The autocorrelation function is influenced by the sample size. In practice, small sample sizes may lead to less reliable estimates of autocorrelation, and statistical tests for autocorrelation may have reduced power. Normalization doesn't directly address sample size bias but is important to ensure that the interpretation is consistent when comparing autocorrelation patterns across different samples.
# 
# *Non-Stationarity:*
# 
# Non-stationarity in time series data, where statistical properties change over time, can introduce bias in autocorrelation estimates. Normalization does not address non-stationarity directly. In such cases, transformations or differencing might be applied to make the series stationary before autocorrelation analysis.
# 
# *Selection Bias:*
# 
# The choice of lag values for autocorrelation analysis can influence results. Selecting lags based on data exploration without proper hypothesis testing can introduce bias. It's important to use statistical tests to determine the significance of autocorrelation at specific lags.
# 
# *Model Specification Bias:*
# 
# Autocorrelation analysis assumes a particular model structure. If the chosen model is not appropriate for the underlying data, it may introduce bias in autocorrelation estimates. Model diagnostics and sensitivity analyses are essential to address model specification bias.
