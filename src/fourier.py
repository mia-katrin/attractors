"""Functions for the Fourier analysis."""

from matplotlib import pyplot as plt
import numpy as np


def plot_fourier(raw_attractor, dt=1.0, filename=None, reduce_x_by=None, figure_size=(5, 3)):
    """
    Plot the power spectrum of the given attractor.

    Parameters
    ----------
    raw_attractor : numpy array
        The attractor to plot the power spectrum of.
    dt : float
        The time step used in the attractor.
    filename : str or None
        The filename to save the plot to, or None if the plot should not be saved.
    reduce_x_by : int or None
        The number of data points to reduce the x-axis by, or None if the plot should not be reduced.
    figure_size : tuple of float
        The size of the plot.

    Returns
    -------
    power : numpy array
        The power spectrum of the given attractor.
    freqs : numpy array
        The corresponding frequencies of the power spectrum.
    """

    n_dims = raw_attractor.shape[1]

    L = raw_attractor.shape[0]

    plt.figure(figsize=(5, 3))

    power = 0
    for i in range(n_dims):
        xi = raw_attractor[:, i] - raw_attractor[:, i].mean()  # The whole time series, centered
        fft = np.abs(np.fft.rfft(xi)) ** 2
        plt.plot(fft)
        power += fft
    power /= n_dims

    plt.xlabel("Frequency index")
    plt.ylabel("Power")
    plt.show()

    power = power / power.max()

    plt.figure(figsize=figure_size)  # Other size used in article was (1.5, 3)
    freqs = np.fft.rfftfreq(L, d=dt)
    if reduce_x_by is not None:
        plt.plot(freqs[: L // reduce_x_by], power[: L // reduce_x_by])
    else:
        plt.plot(freqs, power)
    plt.xlabel("Frequency (1/s)")
    plt.ylabel("Power / max(power)")
    plt.semilogy()
    plt.ylim(1e-5, 1)
    if filename is not None:
        plt.savefig(f"{filename}.svg")
    plt.show()

    return power, freqs


def find_peaks(power, k=None):
    """
    Find the local peaks in the given power spectrum.

    Parameters
    ----------
    power : numpy array
        The power spectrum to find the local peaks in.
    k : int or None
        The number of local peaks to return, or None if all local peaks should be returned.

    Returns
    -------
    local_peaks : list of int
        The indices of the local peaks in the given power spectrum.
    """
    local_peaks = []
    for i in range(1, len(power) - 1):
        if power[i] > power[i - 1] and power[i] > power[i + 1]:
            local_peaks.append(i)

    if k is None:
        return local_peaks

    # only return the top k peaks
    power_sorted = sorted(local_peaks, key=lambda x: power[x], reverse=True)

    return power_sorted[:k]
