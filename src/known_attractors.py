"""Code for known attractors and attractor visualization."""

import tensorflow as tf
import numpy as np
from plotly import graph_objs as go


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Compute the derivative of the Lorenz system at time t and state.

    Parameters
    ----------
    t : float
        Time at which to compute the derivative.
    state : tf.Tensor
        The state of the system at time t.
    sigma : float, optional
        The Prandtl number of the system. Defaults to 10.0.
    rho : float, optional
        The Rayleigh number of the system. Defaults to 28.0.
    beta : float, optional
        The aspect ratio of the system. Defaults to 8.0 / 3.0.

    Returns
    -------
    deriv : tf.Tensor
        The derivative of the Lorenz system at time t and state.
    """
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return tf.stack([dx, dy, dz], axis=-1)


def lorenz_map(state, step_size=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Compute one step of the Lorenz system given the current state and step size.

    Parameters
    ----------
    state : tf.Tensor
        The current state of the system.
    step_size : float, optional
        The step size of the integration. Defaults to 0.01.
    sigma : float, optional
        The Prandtl number of the system. Defaults to 10.0.
    rho : float, optional
        The Rayleigh number of the system. Defaults to 28.0.
    beta : float, optional
        The aspect ratio of the system. Defaults to 8.0 / 3.0.

    Returns
    -------
    next_state : tf.Tensor
        The state of the system after one step of integration.
    """
    dt = step_size
    deriv = lorenz(0.0, state, sigma, rho, beta)
    return state + dt * deriv


def van_der_pol_map(state, step_size=0.01, mu=1.0, lam=1.0):
    """
    Compute one step of the van der Pol oscillator given the current state and step size.

    Parameters
    ----------
    state : tf.Tensor
        The current state of the system.
    step_size : float, optional
        The step size of the integration. Defaults to 0.01.
    mu : float, optional
        The strength of the non-linearity in the system. Defaults to 1.0.
    lam : float, optional
        The strength of the linear damping in the system. Defaults to 1.0.

    Returns
    -------
    next_state : tf.Tensor
        The state of the system after one step of integration.
    """
    dt = step_size

    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]

    x_next = x + dt * y
    y_next = y + dt * (mu * (1.0 - x**2) * y - x)
    z_next = z + dt * (-lam * z)

    return tf.stack([x_next, y_next, z_next], axis=-1)


def plot_attractors(states_np):
    """
    Plot a 3D visualization of the given states.

    Parameters
    ----------
    states_np : np.ndarray
        The states to plot.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The plotly figure object containing the plotted data.
    """
    timesteps = np.arange(states_np.shape[0])

    fig = go.Figure(
        data=go.Scatter3d(
            x=states_np[:, 0],
            y=states_np[:, 1],
            z=states_np[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=1.0, colorscale="plasma", color=timesteps),
        )
    )

    fig.update_traces(selector=dict(mode="markers"))

    fig.add_scatter3d(
        x=states_np[:, 0],
        y=states_np[:, 1],
        z=states_np[:, 2],
        mode="lines",
        line=dict(colorscale="plasma", color=timesteps),
    )
    fig.update_traces(selector=dict(mode="lines"))

    # Remove axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )

    return fig


def run_map(state, map_function, steps=2000, step_size=0.01):
    """
    Run the given map function from the given state for the given number of steps.

    Parameters
    ----------
    state : tf.Tensor
        The initial state of the system.
    map_function : callable
        The function to map the state to the next one.
    steps : int, optional
        The number of steps to take. Defaults to 2000.
    step_size : float, optional
        The step size of the integration. Defaults to 0.01.

    Returns
    -------
    states_np : np.ndarray
        The states of the system after integration.
    """
    trajectory = []
    for _ in range(steps):
        state = map_function(state, step_size=step_size)
        trajectory.append(state)

    trajectory = tf.stack(trajectory)

    states_np = trajectory.numpy()

    return states_np
