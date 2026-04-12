"""Functions for calculating and visualizing the Lyapunov spectrum."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tqdm


def gram_schmidt(u):
    """
    Gram-Schmidt orthogonalization of a list of tensors.

    Parameters
    ----------
    u : list of tensors
        The list of tensors to be orthogonalized.

    Returns
    -------
    v : tensor
        The orthogonalized tensor.
    w_len : tensor
        The length of each tensor in the orthogonalized tensor.
    """
    w0 = u[0]
    w_len = [tf.linalg.norm(w0)]
    v = [w0 / w_len[0]]

    for p in range(1, len(u)):
        w_p = u[p]
        for i in range(p):
            w_p = w_p - tf.tensordot(w_p, v[i], axes=1) * v[i]
        w_len.append(tf.linalg.norm(w_p))
        v.append(w_p / tf.maximum(w_len[p], 1e-12))

    v = tf.stack(v)
    w_len = tf.stack(w_len)

    # Verify that v is orthogonal
    """for i in range(len(v)):
        for j in range(i):
            assert np.abs(np.dot(v[i], v[j])) < 1e-3, np.abs(np.dot(v[i], v[j]))"""

    return v, w_len


def QR(u):
    """
    QR decomposition of a list of tensors.

    Parameters
    ----------
    u : list of tensors
        The list of tensors to be QR-decomposed.

    Returns
    -------
    v : tensor
        The orthogonalized tensor.
    w_len : tensor
        The length of each tensor in the orthogonalized tensor.
    """
    Q, R = tf.linalg.qr(tf.transpose(u))  # u transposed is (N, k), k is always < N, so Q is (N, k), R is (k, k)
    R_diag = tf.linalg.diag_part(R)
    R_signs = tf.where(R_diag >= 0, 1.0, -1.0)
    R_signs = tf.cast(R_signs, tf.float64)
    Q = Q * R_signs
    v = tf.transpose(Q)  # Q is (N, k), so transposed becomes (k, N)

    w_len = tf.abs(tf.linalg.diag_part(R))

    # Verify that v is orthogonal
    """for i in range(len(v)):
        for j in range(i):
            assert np.abs(np.dot(v[i], v[j])) < 1e-3, np.abs(np.dot(v[i], v[j]))"""

    return v, w_len


def convert_to_tensor(x):
    """
    Converts a numpy array or a scalar to a tensorflow tensor.

    Parameters
    ----------
    x : numpy array or scalar
        The input to be converted.

    Returns
    -------
    tf.Tensor
        The converted tensor.
    """
    if type(x) == np.ndarray:
        return tf.convert_to_tensor(x, dtype=tf.float64)
    else:
        return tf.cast(x, tf.float64)


def get_v(k, N, ortho_mode, init_mode):
    """
    Generates k orthogonal vectors of length N.

    Parameters
    ----------
    k : int
        The number of vectors.
    N : int
        The length of each vector.
    ortho_mode : str
        The method to use for orthogonalization. Supported values are "gram_schmidt" and "QR".
    init_mode : str
        The method to use for initializing the vectors. Supported values are "initializer" and "random".

    Returns
    -------
    v : tf.Tensor
        The k orthogonal vectors of length N.
    """
    if init_mode == "initializer":
        #  "If the matrix has fewer rows than columns then the output will have orthogonal rows."
        # -> https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
        v = tf.keras.initializers.Orthogonal()(shape=(k, N))  # rows are vectors
        v = tf.linalg.l2_normalize(v, axis=1)  # normalize each row
    else:
        u = np.random.randn(k, N).astype(np.float64)
        if ortho_mode == "gram_schmidt":
            v, _ = gram_schmidt(u)
        elif ortho_mode == "QR":
            Q, R = tf.linalg.qr(tf.transpose(u))
            R_diag = tf.linalg.diag_part(R)
            R_signs = tf.where(R_diag >= 0, 1.0, -1.0)
            R_signs = tf.cast(R_signs, tf.float64)
            Q = Q * R_signs
            v = tf.transpose(Q)  # (k, N)
        else:
            raise ValueError(f"Unsupported ortho_mode: {ortho_mode}")

    # print(f"The norm of the k={k} vectors is", tf.norm(v, axis=1).numpy())  # Should be 1

    return v


def estimate_lyapunov_spectrum(
    model,
    x_star,
    k=3,
    dt=1.0,
    steps=1000,
    epsilon=1e-4,
    mode="finite_diff",
    ortho_mode="gram_schmidt",
    init_mode="initializer",
    dynamic_epsilon=False,
    warm_up=0,
):
    """
    Estimates the Lyapunov spectrum of a model around a given point.

    Parameters
    ----------
    model : tf.keras.Model
        The model to estimate the Lyapunov spectrum for.
    x_star : tf.Tensor
        The point to estimate the Lyapunov spectrum for.
    k : int, optional
        The number of vectors to use for estimating the Lyapunov spectrum. Defaults to 3.
    dt : float, optional
        The time step to use for estimating the Lyapunov spectrum. Defaults to 1.0.
    steps : int, optional
        The number of steps to use for estimating the Lyapunov spectrum. Defaults to 1000.
    epsilon : float, optional
        The perturbation size to use for estimating the Lyapunov spectrum. Defaults to 1e-4.
    mode : str, optional
        The method to use for estimating the Lyapunov spectrum. Supported values are "finite_diff", "central_diff", and "jacobian_vector". Defaults to "finite_diff".
    ortho_mode : str, optional
        The method to use for orthogonalizing the vectors. Supported values are "gram_schmidt" and "QR". Defaults to "gram_schmidt".
    init_mode : str, optional
        The method to use for initializing the vectors. Supported values are "initializer" and "random". Defaults to "initializer".
    dynamic_epsilon : bool, optional
        Whether to use a dynamic perturbation size based on the norm of the current state. Defaults to False.
    warm_up : int, optional
        The number of steps to use for warming up the orthogonal vectors before estimating the Lyapunov spectrum. Defaults to 0.

    Returns
    -------
    tf.Tensor
        The estimated Lyapunov spectrum.
    list
        The estimated Lyapunov exponents at each step.
    """
    tf.keras.backend.set_floatx("float64")

    p = k
    K = steps

    x_star = convert_to_tensor(x_star)
    x = tf.identity(x_star)

    reshaping = False
    if len(x_star.shape) == 1:
        N = x_star.shape[-1]
    elif len(x_star.shape) == 4:
        H, W, C = x_star.shape[-3:]
        N = H * W * C
        reshaping = True
    else:
        raise ValueError(f"Unsupported shape: {x_star.shape}")

    v = get_v(p, N, ortho_mode, init_mode)

    lyap_sum = tf.zeros(p, dtype=tf.float64)
    lyaps = []

    for step in tqdm.tqdm(range(K + warm_up)):
        if dynamic_epsilon:
            perturb = epsilon * tf.maximum(1.0, tf.norm(x))
        else:
            perturb = epsilon

        x_next = tf.cast(model(x, step_size=dt), tf.float64)

        u = []
        for i in range(p):
            v_i = v[i]
            if reshaping:
                v_i = tf.reshape(v_i, (1, H, W, C))

            if mode == "finite_diff":
                model_output = tf.cast(model(x + v_i * perturb, step_size=dt), tf.float64)
                u_i = (model_output - x_next) / perturb  # estimating the Jacobian vector product

                if reshaping:
                    u_i = tf.reshape(u_i, (N))

                u.append(u_i)
            elif mode == "central_diff":
                model_output_plus = tf.cast(model(x + v_i * perturb, step_size=dt), tf.float64)
                model_output_minus = tf.cast(model(x - v_i * perturb, step_size=dt), tf.float64)
                u_i = (model_output_plus - model_output_minus) / (2 * perturb)  # estimating the Jacobian vector product

                if reshaping:
                    u_i = tf.reshape(u_i, (N))

                u.append(u_i)
            elif mode == "jacobian_vector":
                with tf.autodiff.ForwardAccumulator(primals=x, tangents=v_i) as acc:
                    model_output = tf.cast(model(x, step_size=dt), tf.float64)

                    if reshaping:
                        u.append(tf.reshape(acc.jvp(model_output), (N)))
                    else:
                        u.append(acc.jvp(model_output))

        u = tf.stack(u)

        if ortho_mode == "gram_schmidt":
            v, w_len = gram_schmidt(u)
        elif ortho_mode == "QR":
            v, w_len = QR(u)

        if step >= warm_up:
            lyap_sum += tf.math.log(w_len + 1e-12)
            lyaps.append(lyap_sum / ((step + 1 - warm_up) * dt))

        x = x_next

    res = lyap_sum / ((K) * dt)

    tf.keras.backend.set_floatx("float32")

    return res, lyaps


def plot_lyaps(res, lyaps):
    """
    Plot the Lyapunov exponents over time.

    Parameters
    ----------
    res : list
        The estimated Lyapunov exponents.
    lyaps : list
        The estimated Lyapunov exponents over time.

    Returns
    -------
    None
    """
    plt.figure(figsize=(5, 3))
    plt.xlabel("Timesteps (t)")
    plt.ylabel("Lyapunov exponents (λ)")
    for i, lyap in enumerate(np.array(lyaps).T):
        plt.plot(lyap, label=f"λ{i+1}: " + str(np.round(res[i], 3)))

    plt.legend()
