"""The green gecko code. This belongs to the Colab notebook developed by Mordvintsev et al.
for the ["Growing Neural Cellular Automata"](http://distill.pub/2020/growing-ca) article.

The original code is available here:
https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb.

The code has been modified to be compatible a Mac M1 machine, with tensorflow-macos.

Specifically, the percieve function has been modified to work with tensorflow-macos.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D


def to_rgba(x):
    return x[..., :4]


def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)


def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb


def get_living_mask(x):
    alpha = x[:, :, :, 3:4]
    return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], "SAME") > 0.1


def make_seed(size, channel_n, n=1):
    x = np.zeros([n, size, size, channel_n], np.float32)
    x[:, size // 2, size // 2, 3:] = 1.0
    return x


@tf.function
def make_circle_masks(n, h, w):
    x = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = tf.cast(x * x + y * y < 1.0, tf.float32)
    return mask


class CAModel(tf.keras.Model):

    def __init__(self, channel_n, fire_rate):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        self.dmodel = tf.keras.Sequential(
            [
                Conv2D(128, 1, activation=tf.nn.relu),
                Conv2D(self.channel_n, 1, activation=None, kernel_initializer=tf.zeros_initializer),
            ]
        )

        self(tf.zeros([1, 3, 3, channel_n]))  # dummy call to build the model

    def perceive(self, x, angle=0.0):
        batch, height, width, in_channels = x.shape

        # Identity kernel
        identify = tf.constant([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
        # Sobel kernels
        dx = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32) / 8.0
        dy = tf.transpose(dx)

        # Expand kernels to depthwise shape
        # shape: [3,3,1,1] → will be broadcasted to all input channels
        id_k = identify[:, :, None, None]
        dx_k = dx[:, :, None, None]
        dy_k = dy[:, :, None, None]

        # Repeat kernels across all input channels (channel_multiplier=1)
        id_k = tf.repeat(id_k, in_channels, axis=2)
        dx_k = tf.repeat(dx_k, in_channels, axis=2)
        dy_k = tf.repeat(dy_k, in_channels, axis=2)

        # Apply depthwise convolution
        y_id = tf.nn.depthwise_conv2d(x, id_k, [1, 1, 1, 1], "SAME")
        y_dx = tf.nn.depthwise_conv2d(x, dx_k, [1, 1, 1, 1], "SAME")
        y_dy = tf.nn.depthwise_conv2d(x, dy_k, [1, 1, 1, 1], "SAME")

        # Reshape and interleave channels: from [H,W,in_channels] → [H,W,in_channels*3]
        # Stack along last axis per channel
        y = tf.reshape(tf.stack([y_id, y_dx, y_dy], axis=-1), [batch, height, width, in_channels * 3])

        return y

    @tf.function
    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0, return_fire_mask=False, fire_mask=None):
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.dmodel(y) * step_size

        if fire_rate is None:
            fire_rate = self.fire_rate

        if fire_mask is None:
            fire_mask = tf.random.uniform(tf.shape(x[:, :, :, :1]))

        update_mask = fire_mask <= fire_rate
        x = x + dx * tf.cast(update_mask, tf.float32)

        post_life_mask = get_living_mask(x)

        life_mask = pre_life_mask & post_life_mask
        res = x * tf.cast(life_mask, tf.float32)

        if return_fire_mask:
            return res, fire_mask
        else:
            return res
