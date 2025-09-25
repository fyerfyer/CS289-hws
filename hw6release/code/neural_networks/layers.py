"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
Modified by: Bill Zheng, Tejas Prabhune, Spring 2025
Website: github.com/WJ2003B, github.com/tejasprabhune
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    eps: float = 1e-8,
    momentum: float = 0.95,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )
    
    elif name == "batchnorm1d":
        return BatchNorm1D(eps=eps, momentum=momentum,)

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict ({"X": None, "Z": None})  # cache for backprop
        self.gradients = OrderedDict({
            "W": np.zeros_like(W),
            "b": np.zeros_like(b)
        })
                                         
        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        W = self.parameters.get('W')
        b = self.parameters.get('b')
        Z = X @ W + b 

        # perform an affine transformation and activation
        Y = self.activation(Z)

        # store information necessary for backprop in `self.cache`
        self.cache['X'] = X
        self.cache['Z'] = Z 
        return Y 

        ### END YOUR CODE ###

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache
        X = self.cache.get('X')
        Z = self.cache.get('Z')
        W = self.parameters.get('W')
        
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer
        dLdZ = self.activation.backward(Z, dLdY)
        dLdW = X.T @ dLdZ
        dLdb = np.sum(dLdZ, axis=0, keepdims=True)
        dLdX = dLdZ @  W.T 

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.
        self.gradients['W'] = dLdW
        self.gradients['b'] = dLdb

        ### END YOUR CODE ###

        return dLdX


class BatchNorm1D(Layer):
    def __init__(
        self, 
        weight_init: str = "xavier_uniform",
        eps: float = 1e-8,
        momentum: float = 0.9,
    ) -> None:
        super().__init__()
        
        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init,)

        self.eps = eps
        self.momentum = momentum

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        gamma = np.zeros((self.n_in, ))
        beta = np.zeros((self.n_in, ))

        self.parameters = OrderedDict({"gamma": gamma, "beta": beta}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"X": None, "X_hat": None, 
                                  "mu": None, "var": None, 
                                  "running_mu": None, "running_var": None})  
        # cache for backprop
        self.gradients = OrderedDict({"gamma": None, "beta": beta})

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray, mode: str = "train") -> np.ndarray:
        """ Forward pass for 1D batch normalization layer.
        Allows taking in an array of shape (B, C) and performs batch normalization over it. 

        We use Exponential Moving Average to update the running mean and variance. with alpha value being equal to self.gamma

        You should set the running mean and running variance to the mean and variance of the first batch after initializing it.
        You should also make separate cases for training mode and testing mode.
        """
        ### BEGIN YOUR CODE ###

        # implement a batch norm forward pass
        if mode == 'train':
            mu = np.mean(X, axis=0)
            sigma = np.var(X, axis=0)
            self.cache['running_mu'] = mu
            self.cache['running_var'] = sigma
        elif mode == 'test':
            mu = self.cache['running_mu']
            sigma = self.cache['running_var']
        else: 
            raise ValueError('invalid mode str')

        X_hat = (X - mu) / np.sqrt(sigma - self.eps)

        # cache any values required for backprop
        self.cache['X'] = X 
        self.cache['X_hat'] = X_hat
        self.chche['mu'] = mu 
        self.cache['sigma'] = sigma

        gamma = self.parameters['gamma']
        beta = self.parameters['beta']

        Y = gamma * X_hat + beta 

        ### END YOUR CODE ###
        return Y 

    def backward(self, dY: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward method for batch normalization layer. You don't need to implement this to get full credit, although it is
        fun to do so if you have the time.
        """

        ### BEGIN YOUR CODE ###
        X = self.cache['X']
        X_hat = self.cache['X_hat']
        mu = self.cache['mu']
        var = self.cache['var']
        gamma = self.parameters['gamma']
        batch_size = X.shape[0]

        # implement backward pass for batchnorm.
        dGammadY = np.sum(dY * X_hat, axis=0)
        dBetadY = np.sum(dY, axis=0)
        dX_hatdY = dY * dGammadY
        dVardY = np.sum(dX_hatdY * (X - mu) * (-0.5) * (var + self.eps)**(-1.5), axis=0)
        dMudY = (np.sum(dX_hatdY * (-1.0 / np.sqrt(var + self.eps)), axis=0) + 
           dVardY * np.sum(-2.0 * (X - mu), axis=0) / batch_size)
        dXdY = (dX_hatdY / np.sqrt(var + self.eps) + 
          dVardY * 2.0 * (X - mu) / batch_size + 
          dMudY / batch_size)
        ### END YOUR CODE ###
        
        return dXdY

class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []}) # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)}) # parameter gradients initialized to zero
                                                                                     # MUST HAVE THE SAME KEYS AS `self.parameters`

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        W = self.parameters["W"]
        b = self.parameters["b"]
        pad_h, pad_w = self.pad
        stride = self.stride

        k_h, k_w, in_channels, _ = W.shape
        n_examples, in_rows, in_cols, _ = X.shape 

        out_rows = (in_rows + 2 * pad_h - k_h) // stride + 1 
        out_cols = (in_cols + 2 * pad_w - k_w) // stride + 1 

        X_padded = np.pad(
            X,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant'
        )

        view_shape = (n_examples, out_rows, out_cols, k_h, k_w, in_channels)
        s_n, s_r, s_c, s_ch = X_padded.strides
        view_strides = (s_n, s_r * stride, s_c * stride, s_r, s_c, s_ch)

        X_windows = np.lib.stride_tricks.as_strided(
            X_padded, 
            shape=view_shape, 
            strides=view_strides
        )

        out = np.einsum('nhwkij,kijo->nhwo', X_windows, W)
        out += b 

        # kernel_height, kernel_width, in_channels, out_channels = W.shape
        # n_examples, in_rows, in_cols, in_channels = X.shape
        # pad_height, pad_width = self.pad
        # stride = self.stride

        # out_row = (in_rows + 2 * pad_height - kernel_height) // stride + 1
        # out_col = (in_cols + 2 * pad_width - kernel_width) // stride + 1
        # out = np.zeros((n_examples, out_row, out_col, out_channels))

        # X_padded = np.pad(
        #     X, 
        #     ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)),
        #     mode="constant",
        # )

        # for n in range(n_examples):
        #     for r in range(out_row):
        #         for c in range(out_col):
        #             r_start = r * stride
        #             r_end = r_start + kernel_height
        #             c_start = c * stride
        #             c_end = c_start + kernel_width

        #             X_patch = X_padded[n, r_start:r_end, c_start:c_end, :]
        #             conv_values = np.sum(X_patch * W, axis=(0, 1, 2))
        #             out[n, r, c, :] = conv_values
        
        # out += b
        self.cache['X'] = X 
        self.cache['Z'] = out
        out = self.activation.forward(out)


        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  gradient of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###

        # perform a backward pass
        # X = self.cache['X']
        # Z = self.cache['Z']
        # W = self.parameters['W']

        # pad_h, pad_w = self.pad
        # stride = self.stride

        # n_examples, in_rows, in_cols, in_channels = X.shape
        # kernal_h, kernal_w, _, out_channels = W.shape
        # _, out_rows, out_cols, _ = dLdY.shape

        # dLdZ = self.activation.backward(Z) * dLdY
        # dLdb = np.sum(dLdZ, axis=(0, 1, 2))
        # self.gradients['b'] = dLdb.reshape(1, -1)

        # dLdW = np.zeros_like(W)
        # dLdX = np.zeros_like(X)
        # X_padded = np.pad(
        #     X, 
        #     ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        #     mode='constant'
        # )

        # dLdX_padded = np.zeros_like(X_padded)

        # for n in range(n_examples):
        #     for r in range(out_rows):
        #         for c in range(out_cols):
        #             r_start = r * stride
        #             r_end = r_start + kernal_h
        #             c_start = c * stride
        #             c_end = c_start + kernal_w

        #             X_patch = X_padded[n, r_start:r_end, ]
        #             dLdZ_slice = dLdZ[n, r, c, :]
        #             dLdW += np.einsum('ijk,l->ijkl', X_patch, dLdZ_slice)

        #             dLdX_patch = np.einsum('ijkl,l->ijk', W, dLdZ_slice)
        #             dLdX_padded[n, r_start:r_end, c_start:c_end, :] += dLdX_patch
        
        # self.gradients['W'] = dLdW
        # if self.pad == (0, 0):
        #     dLdX = dLdX_padded
        # else:
        #     dLdX = dLdX_padded[:, pad_h:-pad_h, pad_w:-pad_w, :]

        X = self.cache['X']
        Z = self.cache['Z']
        W = self.cache['W']
        pad_h, pad_w = self.pad
        stride = self.stride
        n_examples, out_rows, out_cols, out_channels = dLdZ.shape
        k_h, k_w, in_channels = self.kernel_shape

        dLdZ = self.activation.backward(Z, dLdY)
        dLdb = np.sum(dLdZ, axis=(0, 1, 2))
        self.gradients['b'] = dLdb

        X_padded = np.pad(
            X,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant'
        )

        view_shape = (n_examples, out_rows, out_cols, k_h, k_w, in_channels)
        s_n, s_r, s_c, s_ch = X_padded.strides
        view_strides = (s_n, s_r * stride, s_c * stride, s_r, s_c, s_ch)
        X_windows = np.lib.stride_tricks.as_strided(
            X_padded,
            shape=view_shape,
            strides=view_strides
        )

        dLdW = np.einsum('nhwijk,nhwo->ijko', X_windows, dLdZ)
        self.gradients['W'] = dLdW

        W_rot = np.rot90(W, 2, axes=(0, 1))
        W_transposed = W_rot.transpose(0, 1, 3, 2)
        if stride > 1:
            unsampled = np.zeros(
                            (n_examples, 
                            (out_rows - 1) * stride + 1,
                            (out_cols - 1) * stride + 1,
                            out_channels)
                        )
            unsampled[:, ::stride, ::stride, :] = dLdZ
        else:
            unsampled = dLdZ 

        unsampled_padded = np.pad(
            unsampled, 
            ((0,0), (k_h - 1 - pad_h, k_h - 1 - pad_h), (k_w - 1 - pad_w, k_w - 1 - pad_w), (0, 0)),
            mode='constant'
        )

        _, in_rows, in_cols, _ = X.shape 
        dLdZ_view_shape = (n_examples, in_rows, in_cols, k_h, k_w, out_channels)
        u_s_n, u_s_r, u_s_c, u_s_ch = unsampled_padded.strides
        dLdZ_view_strides = (u_s_n, u_s_r, u_s_c, u_s_r, u_s_c, u_s_ch)
        dLdZ_windows = np.lib.stride_tricks.as_strided(
            unsampled_padded,
            shape=dLdZ_view_shape,
            strides=dLdZ_view_strides
        )

        dLdX = np.einsum("nhwijo,ijok->nhwk", dLdZ_windows, W_transposed)
        
        ### END YOUR CODE ###
        return dLdX

class Pool2D(Layer):    
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {}
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # implement the forward pass
        n_examples, in_rows, in_cols, channels = X.shape
        k_h, k_w = self.kernel_shape
        pad_h, pad_w = self.pad
        stride = self.stride

        out_rows = (in_rows + 2 * pad_h - k_h) // stride + 1 
        out_cols = (in_cols + 2 * pad_w - k_w) // stride + 1 

        X_padded = np.pad(
            X,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant'
        )

        view_shape = (n_examples, out_rows, out_cols, k_h, k_w, channels)
        s_n, s_c, s_r, s_ch = X_padded.strides
        view_strides = (s_n, s_r * stride, s_c * stride, s_r, s_c, s_ch)
        X_windows = view_strides = np.lib.stride_tricks.as_strided(
            X_padded, 
            shape=view_shape, 
            strides=view_strides
        )

        # Rearrange to (N, out_H, out_W, C, k_H, k_W)
        X_windows_permuted = X_windows.transpose(0, 1, 2, 5, 3, 4)
        X_pool = self.pool_fn(X_windows_permuted, axis=(4, 5))

        # cache any values required for backprop
        self.cache = {
            "X_padded": X_padded,
            "X_windows": X_windows,   # (N,H_out,W_out,k_h,k_w,C)
            "out_rows": out_rows,
            "out_cols": out_cols,
        }

        ### END YOUR CODE ###

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # perform a backward pass
        X_padded = self.cache["X_pad"]
        X_windows = self.cache["X_windows"]
        out_rows, out_cols = self.cache["out_rows"], self.cache["out_cols"]
        k_h, k_w = self.kernel_shape
        stride = self.stride
        pad_h, pad_w = self.pad
        n_examples, _, _, in_channels = X_padded.shape

        dLdX_padded = np.zeros_like(X_padded)

        if self.mode == 'max':
            mask_vals = np.max(X_windows, axis=(3, 4), keepdims=True)
            mask = np.sum(X_windows == mask_vals).astype(np.float32)
            mask_sum = np.sum(mask, axis=(3, 4), keepdims=True)
            mask /= (mask_sum + 1e-8)

            dLdY_exp = dLdY[:, :, :, None, None, :] # (batch_size, out_rows, out_cols, k_h, k_w, channels)
            grad_windows = mask * dLdY_exp
        elif self.mode == 'average':
            avg_factor = 1.0 / (k_h * k_w)
            dLdY_exp = dLdY[:, :, :, None, None, :] * avg_factor
            grad_windows = np.broadcast_to(dLdY_exp, X_windows.shape)
        
        i_idx = np.arange(out_rows) * stride
        j_idx = np.arange(out_cols) * stride
        di = np.arange(k_h)
        dj = np.arange(k_w)
        ii, dii = np.meshgrid(i_idx, di, indexing='ij')
        jj, djj = np.meshgrid(j_idx, dj, indexing='ij')

        row_idx = ii + dii 
        col_idx = jj + djj

        row_idx = np.reshape(1, out_rows, 1, k_h, 1, 1)
        col_idx = np.reshape(1, 1, out_cols, 1, k_w, 1)

        n_idx = np.arange(n_examples).reshape(n_examples, 1, 1, 1, 1, 1)
        c_idx = np.arange(in_channels).reshape(1, 1, 1, 1, 1, in_channels)

        np.add.at(dLdX_padded, (n_idx, row_idx, col_idx, c_idx), grad_windows)
        if pad_h == 0 and pad_w == 0:
            gradX = dLdX_padded
        else:
            gradX = dLdX_padded[:, pad_h:-pad_h, pad_w:-pad_w, :]

        ### END YOUR CODE ###

        return gradX

class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        gradX = dLdY.reshape(in_dims)
        return gradX
