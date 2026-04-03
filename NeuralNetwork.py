#!/usr/bin/env python3

import os
import csv
import copy
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from dataclasses import dataclass


class NeuralNetwork:

    class LayerStrategies(Enum):
        CONSTANT_WIDTH = 1
        LINEAR_TAPER_FUNNEL = 2
        LINEAR_TAPER_FUNNEL_OUTPUT = 3
        GEOMETRIC_TAPER = 4
        EXPANSION_COMPRESSION = 5
        BOTTLENECK_HOURGLASS = 6
        POWER_OF_TWO = 7
        REVERSE_POWER_OF_TWO = 8
        PARAMETER_BUDGET = 9
    
    class StartWidthHeuristics(Enum):
        INPUT_WIDTH = 1
        CAPPED_INPUT_WIDTH = 2
        OUTPUT_AWARE = 3
    
    class WeightInitStrategy(Enum):
        XAVIER_NORMAL = 1
        XAVIER_UNIFORM = 2
        HE_NORMAL = 3
        HE_UNIFORM = 4
        LECUN_NORMAL = 5
        LECUN_UNIFORM = 6
        ZERO = 7
    
    class BiasInitStrategy(Enum):
        ZERO = 1
        CONSTANT = 2
        NORMAL = 3
        UNIFORM = 4

    class LossType(Enum):
        MSE = 1
        MULTI_CLASS_CROSS_ENTROPY = 2
        BINARY_CROSS_ENTROPY = 3
    
    class DataAugmentation(Enum):
        JITTER_NOISE = 1
        SAME_CLASS_INTERPOLATION = 2
        MEASUREMENT_NOISE = 3
    
    class HiddenActivationType(Enum):
        SIGMOID = 1
        RELU = 2
        LEAKY_RELU = 3
        TANH = 4
    
    class OutputActivationType(Enum):
        SIGMOID = 1
        SOFTMAX = 2
    
    class LearningDecayType(Enum):
        STEP_DECAY = 1
        INVERSE_DECAY = 2
        EXPONENTIAL_DECAY = 3
    
    class Datasets(Enum):
        NPZ = 1
        JSON = 2
        PICKLE = 3
    
    @dataclass(frozen=True)
    class TrainDefaults:
        learning_rate: float = 1e-3
        epochs: int = 100
        reg: float = 1e-4
        epsilon: float = 1e-12
        step: int = 10
        drop_out_rate: float = 0.03
        l2_lambda: float = 0.03
        batch_size: int = 4
        patience: int = 100
        decay_factor: float = 0.5
        seed: int = 42
        start_width_heuristic_cap: int = 512
        output_aware_multiplier: int = 4
        expansion_multiplier: int = 2
    
    @dataclass
    class TrainResults:
        losses: list = None
        val_losses: list = None
        best_val_loss: float = 0.0
        val_accuracies: list = None
        best_epoch: int = 0
        final_loss: float = 0.0
        accuracy: float = 0.0
        final_learning_rate: float = 0.0
        figure_title: str = "Training Results"
    
    @dataclass
    class Paths:
        model_path: str = "models/"
        csv_path: str = "data/csv/"
        npz_path: str = "data/npz/"
        json_path: str = "data/json/"
        pickle_path: str = "data/pickle/"
        default_data: str = "Untitled"
    
    @dataclass
    class Extensions:
        csv: str = ".csv"
        npz: str = ".npz"
        json: str = ".json"
        pickle: str = ".pkl"
    
    @dataclass
    class Encodings:
        UTF_8: str = "utf_8"

    def __init__(self, number_of_features, layers, loss_type=None, output_activation_type=None, hidden_activation_type=None, hidden_weight_init_strategy=None, output_weight_init_strategy=None, bias_init_strategy=None, init_seed=None, init_random_range=None, hidden_bias_value=0.0, output_positive_prior=None):
        if type(number_of_features) is not int:
            raise TypeError("number_of_features must be an integer")
        if number_of_features < 1:
            raise ValueError("number_of_features must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for nodes in layers:
            if type(nodes) is not int or nodes < 1:
                raise TypeError("layers must be a list of positive integers")
        
        if loss_type is None:
            match output_activation_type:
                case self.OutputActivationType.SIGMOID:
                    loss_type = self.LossType.BINARY_CROSS_ENTROPY
                case self.OutputActivationType.SOFTMAX:
                    loss_type = self.LossType.MULTI_CLASS_CROSS_ENTROPY
                case _:
                    raise ValueError(f"Unknown Output Activation Type, supported values are : {self.OutputActivationType.SIGMOID}, {self.OutputActivationType.SOFTMAX}")
        if loss_type not in self.LossType:
            raise ValueError(f"loss_type must be {self.LossType.MSE}, {self.LossType.BINARY_CROSS_ENTROPY}, {self.LossType.MULTI_CLASS_CROSS_ENTROPY}")
        if output_activation_type is self.OutputActivationType.SIGMOID:
            allowed_losses = {
                self.LossType.BINARY_CROSS_ENTROPY,
                self.LossType.MSE
            }
        elif output_activation_type is self.OutputActivationType.SOFTMAX:
            allowed_losses = {
                self.LossType.MULTI_CLASS_CROSS_ENTROPY,
                self.LossType.MSE
            }
        else:
            raise ValueError(
                f"Unknown Output Activation Type, supported values are: "
                f"{self.OutputActivationType.SIGMOID}, {self.OutputActivationType.SOFTMAX}"
            )

        if loss_type not in allowed_losses:
            raise ValueError(
                f"Incompatible loss_type {loss_type} for output_activation_type "
                f"{output_activation_type}"
            )

        if output_activation_type not in self.OutputActivationType:
            raise ValueError(f"output_activation_type must be {self.OutputActivationType.SIGMOID}, {self.OutputActivationType.SOFTMAX}")
        if hidden_activation_type not in self.HiddenActivationType:
            raise ValueError(f"hidden_activation_type must be {self.HiddenActivationType.SIGMOID}, {self.HiddenActivationType.RELU}, {self.HiddenActivationType.LEAKY_RELU}, {self.HiddenActivationType.TANH}")
        
        if hidden_weight_init_strategy is not None and hidden_weight_init_strategy not in self.WeightInitStrategy:
            raise ValueError(f"hidden_weight_init_strategy must be one of {list(self.WeightInitStrategy)}")

        if output_weight_init_strategy is not None and output_weight_init_strategy not in self.WeightInitStrategy:
            raise ValueError(f"output_weight_init_strategy must be one of {list(self.WeightInitStrategy)}")

        if hidden_weight_init_strategy is None:
            hidden_weight_init_strategy = self.default_hidden_weight_init_strategy(hidden_activation_type)

        if output_weight_init_strategy is None:
            output_weight_init_strategy = self.default_output_weight_init_strategy(output_activation_type)

        if bias_init_strategy is None:
            bias_init_strategy = self.BiasInitStrategy.ZERO
        elif bias_init_strategy not in self.BiasInitStrategy:
            raise ValueError(f"bias_init_strategy must be one of {list(self.BiasInitStrategy)}")

        if hidden_bias_value is None:
            hidden_bias_value = 0.0
        if not isinstance(hidden_bias_value, (int, float, np.floating)):
            raise TypeError("hidden_bias_value must be a float")
        hidden_bias_value = float(hidden_bias_value)
        
        if init_seed is not None and init_random_range is not None:
            raise ValueError("Provide either init_seed or init_rng, not both")
        if init_random_range is not None and not isinstance(init_random_range, np.random.Generator):
            raise TypeError("init_rng must be a numpy.random.Generator")

        if output_positive_prior is not None:
            if not isinstance(output_positive_prior, (int, float, np.floating)):
                raise TypeError("output_positive_prior must be a float")
            output_positive_prior = float(output_positive_prior)
            if not (0.0 < output_positive_prior < 1.0):
                raise ValueError("output_positive_prior must be strictly between 0 and 1")
    
        self.__init_random_range = init_random_range if init_random_range is not None else np.random.default_rng(init_seed)
        self.__L = len(layers)
        self.__cache = []
        self.__WB = []
        self.__loss_type = loss_type
        self.__output_activation_type = output_activation_type
        self.__hidden_activation_type = hidden_activation_type

        for i in range(self.__L):
            fan_in = number_of_features if i == 0 else layers[i - 1]
            fan_out = layers[i]

            current_weight_init_strategy = (
                output_weight_init_strategy if i == self.__L - 1
                else hidden_weight_init_strategy
            )

            current_alpha = 0.0
            if i != self.__L - 1 and self.__hidden_activation_type == self.HiddenActivationType.LEAKY_RELU:
                current_alpha = 0.01

            random_weights = self.init_weights(
                fan_out,
                fan_in,
                current_weight_init_strategy,
                rng=self.__init_random_range,
                alpha=current_alpha
            )

            random_biases = self.init_biases(
                fan_out,
                bias_init_strategy,
                rng=self.__init_random_range,
                is_output_layer=(i == self.__L - 1),
                hidden_activation_type=self.__hidden_activation_type,
                output_activation_type=self.__output_activation_type,
                hidden_bias_value=hidden_bias_value,
                output_positive_prior=output_positive_prior
            )

            self.__WB.append((random_weights, random_biases))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def WB(self):
        return self.__WB

    @property
    def output_activation_type(self):
        return self.__output_activation_type

    @property
    def hidden_activation_type(self):
        return self.__hidden_activation_type
    
    @classmethod
    def init_layers(cls, X, number_of_hidden_layers, output_width=None, start_width=None, hidden_width=None, parameter_budget=None, layer_strategy=None, start_width_heurist=None, start_width_heuristic_cap=512, output_aware_multiplier=4, expansion_multiplier=2):
        if layer_strategy is None:
            layer_strategy = cls.LayerStrategies.GEOMETRIC_TAPER
        if start_width_heurist is None:
            start_width_heurist = cls.StartWidthHeuristics.OUTPUT_AWARE

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")
        if X.ndim != 2:
            raise ValueError("X must be a 2D feature-first matrix")
        if not isinstance(number_of_hidden_layers, int):
            raise TypeError("number_of_hidden_layers must be an integer")
        if number_of_hidden_layers < 0:
            raise ValueError("number_of_hidden_layers must be >= 0")
        if output_width is None or not isinstance(output_width, int):
            raise TypeError("output_dim must be an integer")
        if output_width < 1:
            raise ValueError("output_dim must be a positive integer")

        input_width = X.shape[1]

        if start_width is None:
            match start_width_heurist:
                case cls.StartWidthHeuristics.INPUT_WIDTH:
                    start_width = input_width
                case cls.StartWidthHeuristics.CAPPED_INPUT_WIDTH:
                    start_width = min(start_width_heuristic_cap, input_width)
                case cls.StartWidthHeuristics.OUTPUT_AWARE:
                    start_width = max(output_width * 4, min(start_width_heuristic_cap, input_width))
                case _:
                    raise ValueError(f"Unknown Start Width Heuristic, supported values are : {cls.StartWidthHeuristics.INPUT_WIDTH}, {cls.StartWidthHeuristics.CAPPED_INPUT_WIDTH}, {cls.StartWidthHeuristics.OUTPUT_AWARE}")
        elif not isinstance(start_width, int):
            raise TypeError("start_width must be an integer")
        elif start_width < 1:
            raise ValueError("start_width must be a positive integer")

        if number_of_hidden_layers == 0:
            return [output_width]
        
        if layer_strategy == cls.LayerStrategies.LINEAR_TAPER_FUNNEL:
            if hidden_width is None:
                hidden_width = max(output_width * 2, start_width // 2)
            elif not isinstance(hidden_width, int):
                raise TypeError("hidden_width must be an integer")
            elif hidden_width < 1:
                raise ValueError("hidden_width must be a positive integer")

        if layer_strategy == cls.LayerStrategies.PARAMETER_BUDGET:
            if parameter_budget is None:
                raise TypeError("parameter_budget must be provided for PARAMETER_BUDGET")
            if not isinstance(parameter_budget, int):
                raise TypeError("parameter_budget must be an integer")
            if parameter_budget < 1:
                raise ValueError("parameter_budget must be a positive integer")

        match layer_strategy:
            case cls.LayerStrategies.CONSTANT_WIDTH:
                hidden = [start_width] * number_of_hidden_layers

            case cls.LayerStrategies.LINEAR_TAPER_FUNNEL:
                hidden = [
                    max(
                        1,
                        int(round(
                            start_width + (l / (number_of_hidden_layers + 1)) * (hidden_width - start_width)
                        ))
                    )
                    for l in range(1, number_of_hidden_layers + 1)
                ]
            
            case cls.LayerStrategies.LINEAR_TAPER_FUNNEL_OUTPUT:
                hidden = [
                    max(
                        1,
                        int(round(
                            start_width + (l / (number_of_hidden_layers + 1)) * (output_width - start_width)
                        ))
                    )
                    for l in range(1, number_of_hidden_layers + 1)
                ]

            case cls.LayerStrategies.GEOMETRIC_TAPER:
                ratio = (output_width / start_width) ** (1 / (number_of_hidden_layers + 1))
                hidden = [
                    max(1, int(round(start_width * (ratio ** l))))
                    for l in range(1, number_of_hidden_layers + 1)
                ]
            
            case cls.LayerStrategies.EXPANSION_COMPRESSION:
                peak_width = max(start_width, input_width * expansion_multiplier)
                if number_of_hidden_layers == 1:
                    layers = [peak_width]
                else:
                    down_ratio = (output_width / peak_width) ** (1 / number_of_hidden_layers)
                    layers = [
                        max(1, int(round(peak_width * (down_ratio ** l))))
                        for l in range(number_of_hidden_layers)
                    ]
                layers.append(output_width)
                return layers

            case cls.LayerStrategies.BOTTLENECK_HOURGLASS:
                bottleneck = max(1, min(output_width * 2, start_width // 4))
                if number_of_hidden_layers == 1:
                    layers = [bottleneck]
                else:
                    left_count = number_of_hidden_layers // 2
                    right_count = number_of_hidden_layers - left_count

                    left = []
                    if left_count > 0:
                        left_ratio = (bottleneck / start_width) ** (1 / left_count)
                        left = [
                            max(1, int(round(start_width * (left_ratio ** l))))
                            for l in range(1, left_count + 1)
                        ]

                    right = []
                    if right_count > 0:
                        right_start = left[-1] if left else bottleneck
                        right_ratio = (start_width / right_start) ** (1 / right_count)
                        right = [
                            max(1, int(round(right_start * (right_ratio ** l))))
                            for l in range(1, right_count + 1)
                        ]

                    layers = (left + right)[:number_of_hidden_layers]

                layers.append(output_width)
                return layers

            case cls.LayerStrategies.POWER_OF_TWO:
                first_power = max(1, 2 ** int(np.floor(np.log2(start_width))))
                layers = [
                    max(1, first_power // (2 ** l))
                    for l in range(number_of_hidden_layers)
                ]
                layers.append(output_width)
                return layers

            case cls.LayerStrategies.REVERSE_POWER_OF_TWO:
                base_power = max(1, 2 ** int(np.floor(np.log2(start_width))))
                layers = [
                    max(1, base_power * (2 ** l))
                    for l in range(number_of_hidden_layers)
                ]
                layers.append(output_width)
                return layers

            case cls.LayerStrategies.PARAMETER_BUDGET:
                if parameter_budget < output_width:
                    raise ValueError("parameter_budget must be >= output_dim")

                if number_of_hidden_layers == 1:
                    width = max(
                        1,
                        int((parameter_budget - output_width) / (input_width + output_width + 1))
                    )
                else:
                    a = number_of_hidden_layers - 1
                    b = input_width + output_width + number_of_hidden_layers
                    c = output_width - parameter_budget
                    disc = max(0.0, b * b - 4 * a * c)

                    if a == 0:
                        width = max(1, int(-c / max(1, b)))
                    else:
                        width = max(1, int((-b + np.sqrt(disc)) / (2 * a)))

                layers = [width] * number_of_hidden_layers
                layers.append(output_width)
                return layers

            case _:
                raise ValueError(f"Unknown Layer Strategy, supported values are: {cls.LayerStrategies.CONSTANT_WIDTH}, {cls.LayerStrategies.LINEAR_TAPER_FUNNEL}, {cls.LayerStrategies.LINEAR_TAPER_FUNNEL_OUTPUT}, {cls.LayerStrategies.GEOMETRIC_TAPER}, {cls.LayerStrategies.EXPANSION_COMPRESSION}, {cls.LayerStrategies.BOTTLENECK_HOURGLASS}, {cls.LayerStrategies.POWER_OF_TWO}, {cls.LayerStrategies.REVERSE_POWER_OF_TWO}, {cls.LayerStrategies.PARAMETER_BUDGET}")

        return hidden + [output_width]
    
    @classmethod
    def init_weights(cls, fan_out, fan_in, weight_init_strategy, rng=None, alpha=0.0):
        if rng is None:
            rng = np.random.default_rng()

        if fan_in < 1 or fan_out < 1:
            raise ValueError("fan_in and fan_out must be positive integers")

        match weight_init_strategy:
            case cls.WeightInitStrategy.XAVIER_NORMAL:
                std = np.sqrt(2.0 / (fan_in + fan_out))
                return rng.standard_normal((fan_out, fan_in)) * std

            case cls.WeightInitStrategy.XAVIER_UNIFORM:
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                return rng.uniform(-limit, limit, size=(fan_out, fan_in))

            case cls.WeightInitStrategy.HE_NORMAL:
                std = np.sqrt(2.0 / ((1.0 + alpha ** 2) * fan_in))
                return rng.standard_normal((fan_out, fan_in)) * std

            case cls.WeightInitStrategy.HE_UNIFORM:
                limit = np.sqrt(6.0 / ((1.0 + alpha ** 2) * fan_in))
                return rng.uniform(-limit, limit, size=(fan_out, fan_in))

            case cls.WeightInitStrategy.LECUN_NORMAL:
                std = np.sqrt(1.0 / fan_in)
                return rng.standard_normal((fan_out, fan_in)) * std

            case cls.WeightInitStrategy.LECUN_UNIFORM:
                limit = np.sqrt(3.0 / fan_in)
                return rng.uniform(-limit, limit, size=(fan_out, fan_in))

            case cls.WeightInitStrategy.ZERO:
                return np.zeros((fan_out, fan_in))

            case _:
                raise ValueError(f"Unknown Weight Init Strategy, supported values are : {list(NeuralNetwork.WeightInitStrategy)}")
    
    @staticmethod
    def logit(p, epsilon=1e-12):
        p = float(np.clip(p, epsilon, 1.0 - epsilon))
        return np.log(p / (1.0 - p))

    @classmethod
    def init_biases(cls, fan_out, bias_init_strategy, rng=None, value=0.01, std=1e-3, is_output_layer=False, hidden_activation_type=None, output_activation_type=None, hidden_bias_value=0.0, output_positive_prior=None, epsilon=1e-12):
        if fan_out < 1:
            raise ValueError("fan_out must be a positive integer")
    
        if rng is None:
            rng = np.random.default_rng()
        
        if is_output_layer and output_positive_prior is not None:
            if output_activation_type != cls.OutputActivationType.SIGMOID:
                raise ValueError("output_positive_prior is only supported for SIGMOID output layers")
            if fan_out != 1:
                raise ValueError("output_positive_prior requires fan_out == 1")

            b0 = cls.logit(output_positive_prior, epsilon)
            return np.full((fan_out, 1), b0, dtype=float)
                
        if (not is_output_layer and hidden_activation_type == cls.HiddenActivationType.RELU and hidden_bias_value != 0.0):
            return np.full((fan_out, 1), hidden_bias_value, dtype=float)

        match bias_init_strategy:
            case cls.BiasInitStrategy.ZERO:
                return np.zeros((fan_out, 1))

            case cls.BiasInitStrategy.CONSTANT:
                return np.full((fan_out, 1), value)

            case cls.BiasInitStrategy.NORMAL:
                return rng.standard_normal((fan_out, 1)) * std

            case cls.BiasInitStrategy.UNIFORM:
                return rng.uniform(-value, value, size=(fan_out, 1))

            case _:
                raise ValueError("Unknown Bias Init Strategy")
    
    @staticmethod
    def linear_model(W, X, b):
        return X @ W.T + b.T
    
    @staticmethod
    def sigmoid(z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)

        pos = z >= 0
        neg = ~pos

        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)

        return out
    
    @staticmethod
    def relu(z):
        z = np.asarray(z, dtype=float)
        return np.maximum(0.0, z)

    @staticmethod
    def tanh(z):
        z = np.asarray(z, dtype=float)
        return np.tanh(z)
    
    @staticmethod
    def leaky_relu(z, alpha=0.01):
        z = np.asarray(z, dtype=float)
        return np.where(z > 0, z, alpha * z)
        
    @staticmethod
    def drelu_from_output(a):
        a = np.asarray(a, dtype=float)
        return (a > 0).astype(float)
        
    @staticmethod
    def dsigmoid_from_output(a):
        return a * (1 - a)
    
    @staticmethod
    def dtanh_from_output(a):
        a = np.asarray(a, dtype=float)
        return 1.0 - a ** 2

    @staticmethod
    def dleaky_relu_from_output(a, alpha=0.01):
        a = np.asarray(a, dtype=float)
        return np.where(a > 0, 1.0, alpha)
    
    @staticmethod
    def softmax(Z):
        Z = np.asarray(Z, dtype=float)
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    @staticmethod
    def mse_loss(Y, A):
        m = Y.shape[0]
        return np.sum((A - Y) ** 2) / m

    @staticmethod
    def multi_class_cross_entropy_loss(Y, A, epsilon=1e-12):
        m = Y.shape[0]
        A = np.clip(A, epsilon, 1.0)
        return -np.sum(Y * np.log(A)) / m

    @staticmethod
    def binary_cross_entropy_loss(Y, A, epsilon=1e-12):
        m = Y.shape[0]
        A = np.clip(A, epsilon, 1.0 - epsilon)
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        
    def default_hidden_weight_init_strategy(self, hidden_activation_type):
        match hidden_activation_type:
            case self.HiddenActivationType.RELU | self.HiddenActivationType.LEAKY_RELU:
                return self.WeightInitStrategy.HE_NORMAL
            case self.HiddenActivationType.SIGMOID | self.HiddenActivationType.TANH:
                return self.WeightInitStrategy.XAVIER_NORMAL
            case _:
                raise ValueError(f"Unknown Hidden Activation Type, supported values are : {self.HiddenActivationType.SIGMOID}, {self.HiddenActivationType.RELU}, {self.HiddenActivationType.LEAKY_RELU}, {self.HiddenActivationType.TANH}")

    def default_output_weight_init_strategy(self, output_activation_type):
        match output_activation_type:
            case self.OutputActivationType.SIGMOID | self.OutputActivationType.SOFTMAX:
                return self.WeightInitStrategy.XAVIER_NORMAL
            case _:
                raise ValueError(
                    f"Unknown Output Activation Type, supported values are : {self.OutputActivationType.SIGMOID}, {self.OutputActivationType.SOFTMAX}")

    def hidden_derivative_from_output(self, a, hidden_activation_type=HiddenActivationType.SIGMOID):
        match hidden_activation_type:
            case self.HiddenActivationType.SIGMOID:
                return self.dsigmoid_from_output(a)
            case self.HiddenActivationType.RELU:
                return self.drelu_from_output(a)
            case self.HiddenActivationType.LEAKY_RELU:
                return self.dleaky_relu_from_output(a)
            case self.HiddenActivationType.TANH:
                return self.dtanh_from_output(a)
            case _:
                raise ValueError(f"Unknown Hidden Activation Type, supported values are : {self.HiddenActivationType.SIGMOID}, {self.HiddenActivationType.RELU}, {self.HiddenActivationType.LEAKY_RELU}, {self.HiddenActivationType.TANH}")

    def activation(self, A, hidden_activation_type=HiddenActivationType.SIGMOID):
        match hidden_activation_type:
            case self.HiddenActivationType.SIGMOID:
                return self.sigmoid(A)
            case self.HiddenActivationType.RELU:
                return self.relu(A)
            case self.HiddenActivationType.LEAKY_RELU:
                return self.leaky_relu(A)
            case self.HiddenActivationType.TANH:
                return self.tanh(A)
            case _:
                raise ValueError(f"Unknown Hidden Activation Type, supported values are : {self.HiddenActivationType.SIGMOID}, {self.HiddenActivationType.RELU}, {self.HiddenActivationType.LEAKY_RELU}, {self.HiddenActivationType.TANH}")
    
    def loss(self, Y, A, loss_type=None, epsilon=None):
        if epsilon is None:
            epsilon = self.TrainDefaults().epsilon

        if loss_type == self.LossType.MSE:
            return self.mse_loss(Y, A)
        if loss_type == self.LossType.MULTI_CLASS_CROSS_ENTROPY:
            return self.multi_class_cross_entropy_loss(Y, A, epsilon)
        if loss_type == self.LossType.BINARY_CROSS_ENTROPY:
            return self.binary_cross_entropy_loss(Y, A, epsilon)

        is_binary = (self.__output_activation_type == self.OutputActivationType.SIGMOID and A.shape[0] == 1)

        if is_binary:
            return self.binary_cross_entropy_loss(Y, A, epsilon)

        return self.multi_class_cross_entropy_loss(Y, A, epsilon)

    def output_delta(self, y_hat, y):
        return y_hat - y

    def bernoulli(self, shape, p):
        keep_prob = 1.0 - p
        return (np.random.random(shape) < keep_prob).astype(float)

    def sum_weight_squares(self, WB):
        return sum(np.sum(W * W) for W, _ in WB)

    def forward_pass(self, X, hidden_activation_type=HiddenActivationType.RELU, output_activation_type=OutputActivationType.SIGMOID, cfg=None, training_mode=False):
        drop_out_rate = 0.0
        keep_prob = 0.0
        
        if cfg is None and training_mode:
            cfg = self.TrainDefaults()

            drop_out_rate = cfg.drop_out_rate
            if not (0.0 <= drop_out_rate < 1.0):
                raise ValueError("drop_out_rate must be in [0.0, 1.0)")
            
            keep_prob = 1.0 - drop_out_rate

        self.__cache = [X]
        activation_cache = []
        M = []

        for i in range(self.__L):
            W, b = self.__WB[i]
            A_prev = self.__cache[i]
            Z = self.linear_model(W, A_prev, b)

            if i == self.__L - 1:
                if output_activation_type == self.OutputActivationType.SIGMOID:
                    A_raw = self.sigmoid(Z)
                elif output_activation_type == self.OutputActivationType.SOFTMAX:
                    A_raw = self.softmax(Z)
                else:
                    raise ValueError("output_activation_type must be SIGMOID or SOFTMAX")
                A = A_raw
                M.append(None)
            else:
                A_raw = self.activation(Z, hidden_activation_type)
                if training_mode and drop_out_rate > 0.0:
                    mask = self.bernoulli(A_raw.shape, drop_out_rate)
                    A = (A_raw * mask) / keep_prob
                    M.append(mask)
                else:
                    A = A_raw
                    M.append(None)

            activation_cache.append(A_raw)
            self.__cache.append(A)

        return A, self.__cache, activation_cache, M
    
    def backward_propagation(self, Y, cache, activation_cache, M, hidden_activation_type=HiddenActivationType.RELU, cfg=None, training_mode=False):
        drop_out_rate = 0.0
        keep_prob = 0.0
        
        if training_mode and cfg is None:
            cfg = self.TrainDefaults()
        
        if training_mode and cfg is not None:
            drop_out_rate = cfg.drop_out_rate
            if not (0.0 <= drop_out_rate < 1.0):
                raise ValueError("drop_out_rate must be in [0.0, 1.0)")
            keep_prob = 1.0 - drop_out_rate

        m = Y.shape[0]
        grads = [None] * self.__L
        
        dZ = self.output_delta(cache[-1], Y)

        for l in range(self.__L - 1, -1, -1):
            A_prev = cache[l]
            
            dW = (dZ.T @ A_prev) / m
            db = np.sum(dZ, axis=0, keepdims=True).T / m
            grads[l] = (dW, db)

            if l > 0:
                W_l, _ = self.__WB[l]
                A_l = cache[l]
                dA_prev = dZ @ W_l

                if training_mode and drop_out_rate > 0.0 and M[l - 1] is not None:
                    dA_prev = (dA_prev * M[l - 1]) / keep_prob
                
                A_prev = activation_cache[l - 1]
                dZ = dA_prev * self.hidden_derivative_from_output(A_prev, hidden_activation_type)

        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(self.__L):
            W, b = self.__WB[l]
            dW, db = grads[l]
            self.__WB[l] = (W - learning_rate * dW,
                            b - learning_rate * db)
    
    def step_decay(self, initial_lr, decay_factor, epoch, step):
        return initial_lr * (decay_factor ** (epoch // step))

    def inverse_decay(self, initial_lr, decay_factor, epoch):
        return initial_lr / (1 + decay_factor * epoch)
    
    def exponential_decay(self, initial_lr, decay_factor, epoch):
        return initial_lr * (decay_factor ** epoch)
    
    def learning_decay(self, initial_lr, decay_factor, epoch, step, learning_decay_type=LearningDecayType.STEP_DECAY):
        match learning_decay_type:
            case self.LearningDecayType.STEP_DECAY:
                return self.step_decay(initial_lr, decay_factor, epoch, step)
            case self.LearningDecayType.INVERSE_DECAY:
                return self.inverse_decay(initial_lr, decay_factor, epoch)
            case self.LearningDecayType.EXPONENTIAL_DECAY:
                return self.exponential_decay(initial_lr, decay_factor, epoch)
            case _:
                raise ValueError(f"Unknown Learning Decay Type, supported values are : {self.LearningDecayType.STEP_DECAY}")

    def train(self, X, Y, X_val, Y_val, learning_decay_type=None, data_augmentation_type=None, l2=False, dropout=False, cfg=None, log=True, graph=True, finalize=False):
        if X.ndim != 2:
            raise ValueError("X must be 2D: (samples, features)")
        if Y.ndim != 2:
            raise ValueError("Y must be 2D: (samples, outputs)")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        if X.shape[1] != self.__WB[0][0].shape[1]:
            raise ValueError("X feature count does not match model input size")

        if cfg is None:
            cfg = self.TrainDefaults()
        
        learning_rate = cfg.learning_rate
        epochs = cfg.epochs
        step = cfg.step
        epsilon = cfg.epsilon
        l2_lambda = cfg.l2_lambda
        drop_out_rate = cfg.drop_out_rate
        patience = cfg.patience
        decay_factor = cfg.decay_factor
        batch_size = cfg.batch_size
        seed = cfg.seed

        base_m = X.shape[0]

        best_val_loss = float("inf")
        best_WB = copy.deepcopy(self.__WB)
        patience_counter = 0

        current_lr = learning_rate
        initial_lr = learning_rate

        losses = []
        val_losses = []
        val_accuracies = []
        best_epoch = None
        steps = []

        random_range = np.random.default_rng(seed)

        if type(epochs) is not int:
            raise TypeError("iterations must be an integer")
        if epochs < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(learning_rate, (float, np.floating)):
            raise TypeError("learning_rate must be a float")
        learning_rate = float(learning_rate)
        if learning_rate < 0:
            raise ValueError("learning_rate must be positive")
        if not (0.0 <= drop_out_rate < 1.0):
            raise ValueError("drop_out_rate must be in [0.0, 1.0)")
        if log or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > epochs:
                raise ValueError("step must be positive and <= iterations")
        
        train_data_loss = 0.0
        train_reg_loss = 0.0
        train_total_loss = train_data_loss + train_reg_loss

        for epoch in range(epochs):

            X_shuf, Y_shuf = self.shuffle_dataset(X, Y, base_m, random_range)

            if data_augmentation_type is not None:
                X_epoch, Y_epoch = self.data_augmentation(X_shuf, Y_shuf, data_augmentation_type, random_range)
            else:
                X_epoch, Y_epoch = X_shuf, Y_shuf
            
            epoch_m = X_epoch.shape[0]

            for batch_start in range(0, epoch_m, batch_size):
                end = min(batch_start + batch_size, epoch_m)

                X_batch = X_epoch[batch_start:end, :]
                Y_batch = Y_epoch[batch_start:end, :]

                batch_m = Y_batch.shape[0]

                if dropout:
                    A, cache, activation_cache, M = self.forward_pass(X_batch, hidden_activation_type=self.__hidden_activation_type, output_activation_type=self.__output_activation_type, cfg=cfg, training_mode=True)
                else:
                    A, cache, activation_cache, M = self.forward_pass(X_batch, hidden_activation_type=self.__hidden_activation_type, output_activation_type=self.__output_activation_type, cfg=None, training_mode=False)

                batch_loss = self.loss(Y_batch, A, self.__loss_type, epsilon)
                train_data_loss += batch_loss * (end - batch_start)

                if dropout:
                    grads = self.backward_propagation(Y_batch, cache, activation_cache, M, hidden_activation_type=self.__hidden_activation_type, cfg=cfg, training_mode=True)
                else:
                    grads = self.backward_propagation(Y_batch, cache, activation_cache, M, hidden_activation_type=self.__hidden_activation_type, cfg=None, training_mode=False)
                
                if l2:
                    grads_with_L2 = []
                    for l in range(self.__L):
                        W, b = self.__WB[l]
                        dW, db = grads[l]
                        #dW = dW + l2_lambda * W
                        dW = dW + (l2_lambda / batch_m) * W
                        grads_with_L2.append((dW, db))
    
                    self.update_parameters(grads_with_L2, current_lr)
                else:
                    self.update_parameters(grads, current_lr)
            
            train_data_loss = train_data_loss / epoch_m
            # train_reg_loss = (l2_lambda / 2) * self.sum_weight_squares(self.__WB)
            train_reg_loss = (l2_lambda / (2 * epoch_m)) * self.sum_weight_squares(self.__WB)
            train_total_loss = train_data_loss + train_reg_loss
            _, val_data_loss, val_acc = self.evaluate_dataset(X_val, Y_val)
            val_losses.append(val_data_loss)
            val_accuracies.append(val_acc)

            if epoch >= 0 and epoch % step == 0:
                if learning_decay_type is not None:
                    current_lr = self.learning_decay(initial_lr, decay_factor, epoch, step, learning_decay_type)

                if log:
                    print(
                        "epoch =", epoch,
                        "train_data_loss =", round(train_data_loss, 6),
                        "train_reg_loss =", round(train_reg_loss, 6),
                        "train_total_loss =", round(train_total_loss, 6),
                        "val_data_loss =", round(val_data_loss, 6),
                        "val_acc =", round(val_acc, 4)
                    )

                if graph:
                    losses.append(train_data_loss)
                    steps.append(epoch)

            if val_data_loss < best_val_loss:
                best_val_loss = val_data_loss
                best_WB = copy.deepcopy(self.__WB)
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
 
            if patience_counter >= patience:
                print("early stopping at epoch", epoch)
                break
        
        self.__WB = best_WB if best_WB is not None else self.__WB

        _, final_data_loss, final_accuracy = self.evaluate_dataset(X, Y)
        final_reg_loss = (l2_lambda / (2 * X.shape[0])) * self.sum_weight_squares(self.__WB)
        final_loss = final_data_loss + final_reg_loss

        if graph:
            plt.figure(self.TrainResults.figure_title)
            plt.plot(steps, losses)
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.title("Training Loss")
            plt.show()
        
        if finalize:
            print("\nFinal Predictions : \n")
            predictions, loss, accuracy = self.evaluate_dataset(X, Y)
            
            print("\nFinal Loss : ", loss)
            print("\nFinal Accuracy : ", accuracy)
        
        # return self.TrainResults(losses=losses, val_losses=val_losses, best_val_loss=best_val_loss, val_accuracies=val_accuracies, best_epoch=best_epoch, final_loss=final_loss, accuracy=final_accuracy, final_learning_rate=current_lr)

    def shuffle_dataset(self, X, Y, size, random_range):
        permutation = random_range.permutation(size)
        X_shuf = X[permutation]
        Y_shuf = Y[permutation]
        return X_shuf, Y_shuf
    
    def evaluate_dataset(self, X, Y, epsilon=None, output_activation_type=OutputActivationType.SIGMOID, hidden_activation_type=HiddenActivationType.RELU):
        if epsilon is None:
            epsilon = self.TrainDefaults().epsilon
        
        A, _ = self.predict(X)
        loss = self.loss(Y, A, self.__loss_type, epsilon)

        is_binary = (self.__output_activation_type == self.OutputActivationType.SIGMOID and A.shape[1] == 1)

        if is_binary:
            prediction = (A >= 0.5).astype(float)
            accuracy = np.mean(prediction == Y) * 100.0
        else:
            prediction = np.zeros_like(A)
            prediction[np.arange(A.shape[0]), np.argmax(A, axis=1)] = 1

            predicted_classes = np.argmax(A, axis=1)
            true_classes = np.argmax(Y, axis=1)
            accuracy = np.mean(predicted_classes == true_classes) * 100.0

        return prediction, loss, accuracy
    
    def predict(self, x):
        A, cache, _, _ = self.forward_pass(x, hidden_activation_type=self.__hidden_activation_type, output_activation_type=self.__output_activation_type, training_mode=False)
        return A, cache

    def data_augmentation(self, X, Y, augmentation_type, random_range=None):
        if random_range is None:
            random_range = np.random.default_rng()

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y)

        X_aug = X.copy()
        Y_aug = Y.copy()

        match augmentation_type:
            case self.DataAugmentation.JITTER_NOISE:
                jitter_strength = 0.03
                noise = random_range.uniform(-jitter_strength, jitter_strength, size=X_aug.shape)
                X_aug = X_aug + noise

            case self.DataAugmentation.MEASUREMENT_NOISE:
                measurement_strength = 0.02
                scale = random_range.uniform(
                    1.0 - measurement_strength,
                    1.0 + measurement_strength,
                    size=X_aug.shape
                )
                X_aug = np.maximum(0.0, X_aug * scale)

            case self.DataAugmentation.SAME_CLASS_INTERPOLATION:
                if Y_aug.shape[1] == 1:
                    labels = Y_aug.flatten().astype(int)
                else:
                    labels = np.argmax(Y_aug, axis=1)

                synthetic_x = []
                synthetic_y = []

                for label in np.unique(labels):
                    indices = np.where(labels == label)[0]
                    num_synthetic = len(indices) // 2

                    for _ in range(num_synthetic):
                        if len(indices) >= 2:
                            i1, i2 = random_range.choice(indices, size=2, replace=False)
                        else:
                            i1 = i2 = indices[0]

                        alpha = random_range.uniform(0.2, 0.8)

                        x_new = alpha * X_aug[i1] + (1.0 - alpha) * X_aug[i2]
                        y_new = Y_aug[i1].copy()

                        synthetic_x.append(x_new.reshape(-1, 1))
                        synthetic_y.append(y_new.reshape(-1, 1))

                if synthetic_x:
                    X_aug = np.concatenate([X_aug] + synthetic_x, axis=0)
                    Y_aug = np.concatenate([Y_aug] + synthetic_y, axis=0)

            case _:
                raise ValueError(f"Unknown Data Augmentation Type, supported values are : {self.DataAugmentation.MEASUREMENT_NOISE}, {self.DataAugmentation.JITTER_NOISE}, {self.DataAugmentation.SAME_CLASS_INTERPOLATION}")

        return X_aug, Y_aug
    
    @staticmethod
    def one_hot_encode(Y, classes):
        if type(Y) is not np.ndarray:
            return None
        if type(classes) is not int:
            return None
        try:
            Y = Y.flatten()
            one_hot = np.eye(classes)[Y]
            return one_hot
        except Exception:
            return None

    @staticmethod
    def one_hot_decode(one_hot):
        if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
            return None
        vector = one_hot.argmax(axis=1)
        return vector

    def save_model(self, file_name):
        if not file_name.endswith(".pkl"):
            file_name = file_name + ".pkl"

        modelpath = self.TrainResults.model_path
        if modelpath is not None and not os.path.exists(modelpath):
            os.mkdir(modelpath)

        file_path = modelpath + file_name
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(model_path):
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    
    @classmethod
    def load_from_npz(cls, npz_path):
        lib = np.load(npz_path)
        
        X_train_3D = lib["X_train"]
        Y_train = lib["Y_train"]
        
        X_valid_3D = lib["X_valid"]
        Y_valid = lib["Y_valid"]
        
        X_test_3D = lib["X_test"]
        Y_test = lib["Y_test"]
        
        X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
        X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
        X_test = X_test_3D.reshape((X_test_3D.shape[0], -1))

        X_train /= 255.0
        X_valid /= 255.0
        X_test /= 255.0
        
        return {
            "X_train": X_train,
            "Y_train": Y_train.astype(np.int64),
            "X_valid": X_valid,
            "Y_valid": Y_valid.astype(np.int64),
            "X_test": X_test,
            "Y_test": Y_test.astype(np.int64),
        }
    
    @classmethod
    def save_to_npz(cls, filename=None, compressed=False, **kwargs):
        npz_path = cls.TrainResults.npz_path
        if filename is None:
            filename = npz_path
        
        npz_extension = cls.Extensions.npz
        if not filename.endswith(npz_extension):
            filename = filename + npz_extension
            
        npz_file_path = cls.TrainResults().npz_path
        if npz_file_path is not None and not os.path.exists(npz_file_path):
            os.mkdir(npz_file_path)

        filepath = npz_file_path + filename
        
        if compressed:  
            np.savez_compressed(filepath, **kwargs)
        else:
            np.savez(filepath, **kwargs)
    
    @classmethod
    def save_to_csv(cls, X, Y, filename):
        if filename is None:
            filename = cls.TrainResults().default_data
        if not filename.endswith(cls.Extensions.csv):
            filename = filename + cls.Extensions.csv
        csv_file_path = cls.TrainResults().csv_path
        if csv_file_path is not None and not os.path.exists(csv_file_path):
            os.mkdir(csv_file_path)

        filepath = csv_file_path + filename
        
        n_features = X.shape[1]
        header = [f"f{i}" for i in range(n_features)] + ["label"]
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for x_row, y_value in zip(X, Y):
                writer.writerow(list(map(float, x_row)) + [int(y_value)])
    
    @classmethod
    def load_from_csv(cls, filepath, _newline="", _encoding=None):
        if _encoding is None:
            _encoding = cls.Encodings.UTF_8
        with open(filepath, "r", newline=_newline, encoding=_encoding) as f:
            reader = csv.reader(f)
            header = next(reader)
            
            rows = list(reader)
            
        data = np.array(rows, dtype=np.float32)
        
        X = data[:, :-1]
        Y = data[:, -1].astype(np.int64)
        
        return X, Y

    @classmethod
    def split_to_records(cls, X, Y):
        records = []
        
        for x_row, y_value in zip(X, Y):
            records.append({
                "features": x_row.tolist(),
                "label": int(y_value)
            })
        return records
    
    @classmethod
    def save_to_json(cls, dataset, filename=None, _encoding=None):
        if filename is None:
            filename = cls.TrainResults().default_data 
        if not filename.endswith(cls.Extensions.json):
            filename = filename + cls.Extensions.json
        json_file_path = cls.TrainResults().json_path
        if json_file_path is not None and not os.path.exists(json_file_path):
            os.mkdir(json_file_path)
        
        filepath = json_file_path + filename
        
        if _encoding is None:
            _encoding = cls.Encodings.UTF_8
            
        json_data = {
            "train": cls.split_to_records(dataset["X_train"], dataset["Y_train"]),
            "valid": cls.split_to_records(dataset["X_valid"], dataset["Y_valid"]),
            "test": cls.split_to_records(dataset["X_test"], dataset["Y_test"]),
        }

        with open(filepath, "w", encoding=_encoding) as f:
            json.dump(json_data, f)
    
    @classmethod
    def records_to_split(cls, records):
        X = np.array([r["features"] for r in records], dtype=np.float32)
        Y = np.array([r["labels"] for r in records], dtype=np.int64)
        return X, Y
    
    @classmethod
    def load_from_json(cls, filepath, _encoding=None):
        if _encoding is None:
            _encoding = cls.Encodings.UTF_8
            
        with open(filepath, "r", encoding=_encoding) as f:
            data = json.load(f)
        
        X_train, Y_train = cls.records_to_split(data["train"])
        X_valid, Y_valid = cls.records_to_split(data["valid"])
        X_test, Y_test = cls.records_to_split(data["test"])

        return {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_valid": X_valid,
            "Y_valid": Y_valid,
            "X_test": X_test,
            "Y_test": Y_test,
        }
    
    @classmethod
    def save_to_pickle(cls, dataset, filename=None):
        if filename is None:
            filename = cls.TrainResults().default_data 
        if not filename.endswith(cls.Extensions.pickle):
            filename = filename + cls.Extensions.pickle
        pickle_file_path = cls.TrainResults().pickle_path
        if pickle_file_path is not None and not os.path.exists(pickle_file_path):
            os.mkdir(pickle_file_path)
        
        filepath = pickle_file_path + filename
        
        with open(filepath, "wb") as f:
            pickle.dump(dataset, f)
    
    @classmethod
    def load_from_pickle(cls, filepath):
        with open(filepath, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    @classmethod
    def prepare_datasets(cls, dataset, number_of_classes):
        X_train = dataset["X_train"]
        Y_train = dataset["Y_train"]

        X_valid = dataset["X_valid"]
        Y_valid = dataset["Y_valid"]

        X_test = dataset["X_test"]
        Y_test = dataset["Y_test"]

        Y_train_one_hot = cls.one_hot_encode(Y_train, number_of_classes)
        Y_valid_one_hot = cls.one_hot_encode(Y_valid, number_of_classes)
        Y_test_one_hot = cls.one_hot_encode(Y_test, number_of_classes)

        return {
            "X_train": X_train,
            "Y_train": Y_train_one_hot,
            "X_valid": X_valid,
            "Y_valid": Y_valid_one_hot,
            "X_test": X_test,
            "Y_test": Y_test_one_hot,
        }
    
    @classmethod
    def load_dataset(cls, file_type, file_path):
        match file_type:
            case cls.Datasets.NPZ:
                return cls.load_from_npz(file_path)
            case cls.Datasets.JSON:
                return cls.load_from_json(file_path)
            case cls.Datasets.PICKLE:
                return cls.load_from_pickle(file_path)
            case _:
                raise ValueError(f"Unknown File Type, supported ones are : {cls.Datasets.NPZ}, {cls.Datasets.PICKLE}, {cls.Datasets.JSON}")
    @classmethod
    def load_csv_datasets(cls, folder):
        X_train, Y_train = cls.load_from_csv(f"{folder}/train.csv")
        X_valid, Y_valid = cls.load_from_csv(f"{folder}/valid.csv")
        X_test, Y_test = cls.load_from_csv(f"{folder}/test.csv")

        return {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_valid": X_valid,
            "Y_valid": Y_valid,
            "X_test": X_test,
            "Y_test": Y_test,
        }

if __name__ == "__main__":
    number_of_classes = 10
    dataset = NeuralNetwork.load_from_npz('data/npz/MNIST.npz')
    prepared_dataset = NeuralNetwork.prepare_datasets(dataset, number_of_classes)
    
    X_train = prepared_dataset["X_train"]
    Y_train = prepared_dataset["Y_train"]
    X_valid = prepared_dataset["X_valid"]
    Y_valid = prepared_dataset["Y_valid"]
    X_test = prepared_dataset["X_test"]
    Y_test = prepared_dataset["Y_test"]

    # NeuralNetwork.save_to_csv(X_train, Y_train, "train.csv")
    # NeuralNetwork.save_to_csv(X_valid, Y_valid, "valid.csv")
    # NeuralNetwork.save_to_csv(X_test, Y_test, "test.csv")
    
    # X_train, Y_train = NeuralNetwork.load_from_csv("data/csv/train.csv")
    
    # dataset = NeuralNetwork.load_from_npz("data/npz/MNIST.npz")
    # NeuralNetwork.save_to_json(dataset, "mnist.json")
    
    # dataset = load_npz_mnist("data/MNIST.npz")
    # save_dataset_to_pickle(dataset, "data_pickle/mnist_dataset.pkl")

    # Y_train_one_hot = NeuralNetwork.one_hot_encode(Y_train, 10)
    # Y_valid_one_hot = NeuralNetwork.one_hot_encode(Y_valid, 10)
    # Y_test_one_hot = NeuralNetwork.one_hot_encode(Y_test, 10)

    limit = 100

    X_train_small = X_train[:limit]
    Y_train_small = Y_train[:limit]

    model = NeuralNetwork(
        X_train.shape[1],
        [128, 64, 10],
        loss_type=NeuralNetwork.LossType.MSE,
        output_activation_type=NeuralNetwork.OutputActivationType.SOFTMAX,
        hidden_activation_type=NeuralNetwork.HiddenActivationType.LEAKY_RELU
    )

    results = model.train(X_train_small, Y_train_small, X_valid, Y_valid, finalize=True, l2=True, dropout=True, graph=True)
    model.save("mnist_small")

    loaded_model = NeuralNetwork.load("mnist_small.pkl")
    
    if loaded_model is not None:
        results = loaded_model.evaluate_dataset(X_test, Y_test)
        print(results[1])