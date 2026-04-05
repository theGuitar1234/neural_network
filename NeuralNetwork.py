#!/usr/bin/env python3

import os
import csv
import copy
import json
import pickle
import matplotlib.pyplot as plt

from enum import Enum
from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

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
    
    class Device(Enum):
        CPU = 1
        CUDA = 2
    
    @dataclass
    class TrainDefaults:
        learning_rate: float = 0.03
        epochs: int = 5000
        reg: float = 1e-4
        epsilon: float = 1e-12
        step: int = 100
        threshold: float = 0.5
        drop_out_rate: float = 0.03
        l2_lambda: float = 0.03
        batch_size: int = 512
        patience: int = 100
        decay_factor: float = 0.5
        seed: int = 42
        start_width_heuristic_cap: int = 512
        output_aware_multiplier: int = 4
        expansion_multiplier: int = 2
        prediction_tolerance: int = 100
        prediction_threshold: int = 1000
        default_format_version: str = "1.0.0"
    
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
        prediction_path: str = "data/prediction/"
        test_prediction_file: str = "test_predictions"
        validation_prediction_file: str = "validation_predictions"
        train_prediction_file: str = "train_predictions"
    
    @dataclass
    class Extensions:
        csv: str = ".csv"
        npz: str = ".npz"
        json: str = ".json"
        pickle: str = ".pkl"
        text: str = ".txt"
    
    @dataclass
    class Encodings:
        UTF_8: str = "utf_8"

    def __init__(self, number_of_features, layers, loss_type=None, output_activation_type=None, hidden_activation_type=None, hidden_weight_init_strategy=None, output_weight_init_strategy=None, bias_init_strategy=None, init_seed=None, init_random_range=None, hidden_bias_value=0.0, output_positive_prior=None, device=Device.CPU):
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
    
        match device:
            case self.Device.CPU:
                self.on_gpu = False
            case self.Device.CUDA:
                if cp is None:
                    raise RuntimeError("CuPy is not installed. Install a CUDA-enabled CuPy package first")
                self.on_gpu = True
            case _:
                raise ValueError("UnSupported Device Type")
    
        if self.on_gpu:
            self.__init_random_range = init_random_range if init_random_range is not None else cp.random.default_rng(init_seed)
        else:
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
            
            random_weights = self.to_device(random_weights, dtype=self.xp.float32)
            random_biases = self.to_device(random_biases, dtype=self.xp.float32)

            self.__WB.append((random_weights, random_biases))
    
    def scalar_to_python(self, x):
        if self.on_gpu:
            return float(cp.asnumpy(x))
        return float(x)
    
    def to_cpu_WB(self):
        if not self.on_gpu:
            return self.__WB
        return [(cp.asnumpy(W), cp.asnumpy(b)) for W, b in self.__WB]
    
    def cpu_to_cuda_WB(self, WB):
        if self.on_gpu:
            return [(cp.asarray(W), cp.asarray(b)) for W, b in WB]
        return [(np.asarray(W), np.asarray(b)) for W, b in WB]
    
    def to_device(self, x, dtype=None):
        if self.on_gpu:
            return cp.asarray(x, dtype=dtype)
        return np.asarray(x, dtype=dtype)

    def tp_cpu(self, x):
        if self.on_gpu:
            return cp.asnumpy(x)
        return np.asarray(x)
    
    def cpu_copy(self):
        model_cpu = copy.copy(self)
        model_cpu.on_gpu = False
        model_cpu._NeuralNetwork__WB = [(W.copy(), b.copy()) for W, b in self.to_cpu_WB()]
        model_cpu._NeuralNetwork__init_random_range = None
        model_cpu._NeuralNetwork__cache = []
        return model_cpu
    
    def move_to(self, device):
        match device:
            case self.Device.CPU:
                wb_cpu = self.to_cpu_WB()
                self._NeuralNetwork__WB = wb_cpu
                self.on_gpu = False
            case self.Device.CUDA:
                if cp is None:
                    raise RuntimeError("CuPy is not installed")
                self._NeuralNetwork__WB = [(cp.asarray(W), cp.asarray(b)) for W, b in self.to_cpu_WB()]
                self.on_gpu = True
            case _:
                raise ValueError("Unsupported device")
        
    @property
    def xp(self):
        return cp if self.on_gpu else np

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
    def init_layers(
        cls, 
        X, 
        number_of_hidden_layers, 
        output_width=None, 
        start_width=None, 
        hidden_width=None, 
        parameter_budget=None, 
        layer_strategy=None, 
        start_width_heurist=None, 
        start_width_heuristic_cap=512, 
        output_aware_multiplier=4, 
        expansion_multiplier=2
    ):
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
    
    def linear_model(self, W, X, b):
        return X @ W.T + b.T
    
    def sigmoid(self, z):
        xp = self.xp
        z = xp.asarray(z, dtype=float)
        out = xp.empty_like(z)

        pos = z >= 0
        neg = ~pos

        out[pos] = 1.0 / (1.0 + xp.exp(-z[pos]))
        ez = xp.exp(z[neg])
        out[neg] = ez / (1.0 + ez)

        return out
    
    def relu(self, z):
        xp = self.xp
        z = xp.asarray(z, dtype=float)
        return xp.maximum(0.0, z)

    def tanh(self, z):
        xp = self.xp
        z = xp.asarray(z, dtype=float)
        return xp.tanh(z)
    
    def leaky_relu(self, z, alpha=0.01):
        xp = self.xp
        z = xp.asarray(z, dtype=float)
        return xp.where(z > 0, z, alpha * z)

    def drelu_from_output(self, a):
        xp = self.xp
        a = xp.asarray(a, dtype=float)
        return (a > 0).astype(float)
        
    def dsigmoid_from_output(self, a):
        xp = self.xp
        return a * (1 - a)
    
    def dtanh_from_output(self, a):
        xp = self.xp
        a = xp.asarray(a, dtype=float)
        return 1.0 - a ** 2

    def dleaky_relu_from_output(self, a, alpha=0.01):
        xp = self.xp
        a = xp.asarray(a, dtype=float)
        return xp.where(a > 0, 1.0, alpha)
    
    def softmax(self, Z):
        xp = self.xp
        Z = xp.asarray(Z, dtype=float)
        Z_shifted = Z - xp.max(Z, axis=1, keepdims=True)
        exp_Z = xp.exp(Z_shifted)
        return exp_Z / xp.sum(exp_Z, axis=1, keepdims=True)
    
    def mse_loss(self, Y, A):
        xp = self.xp
        m = Y.shape[0]
        return xp.sum((A - Y) ** 2) / m

    def multi_class_cross_entropy_loss(self, Y, A, epsilon=1e-12):
        xp = self.xp
        m = Y.shape[0]
        A = xp.clip(A, epsilon, 1.0)
        return -xp.sum(Y * xp.log(A)) / m

    def binary_cross_entropy_loss(self, Y, A, epsilon=1e-12):
        xp = self.xp
        m = Y.shape[0]
        A = xp.clip(A, epsilon, 1.0 - epsilon)
        return -xp.sum(Y * xp.log(A) + (1 - Y) * xp.log(1 - A)) / m
        
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

    def hidden_derivative_from_output(self, a):
        hidden_activation_type = self.__hidden_activation_type
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

    def activation(self, A):
        hidden_activation_type = self.__hidden_activation_type
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
    
    def loss(self, Y, A, epsilon=None):
        if epsilon is None:
            epsilon = self.TrainDefaults().epsilon

        loss_type = self.__loss_type
        if loss_type == self.LossType.MSE:
            return self.mse_loss(Y, A)
        if loss_type == self.LossType.MULTI_CLASS_CROSS_ENTROPY:
            return self.multi_class_cross_entropy_loss(Y, A, epsilon)
        if loss_type == self.LossType.BINARY_CROSS_ENTROPY:
            return self.binary_cross_entropy_loss(Y, A, epsilon)

        is_binary = (self.__output_activation_type == self.OutputActivationType.SIGMOID and A.shape[1] == 1)

        if is_binary:
            return self.binary_cross_entropy_loss(Y, A, epsilon)

        return self.multi_class_cross_entropy_loss(Y, A, epsilon)

    def output_delta(self, y_hat, y):
        return y_hat - y

    def bernoulli(self, shape, p):
        xp = self.xp
        keep_prob = 1.0 - p
        return (xp.random.random(shape) < keep_prob).astype(float)

    def sum_weight_squares(self, WB):
        xp = self.xp
        return sum(xp.sum(W * W) for W, _ in WB)

    def forward_pass(self, X, cfg=None, training_mode=False):
        output_activation_type = self.__output_activation_type
        hidden_activation_type = self.__hidden_activation_type
        
        drop_out_rate = 0.0
        keep_prob = 0.0
        
        if training_mode:
            if cfg is None:
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
                A_raw = self.activation(Z)
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
    
    def backward_propagation(self, Y, cache, activation_cache, M, cfg=None, training_mode=False):
        hidden_activation_type = self.hidden_activation_type
        
        xp = self.xp
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
            db = xp.sum(dZ, axis=0, keepdims=True).T / m
            grads[l] = (dW, db)

            if l > 0:
                W_l, _ = self.__WB[l]
                A_l = cache[l]
                dA_prev = dZ @ W_l

                if training_mode and drop_out_rate > 0.0 and M[l - 1] is not None:
                    dA_prev = (dA_prev * M[l - 1]) / keep_prob
                
                A_prev = activation_cache[l - 1]
                dZ = dA_prev * self.hidden_derivative_from_output(A_prev)

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

    def count_parameters(self):
        number_of_parameters = 0
        for W, b in self.__WB:
            number_of_parameters = W.size + b.size
        return number_of_parameters
    
    def parameters_breakdown(self):
        breakdown = []
        for i, (W, b) in enumerate(self.__WB, start=1):
            total_parameters = W.size + b.size
            breakdown.append({
                "layer": i,
                "weight_shape": W.shape,
                "bias_shape": b.shape,
                "weight_parameters": W.size,
                "bias_parameters": b.size,
                "total_parameters": total_parameters
            })
        return breakdown
    
    def summary(self):
        print("\nNeural Network Summary\n")
        print(f"Number of layers: {self.__L}")
        print(f"Hidden Activation : {self.__hidden_activation_type.name}")
        print(f"Output Activation : {self.__output_activation_type.name}")
        print(f"Loss Type : {self.__loss_type.name}")
        print(f"Total Number of Parameters : {self.count_parameters()}")
        print()
        breakdown = self.parameters_breakdown()
        print(breakdown)
    
    def train(
        self, 
        X_train, 
        Y_train, 
        X_valid, 
        Y_valid,
        X_test,
        Y_test,
        log=True,
        graph=True,
        finalize=False,
        _log_predictions = False,
        early_stopping=False, 
        restore_best=False, 
        dropout=False,
        l2=False,
        cfg=None, 
        learning_decay_type=None,
        data_augmentation_type=None,
    ):
        xp = self.xp
        if X_train.ndim != 2:
            raise ValueError("X must be 2D: (samples, features)")
        if Y_train.ndim != 2:
            raise ValueError("Y must be 2D: (samples, outputs)")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        if X_train.shape[1] != self.__WB[0][0].shape[1]:
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
        
        if _log_predictions:
            tr = self.Paths
            train_prediction_file = tr.train_prediction_file
            validation_prediction_file = tr.validation_prediction_file
            test_prediction_file = tr.test_prediction_file
            prediction_path = tr.prediction_path
            prediction_tolerance = cfg.prediction_tolerance
            prediction_threshold = cfg.prediction_threshold
            _encoding=self.Encodings().UTF_8

        base_m = X_train.shape[0]

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

        if self.on_gpu:
            xp.random.seed(seed)
            random_range = None
        else:
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

        for epoch in range(epochs):
            train_data_loss = 0.0
            train_reg_loss = 0.0
            train_total_loss = train_data_loss + train_reg_loss

            X_shuffle, Y_shuffle = self.shuffle_dataset(X_train, Y_train, base_m, random_range)

            if data_augmentation_type is not None:
                X_epoch, Y_epoch = self.data_augmentation(X_shuffle, Y_shuffle, data_augmentation_type, random_range)
            else:
                X_epoch, Y_epoch = X_shuffle, Y_shuffle
            
            epoch_m = X_epoch.shape[0]

            for batch_start in range(0, epoch_m, batch_size):
                end = min(batch_start + batch_size, epoch_m)

                X_batch = X_epoch[batch_start:end, :]
                Y_batch = Y_epoch[batch_start:end, :]

                batch_m = Y_batch.shape[0]

                if dropout:
                    A, cache, activation_cache, M = self.forward_pass(X_batch, cfg=cfg, training_mode=True)
                    grads = self.backward_propagation(Y_batch, cache, activation_cache, M, cfg=cfg, training_mode=True)
                else:
                    A, cache, activation_cache, M = self.forward_pass(X_batch, cfg=None, training_mode=False)
                    grads = self.backward_propagation(Y_batch, cache, activation_cache, M, cfg=None, training_mode=False)
                
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
                
                batch_loss = self.loss(Y_batch, A, epsilon)
                train_data_loss += batch_loss * (end - batch_start)
            
            train_data_loss = train_data_loss / epoch_m
            # train_reg_loss = (l2_lambda / 2) * self.sum_weight_squares(self.__WB)
            
            if l2:
                train_reg_loss = (l2_lambda / (2 * epoch_m)) * self.sum_weight_squares(self.__WB)
            train_total_loss = train_data_loss + train_reg_loss
            _, val_data_loss, val_acc = self.evaluate_dataset(X_valid, Y_valid)
            
            train_data_loss_py = self.scalar_to_python(train_data_loss)
            train_reg_loss_py = self.scalar_to_python(train_reg_loss)
            train_total_loss_py = self.scalar_to_python(train_total_loss)
            
            val_data_loss_py = self.scalar_to_python(val_data_loss)
            val_acc_py = self.scalar_to_python(val_acc)
            
            val_losses.append(val_data_loss_py)
            val_accuracies.append(val_acc_py)

            if epoch >= 0 and epoch % step == 0:
                if learning_decay_type is not None:
                    current_lr = self.learning_decay(initial_lr, decay_factor, epoch, step, learning_decay_type)

                if log:
                    print(
                        "epoch =", epoch,
                        "train_data_loss =", round(train_data_loss_py, 6),
                        "train_reg_loss =", round(train_reg_loss_py, 6),
                        "train_total_loss =", round(train_total_loss_py, 6),
                        "val_data_loss =", round(val_data_loss_py, 6),
                        "val_acc =", round(val_acc_py, 4)
                    )

                if graph:
                    losses.append(train_data_loss_py)
                    steps.append(epoch)

            if val_data_loss_py < best_val_loss:
                best_val_loss = val_data_loss_py
                best_WB = copy.deepcopy(self.__WB)
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
 
            if early_stopping and patience_counter >= patience:
                print("early stopping at epoch", epoch)
                break
        
        if restore_best and best_WB is not None:
            self.__WB = best_WB

        _, final_data_loss, final_accuracy = self.evaluate_dataset(X_train, Y_train)
        final_reg_loss = (l2_lambda / (2 * X_train.shape[0])) * self.sum_weight_squares(self.__WB)
        final_loss = final_data_loss + final_reg_loss

        if graph:
            plt.figure(self.TrainResults.figure_title)
            plt.plot(steps, losses)
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.title("Training Loss")
            plt.show()
        
        if finalize:
            print("\nFinal Results : \n")
            train_pred, train_loss, train_acc = self.evaluate_dataset(X_train, Y_train)
            val_pred, val_loss, val_acc = self.evaluate_dataset(X_valid, Y_valid)
            
            print("Train:", train_loss, train_acc)
            print("Valid:", val_loss, val_acc)
            
            if X_test is not None and Y_test is not None:
                test_pred, test_loss, test_acc = self.evaluate_dataset(X_test, Y_test)
                print("Test :", test_loss, test_acc)
        
        train_results = self.TrainResults(
            losses=losses, 
            val_losses=val_losses, 
            best_val_loss=best_val_loss, 
            val_accuracies=val_accuracies, 
            best_epoch=best_epoch, 
            final_loss=final_loss, 
            accuracy=final_accuracy, 
            final_learning_rate=current_lr
        )

        if _log_predictions:
            total_samples = len(Y_train) + len(Y_valid) + (len(Y_test) if Y_test is not None else 0)
            if total_samples > prediction_tolerance:
                print("\nThe current dataset is too large, >100 samples is expected to be written!")
                print("\nDo you really want to write down all those predictions? If not, input [N/No], otherwise [Y/Yes] (case-insensitive)")
                while True:
                    choice = input("Do you wish to continue? [y(Y)/n(N)] ").strip().lower()
                    if choice in ("y", "yes"):
                        break
                    elif choice in ("n", "no"):
                        _log_predictions = False
                        break
                    else:
                        print("Invalid Input, supported values are : [N/n/No, Y/y/Yes]")
            
            self.log_predictions(Y_train, train_pred, _log_predictions, train_prediction_file, prediction_path, prediction_tolerance, prediction_threshold, _encoding)
            self.log_predictions(Y_valid, val_pred, _log_predictions, validation_prediction_file, prediction_path, prediction_tolerance, prediction_threshold, _encoding)
            if Y_test is not None and test_pred is not None:
                self.log_predictions(Y_test, test_pred, _log_predictions, test_prediction_file, prediction_path, prediction_tolerance, prediction_threshold, _encoding)
        
        return train_results
            
    def log_predictions(self, Y, predictions, _log_predictions, prediction_file, prediction_path, prediction_tolerance, prediction_threshold, _encoding):
        if prediction_path is not None and not os.path.exists(prediction_path):
            os.mkdir(prediction_path)

        if _log_predictions:
            prediction_file_name = prediction_file + self.Extensions.text
            prediction_file_path = prediction_path + prediction_file_name
            
            if _encoding is None:
                _encoding = self.Encodings().UTF_8

            with open(prediction_file_path, "w", encoding=_encoding) as f:
                for i in range(len(predictions)):
                    sample = Y[i]
                    prediction = predictions[i]
                    sample_result = np.where(sample == 1)[0]
                    prediction_result = np.where(prediction == 1)[0]
                    f.write(f"Sample : {sample_result}, Prediction : {prediction_result} {"correct" if sample_result == prediction_result else "failed"}\n")
                
            print(f"\nFirst <{prediction_threshold} predictions are written in {prediction_file_path}")  
        else:
            first_sample, last_sample, first_prediction, last_prediction = Y[0], Y[-1], predictions[0], predictions[-1]
            print("Detailed Prediction Logging disabled, falling back to the first and last sample predictions : ")
            print(f"First Sample : {np.where(first_sample == 1)[0]}, Prediction : {np.where(first_prediction == 1)[0]}")
            print(f"Last Sample : {np.where(last_sample == 1)[0]}, Prediction : {np.where(last_prediction == 1)[0]}\n")
    
    def shuffle_dataset(self, X, Y, size, random_range):
        xp = self.xp
        
        if self.on_gpu:
            permutation = xp.random.permutation(size)
        else:
            if random_range is None:
                random_range = np.random.default_rng()
            permutation = random_range.permutation(size)

        X_shuf = X[permutation]
        Y_shuf = Y[permutation]
        return X_shuf, Y_shuf
    
    def evaluate_dataset(self, X, Y, epsilon=None, threshold=None):
        output_activation_type = self.output_activation_type
        xp = self.xp
         
        if epsilon is None:
            epsilon = self.TrainDefaults().epsilon
        
        A, _ = self.predict(X)
        loss = self.loss(Y, A, epsilon)
        
        prediction, is_binary = self.predict_classes(X, output_activation_type, xp, threshold)
        
        if is_binary:
            accuracy = xp.mean(prediction == Y) * 100.0
        else: 
            predicted_classes = xp.argmax(A, axis=1)
            true_classes = xp.argmax(Y, axis=1)
            accuracy = xp.mean(predicted_classes == true_classes) * 100.0

        return prediction, loss, accuracy

    def predict_classes(self, X, output_activation_type=None, xp=None, threshold=None):
        if output_activation_type is None:
            output_activation_type = self.__output_activation_type
        if xp is None:
            xp = self.xp
        
        A = self.predict_proba(X)
        
        is_binary = (output_activation_type == self.OutputActivationType.SIGMOID and A.shape[1] == 1)

        if threshold is None:
            threshold = self.TrainDefaults().threshold
        if is_binary:
            prediction = (A >= threshold).astype(float)
            return prediction, is_binary
        else:
            prediction = xp.zeros_like(A)
            prediction[xp.arange(A.shape[0]), xp.argmax(A, axis=1)] = 1
            return prediction, is_binary

    def predict_proba(self, X):
        A, _ = self.predict(X)
        return A
    
    def predict(self, x):
        A, cache, _, _ = self.forward_pass(x, training_mode=False)
        return A, cache

    def data_augmentation(self, X, Y, augmentation_type, random_range=None):
        xp = self.xp

        X = xp.asarray(X, dtype=float)
        Y = xp.asarray(Y)

        X_aug = X.copy()
        Y_aug = Y.copy()

        if self.on_gpu:
            random_range = xp.random
        elif random_range is None:
            random_range = np.random.default_rng()
        
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
                X_aug = xp.maximum(0.0, X_aug * scale)

            case self.DataAugmentation.SAME_CLASS_INTERPOLATION:
                if Y_aug.shape[1] == 1:
                    labels = Y_aug.flatten().astype(int)
                else:
                    labels = xp.argmax(Y_aug, axis=1)

                synthetic_x = []
                synthetic_y = []

                for label in xp.unique(labels):
                    indices = xp.where(labels == label)[0]
                    num_synthetic = len(indices) // 2

                    for _ in range(num_synthetic):
                        if len(indices) >= 2:
                            i1, i2 = random_range.choice(indices, size=2, replace=False)
                        else:
                            i1 = i2 = indices[0]

                        alpha = random_range.uniform(0.2, 0.8)

                        x_new = alpha * X_aug[i1] + (1.0 - alpha) * X_aug[i2]
                        y_new = Y_aug[i1].copy()

                        synthetic_x.append(x_new)
                        synthetic_y.append(y_new)

                if synthetic_x:
                    synthetic_x = xp.stack(synthetic_x, axis=0)
                    synthetic_y = xp.stack(synthetic_y, axis=0)

                    X_aug = xp.concatenate([X_aug, synthetic_x], axis=0)
                    Y_aug = xp.concatenate([Y_aug, synthetic_y], axis=0)

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
    
    def get_metadata(self, format_version=None, train_history=None):
        if format_version is None:
            format_version = self.TrainDefaults().default_format_version
        state = {
            "format_version": format_version,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "number_of_layers": self.__L,
            "hidden_activation_type": self.__hidden_activation_type.name,
            "output_activation_type": self.__output_activation_type.name,
            "loss_type": self.__loss_type.name,
            "parameter_count": self.count_parameters()
        }
        if train_history is not None:
            state["train_history"] = train_history
        return state

    def save_model(self, file_name, meta=False, format_version=None, train_history=False):
        if not file_name.endswith(".pkl"):
            file_name = file_name + ".pkl"

        modelpath = self.Paths.model_path
        if modelpath is not None and not os.path.exists(modelpath):
            os.mkdir(modelpath)
        
        model_cpu = self.cpu_copy()
        
        file_path = modelpath + file_name
        with open(file_path, "wb") as f:
            pickle.dump(model_cpu, f)
        
        if meta:
            meta_data = self.get_metadata(format_version, train_history)
            meta_data_path = modelpath + file_name + self.Paths.meta_data_flair 
            with open(meta_data_path, "wb") as f:
                pickle.dump(meta_data, f)

    @classmethod
    def load_model(cls, model_path, meta_data_path, device=None, meta=False):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        if device is not None:
            model.move_to(device)
        meta_data = None
        if meta:
            with open(meta_data_path, "rb") as f:
                meta_data = pickle.load(f)
        return model, meta_data
    
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
        npz_path = cls.Paths.npz_path
        if filename is None:
            filename = cls.Paths.default_data
        
        npz_extension = cls.Extensions.npz
        if not filename.endswith(npz_extension):
            filename = filename + npz_extension
            
        npz_file_path = npz_path
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
            _encoding = cls.Encodings().UTF_8
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
            _encoding = cls.Encodings().UTF_8
            
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
        Y = np.array([r["label"] for r in records], dtype=np.int64)
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
    number_of_classes = 2
    dataset = NeuralNetwork.load_from_npz('data/npz/dataset.npz')
    # prepared_dataset = dataset
    prepared_dataset = NeuralNetwork.prepare_datasets(dataset, number_of_classes)
    
    X_train = prepared_dataset["X_train"]
    Y_train = prepared_dataset["Y_train"]
    X_valid = prepared_dataset["X_valid"]
    Y_valid = prepared_dataset["Y_valid"]
    X_test = prepared_dataset["X_test"]
    Y_test = prepared_dataset["Y_test"]

    number_of_features = X_train.shape[1]
    layers = [4, 1, number_of_classes]
    # layers = NeuralNetwork.init_layers(
    #     X=X_train,
    #     number_of_hidden_layers=10,
    #     output_width=10    
    # )
    
    model = NeuralNetwork(
        number_of_features,
        layers,
        loss_type=NeuralNetwork.LossType.BINARY_CROSS_ENTROPY,
        output_activation_type=NeuralNetwork.OutputActivationType.SIGMOID,
        hidden_activation_type=NeuralNetwork.HiddenActivationType.RELU,
        device=NeuralNetwork.Device.CPU
    )
    
    model.summary()
    
    X_train = model.to_device(X_train, dtype=model.xp.float32)
    Y_train = model.to_device(Y_train, dtype=model.xp.float32)

    X_valid = model.to_device(X_valid, dtype=model.xp.float32)
    Y_valid = model.to_device(Y_valid, dtype=model.xp.float32)

    X_test = model.to_device(X_test, dtype=model.xp.float32)
    Y_test = model.to_device(Y_test, dtype=model.xp.float32)

    cfg = NeuralNetwork.TrainDefaults()
    
    # limit = 101
    cfg.step = 100
    cfg.epochs = 1000
    cfg.batch_size = 4
    cfg.learning_rate = 0.01
    
    # X_train = X_train[:limit]
    # Y_train = Y_train[:limit]
    # X_valid = X_valid[:limit]
    # Y_valid = Y_valid[:limit]
    # X_test = X_test[:limit]
    # Y_test = Y_test[:limit]

    # results = model.train(
    #     X_train, 
    #     Y_train, 
    #     X_valid, 
    #     Y_valid,
    #     X_test,
    #     Y_test,
    #     learning_decay_type=None,
    #     data_augmentation_type=None,
    #     cfg=cfg,
    #     early_stopping=False,
    #     restore_best=False,
    #     finalize=True, 
    #     l2=False, 
    #     dropout=False,
    #     graph=False,
    #     _log_predictions=True
    # )
    
    # model.save_model("mnist_small")

    # loaded_model = NeuralNetwork.load_model(
    #     "models/mnist_small.pkl",
    #     device=NeuralNetwork.Device.CPU
    # )