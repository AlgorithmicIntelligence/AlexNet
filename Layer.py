import numpy as np


class ConvolutionalLayer(object):
    def __init__(self, weights_shape, pad="SAME", stride=1, activation_function=None, initializer="ALEXNET_bias1"):
        self.shape = weights_shape
        # LeNet Initializer
        if initializer == "LENET":
            Fi = np.prod(weights_shape[:-1])
            self.weights = np.random.uniform(-2.4/Fi, 2.4/Fi, weights_shape)
            self.biases = np.ones([1, 1, 1, weights_shape[-1]]) * 0.01
            #AlexNet Initializer
        elif initializer == "ALEXNET_bias1":
            self.weights = np.random.normal(0, 0.01, weights_shape)
            self.biases = np.ones([1, 1, 1, weights_shape[-1]])
        elif initializer == "ALEXNET_bias0":
            self.weights = np.random.normal(0, 0.01, weights_shape)
            self.biases = np.zeros([1, 1, 1, weights_shape[-1]])
        self.v_weights = np.zeros(self.weights.shape)
        self.v_biases = np.zeros(self.biases.shape)
        self.activation_function = activation_function
        self.pad = pad
        self.stride = stride
        self.inputs = None
        self.inputs_activation = None
        self.outputs_activation = None
        self.lr = None

    def forward_propagation(self, inputs):
        self.inputs = inputs
        if self.pad == "SAME":
            inputs = np.pad(inputs, ((0, 0), ((self.shape[0]-1)//2, self.shape[0]//2), ((self.shape[1]-1)//2, self.shape[1]//2), (0, 0)), "constant")
        elif self.pad != "VALID":
                raise AssertionError("PADDING NAME ERROR")

        outputs = np.zeros([inputs.shape[0], (inputs.shape[1]-self.shape[0])//self.stride + 1, (inputs.shape[2]-self.shape[1])//self.stride + 1, self.shape[-1]])
        for h in range(outputs.shape[1]):
            for w in range(outputs.shape[2]):
                outputs[:, h, w, :] = np.tensordot(inputs[:, h*self.stride:h*self.stride+self.shape[0], w*self.stride:w*self.stride+self.shape[1], :], self.weights, axes=([1,2,3],[0,1,2])) + self.biases[:, 0, 0, :]
        self.inputs_activation = outputs
        if self.activation_function == "SIGMOID":
            outputs = 1/(1+np.exp(outputs))
            self.outputs_activation = outputs
        elif self.activation_function == "SQUASHING":
            outputs = 1.7159 * np.tanh(2/3 * outputs)
        elif self.activation_function == "RELU":
            outputs = np.where(outputs < 0, 0, outputs)
        elif self.activation_function != "LINEAR":
                raise AssertionError("ACTIVATION FUNCTION NAME ERROR")
        
        return outputs

    def backward_propagation(self, d_outputs, learning_rate, weight_decay=0, momentum=0):
        if self.activation_function == "SIGMOID":
            d_outputs *= (1-self.outputs_activation) * self.outputs_activation
        elif self.activation_function == "SQUASHING":
            d_outputs *= 1.7159 * 2/3 * (1-np.power(np.tanh(2/3 * self.inputs_activation), 2))
        elif self.activation_function == "RELU":
            d_outputs *= np.where(self.inputs_activation < 0, 0, 1)
        
        d_inputs = np.zeros(self.inputs.shape)
        d_weights = np.zeros(self.weights.shape)
        d_biases = np.zeros(self.biases.shape)
        
        if self.pad == "SAME":
            inputs = np.pad(self.inputs, ((0, 0), ((self.shape[0]-1)//2, self.shape[0]//2), ((self.shape[1]-1)//2, self.shape[1]//2),(0, 0)), "constant")
            d_inputs = np.pad(d_inputs, ((0, 0), ((self.shape[0]-1)//2, self.shape[0]//2), ((self.shape[1]-1)//2, self.shape[1]//2),(0, 0)), "constant")
        else:
            inputs = self.inputs
            
        for h in range(d_outputs.shape[1]):
            for w in range(d_outputs.shape[2]):
                d_inputs[:, h*self.stride:h*self.stride+self.shape[0], w*self.stride:w*self.stride+self.shape[1], :] += np.tensordot(d_outputs[:, h, w, :], self.weights, axes=([-1],[-1]))
                d_weights += np.average(np.expand_dims(d_outputs[:, h:h+1, w:w+1, :], axis=-2) * np.expand_dims(inputs[:, h*self.stride:h*self.stride+self.shape[0], w*self.stride:w*self.stride+self.shape[1], :], axis=-1), axis=0)
                d_biases += np.average(d_outputs[:, h, w, :], axis=0).reshape(1, 1, 1, -1)
                
        if self.pad == "SAME":
            d_inputs = d_inputs[:, (self.shape[0]-1)//2:-(self.shape[0]-1)//2, (self.shape[1]-1)//2:-(self.shape[1]-1)//2, :]
        # update
        self.v_weights = momentum * self.v_weights - learning_rate * (d_weights + weight_decay * self.weights)
        self.weights += self.v_weights
        self.v_biases = momentum * self.v_biases - learning_rate * (d_biases + weight_decay * self.biases)
        self.biases += self.v_biases

        return d_inputs
    
    def SDLM(self, d2_outputs, learning_rate):
        if self.activation_function == "SIGMOID":
            d2_outputs *= self.outputs_activation * (1-self.outputs_activation) * (1-2*self.outputs_activation)
        elif self.activation_function == "SQUASHING":
            d2_outputs *= np.power(1.7159 * 2/3 * (1-np.power(np.tanh(2/3 * self.inputs_activation),2)), 2)
        elif self.activation_function == "RELU":
            d2_outputs *= np.where(self.inputs_activation < 0, 0, 1)

        d2_inputs = np.zeros(self.inputs.shape)
        d2_weights = np.zeros(self.weights.shape)

        if self.pad == "SAME":
            inputs = np.pad(self.inputs, ((0, 0), ((self.shape[0]-1)//2, self.shape[0]//2), ((self.shape[1]-1)//2, self.shape[1]//2),(0, 0)), "constant")
            d2_inputs = np.pad(d2_inputs, (((0, 0), (self.shape[0]-1)//2, self.shape[0]//2), ((self.shape[1]-1)//2, self.shape[1]//2),(0, 0)), "constant")
        else:
            inputs = self.inputs
            
        for h in range(d2_outputs.shape[1]):
            for w in range(d2_outputs.shape[2]):
                d2_inputs[:, h*self.stride:h*self.stride+self.shape[0], w*self.stride:w*self.stride+self.shape[1], :] += np.tensordot(d2_outputs[:, h, w, :], np.power(self.weights, 2), axes=([-1],[-1]))
                d2_weights += np.average(np.expand_dims(d2_outputs[:, h:h+1, w:w+1, :], axis=-2) * np.expand_dims(np.power(inputs[:, h*self.stride:h*self.stride+self.shape[0], w*self.stride:w*self.stride+self.shape[1], :], 2), axis=-1), axis=0)
                
        if self.pad == "SAME":
            d2_inputs = d2_inputs[(self.size[0]-1)//2:self.d_inputs.shape[0]-self.size[0]//2+1, (self.size[1]-1)//2:self.d_inputs.shape[1]-self.size[1]//2+1]
        # update
        h = np.sum(d2_weights)
        self.lr = learning_rate / (0.02 + h)

        return d2_inputs


class ConvolutionalCombinationLayer(object):
    def __init__(self, shape, combination, pad="SAME", stride=1, activation_function=None):
        Fi = np.prod(shape[:-1])
        self.shape = shape
        self.combination = combination
        self.weights = []
        self.biases = []
        for f in self.combination:
            weight = np.random.uniform(-2.4 / Fi, 2.4 / Fi, [shape[0], shape[1], len(f), 1])
            bias = np.ones([1, 1, 1, 1]) * 0.01
            self.weights.append(weight)
            self.biases.append(bias)
        self.activation_function = activation_function
        self.pad = pad
        self.stride = stride
        self.inputs = None
        self.inputs_activation = None
        self.outputs_activation = None
        self.lr = None

    def forward_propagation(self, inputs):
        self.inputs = inputs
        if self.pad == "SAME":
            inputs = np.pad(inputs, (((0, 0), (self.shape[0] - 1) // 2, self.shape[0] // 2), ((self.shape[1] - 1) // 2, self.shape[1] // 2), (0, 0)), "constant")
        outputs = np.zeros([inputs.shape[0], (inputs.shape[1] - self.shape[0]) // self.stride + 1, (inputs.shape[2] - self.shape[1]) // self.stride + 1, self.shape[-1]])
        for i, f in enumerate(self.combination):
            for h in range(outputs.shape[1]):
                for w in range(outputs.shape[2]):
                    outputs[:, h, w, i:i+1] = np.tensordot(inputs[:, h * self.stride:h * self.stride + self.shape[0], w * self.stride:w * self.stride + self.shape[1], f], self.weights[i], axes=([1,2,3],[0,1,2])) + self.biases[i]

        self.inputs_activation = outputs
        if self.activation_function == "SIGMOID":
            outputs = 1 / (1 + np.exp(outputs))
            self.outputs_activation = outputs
        elif self.activation_function == "SQUASHING":
            outputs = 1.7159 * np.tanh(2/3 * outputs)

        return outputs

    def backward_propagation(self, d_outputs):
        if self.activation_function == "SIGMOID":
            d_outputs *= (1-self.outputs_activation) * self.outputs_activation
        elif self.activation_function == "SQUASHING":
            # d_outputs = 1.7159 * 2/3 / np.power(np.cosh(2/3 * d_outputs), 2)
            d_outputs *= 1.7159 * 2/3 * (1-np.power(np.tanh(2/3 * self.inputs_activation), 2))

        d_inputs = np.zeros(self.inputs.shape)
        d_weights = []
        d_biases = []
        for f in self.combination:
            d_weight = np.zeros([self.shape[0], self.shape[1], len(f), 1])
            d_bias = np.zeros([1, 1, 1, 1])
            d_weights.append(d_weight)
            d_biases.append(d_bias)
        if self.pad == "SAME":
            inputs = np.pad(self.inputs, (((self.shape[0] - 1) // 2, self.shape[0] // 2), ((self.shape[1] - 1) // 2, self.shape[1] // 2), (0, 0)), "constant")
            d_inputs = np.pad(d_inputs, (((self.shape[0] - 1) // 2, self.shape[0] // 2), ((self.shape[1] - 1) // 2, self.shape[1] // 2), (0, 0)), "constant")
        else:
            inputs = self.inputs
        for i,f in enumerate(self.combination):
            for h in range(d_outputs.shape[1]):
                for w in range(d_outputs.shape[2]):
                    d_inputs[:, h * self.stride:h * self.stride + self.shape[0], w * self.stride:w * self.stride + self.shape[1], f] += d_outputs[:, h:h+1, w:w+1, i:i+1] * np.transpose(self.weights[i], (3, 0, 1, 2))
                    d_weights[i] += np.expand_dims(np.average(d_outputs[:, h:h+1, w:w+1, i:i+1] * inputs[:, h * self.stride:h * self.stride + self.shape[0], w * self.stride:w * self.stride + self.shape[1], f], axis=0), axis=-1)
                    d_biases[i] += np.average(d_outputs[:, h, w, i], axis=0)
        if self.pad == "SAME":
            d_inputs = d_inputs[(self.size[0] - 1) // 2:self.d_inputs.shape[0] - self.size[0] // 2 + 1, (self.size[1] - 1) // 2:self.d_inputs.shape[1] - self.size[1] // 2 + 1]
        # update
        for i in range(len(self.combination)):
            self.weights[i] -= self.lr * d_weights[i]
            self.biases[i] -= self.lr * d_biases[i]

        return d_inputs

    def SDLM(self, d2_outputs, learning_rate):
        if self.activation_function == "SIGMOID":
            d2_outputs *= self.outputs_activation * (1-self.outputs_activation) * (1-2*self.outputs_activation)
        elif self.activation_function == "SQUASHING":
            d2_outputs *= np.power(1.7159 * 2/3 * (1-np.power(np.tanh(2/3 * self.inputs_activation), 2)), 2)
        
        d2_inputs = np.zeros(self.inputs.shape)
        d2_weights = list()
        for f in self.combination:
            d_weight = np.zeros([self.shape[0], self.shape[1], len(f), 1])
            d2_weights.append(d_weight)
   
        if self.pad == "SAME":
            inputs = np.pad(self.inputs, ((0, 0), ((self.shape[0]-1)//2, self.shape[0]//2), ((self.shape[1]-1)//2, self.shape[1]//2),(0, 0)), "constant")
            d2_inputs = np.pad(d2_inputs, (((0, 0), (self.shape[0]-1)//2, self.shape[0]//2), ((self.shape[1]-1)//2, self.shape[1]//2),(0, 0)), "constant")
        else:
            inputs = self.inputs
            
        for i,f in enumerate(self.combination):
            for h in range(d2_outputs.shape[1]):
                for w in range(d2_outputs.shape[2]):
                    d2_inputs[:, h * self.stride:h * self.stride + self.shape[0], w * self.stride:w * self.stride + self.shape[1], f] += d2_outputs[:, h:h+1, w:w+1, i:i+1] * np.transpose(np.power(self.weights[i], 2), (3, 0, 1, 2))
                    d2_weights[i] += np.expand_dims(np.average(d2_outputs[:, h:h+1, w:w+1, i:i+1] * np.power(inputs[:, h * self.stride:h * self.stride + self.shape[0], w * self.stride:w * self.stride + self.shape[1], f], 2), axis=0), axis=-1)
        if self.pad == "SAME":
            d2_inputs = d2_inputs[(self.size[0] - 1) // 2:self.d2_inputs.shape[0] - self.size[0] // 2 + 1, (self.size[1] - 1) // 2:self.d2_inputs.shape[1] - self.size[1] // 2 + 1]
        # update
        h = 0
        for d2_weight in d2_weights:
            h += np.sum(d2_weight)
        self.lr = learning_rate / (0.02 + h)

        return d2_inputs


class PoolingLayer(object):
    def __init__(self, shape, stride=2, mode="MAX", activation_function="SQUASHING"):
        self.shape = shape
        Fi = np.prod(np.array(shape[:2]))
        self.weights = np.random.uniform(-2.4 / Fi, 2.4 / Fi, [1, 1, 1, shape[-1]])
        self.biases = np.ones([1, 1, 1, shape[-1]]) * 0.01
        self.activation_function = activation_function
        self.stride = stride
        self.mode = mode
        self.inputs = None
        self.mask = None
        self.inputs_activation = None
        self.outputs_activation = None
        self.lr = None

    def forward_propagation(self, inputs):
        self.inputs = inputs
        outputs = np.zeros([inputs.shape[0], (inputs.shape[1]-self.shape[0])//self.stride + 1, (inputs.shape[2]-self.shape[1])//self.stride + 1, self.inputs.shape[3]])
        self.mask = np.zeros(outputs.shape, dtype=np.int8)
        for h in range(outputs.shape[1]):
            for w in range(outputs.shape[2]):
                if self.mode == "MAX":
                    flat_inputs = inputs[:, h*self.stride:h*self.stride + self.shape[0], w*self.stride:w*self.stride + self.shape[1], :].reshape(inputs.shape[0], -1, inputs.shape[-1])
                    self.mask[:, h, w, :] = np.argmax(flat_inputs, axis=1)
                    for i in range(outputs.shape[0]):
                        for c in range(outputs.shape[3]):
                            outputs[i, h, w, c] = flat_inputs[i, self.mask[i, h, w, c], c]
                elif self.mode == "AVERAGE":
                    # outputs[:, h, w, :] = np.average(inputs[:, h*self.stride:h*self.stride + self.shape[0], w*self.stride:w*self.stride + self.shape[1], :], axis=(1, 2))
                    outputs[:, h, w, :] = self.weights * np.average(inputs[:, h * self.stride:h * self.stride + self.shape[0], w * self.stride:w * self.stride + self.shape[1], :], axis=(1, 2)) + self.biases

        self.inputs_activation = outputs
        if self.activation_function == "SIGMOID":
            outputs = 1 / (1 + np.exp(outputs))
            self.outputs_activation = outputs
        elif self.activation_function == "SQUASHING":
            outputs = 1.7159 * np.tanh(2/3 * outputs)
        return outputs

    def backward_propagation(self, d_outputs):
        if self.activation_function == "SIGMOID":
            d_outputs *= (1-self.outputs_activation) * self.outputs_activation
        elif self.activation_function == "SQUASHING":
            # d_outputs = 1.7159 * 2/3 / np.power(np.cosh(2/3 * d_outputs), 2)
            d_outputs *= 1.7159 * 2/3 * (1-np.power(np.tanh(2/3 * self.inputs_activation), 2))

        d_inputs = np.zeros(self.inputs.shape)
        d_weights = np.zeros(self.weights.shape)
        d_biases = np.zeros(self.biases.shape)
        for h in range(d_outputs.shape[1]):
            for w in range(d_outputs.shape[2]):
                w_interval = [w * self.stride, w*self.stride+self.shape[0]]
                h_interval = [h*self.stride, h*self.stride+self.shape[1]]
                if self.mode == "MAX":
                    for i in range(d_outputs.shape[0]):
                        for c in range(d_outputs.shape[3]):
                            d_inputs[i, h_interval[0]+self.mask[i, h, w, c]//self.shape[1], w_interval[0]+self.mask[i, h, w, c] % self.shape[0], c] += d_outputs[i, h, w, c]
                elif self.mode == "AVERAGE":
                    # d_inputs[:, h_interval[0]:h_interval[1], w_interval[0]:w_interval[1], :] += np.repeat(np.repeat(d_outputs[:, h:h+1, w:w+1, :], 2, axis=1), 2, axis=2) / self.shape[0] / self.shape[1]
                    d_inputs[:, h_interval[0]:h_interval[1], w_interval[0]:w_interval[1], :] += self.weights * np.repeat(np.repeat(d_outputs[:, h:h+1, w:w+1, :], 2, axis=1), 2, axis=2) / self.shape[0] / self.shape[1]
                    d_weights += np.average(np.average(self.inputs[:, h_interval[0]:h_interval[1], w_interval[0]:w_interval[1], :], axis=(1, 2)) * d_outputs[:, h, w, :], axis=0)
                    d_biases += np.average(d_outputs[:, h, w, :], axis=0)
                    self.weights -= self.lr * d_weights
                    self.biases -= self.lr * d_biases
        
        return d_inputs

    def SDLM(self, d2_outputs, learning_rate):
        if self.activation_function == "SIGMOID":
            d2_outputs *= self.outputs_activation * (1-self.outputs_activation) * (1-2*self.outputs_activation)
        elif self.activation_function == "SQUASHING":
            d2_outputs *= np.power(1.7159 * 2/3 * (1-np.power(np.tanh(2/3 * self.inputs_activation), 2)), 2)

        d2_inputs = np.zeros(self.inputs.shape)
        d2_weights = np.zeros(self.weights.shape)
        for h in range(d2_outputs.shape[1]):
            for w in range(d2_outputs.shape[2]):
                w_interval = [w * self.stride, w*self.stride+self.shape[0]]
                h_interval = [h*self.stride, h*self.stride+self.shape[1]]
                if self.mode == "MAX":
                    weights = self.inputs[h_interval[0]:h_interval[1], w_interval[0]:w_interval[1], :] == np.max(self.inputs[h_interval[0]:h_interval[1], w_interval[0]:w_interval[1], :], axis=(0, 1))
                    d2_inputs[h_interval[0]:h_interval[1], w_interval[0]:w_interval[1], :] += np.repeat(np.repeat(d2_outputs[h, w, :] * weights, 2, axis=0), 2, axis=1)
                elif self.mode == "AVERAGE":
                    # d2_inputs[:, h_interval[0]:h_interval[1], w_interval[0]:w_interval[1], :] += np.repeat(np.repeat(d2_outputs[:, h:h+1, w:w+1, :], 2, axis=1), 2, axis=2) / self.shape[0] / self.shape[1]
                    d2_inputs[:, h_interval[0]:h_interval[1], w_interval[0]:w_interval[1], :] += np.power(self.weights / self.shape[0] / self.shape[1], 2) * np.repeat(np.repeat(d2_outputs[:, h:h+1, w:w+1, :], 2, axis=1), 2, axis=2)
                    d2_weights += np.average(np.power(np.average(self.inputs[:, h_interval[0]:h_interval[1], w_interval[0]:w_interval[1], :], axis=(1, 2)), 2) * d2_outputs[:, h, w, :], axis=0)
        h = np.sum(d2_weights)
        self.lr = learning_rate / (0.02 + h)
        return d2_inputs


class LocalResponseNormalization(object):
    def __init__(self, depth_radius, bias, alpha, beta):
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.weights = None

    def forward_propagation(self, inputs):
        self.weights = np.zeros(inputs.shape)
        outputs = np.zeros(inputs.shape)

        for h in range(outputs.shape[1]):
            for w in range(outputs.shape[2]):
                sum_queue = np.zeros((inputs.shape[0], self.depth_radius * 2 + 1))
                for c in range(outputs.shape[3]+self.depth_radius):
                    sum_queue -= sum_queue[:, 0:1]
                    sum_queue[:, :-1] = sum_queue[:, 1:]
                    if c < outputs.shape[3]:
                        sum_queue[:, -1] += inputs[:, h, w, c]**2
                    if self.depth_radius <= c:
                        self.weights[:, h, w, c-self.depth_radius] = 1/(self.bias+self.alpha*sum_queue[:, -1])**self.beta
                        outputs[:, h, w, c-self.depth_radius] = inputs[:, h, w, c-self.depth_radius] * self.weights[:, h, w, c-self.depth_radius]

        return outputs

    def backward_propagation(self, d_outputs):
        d_inputs = d_outputs * self.weights
        return d_inputs


class DropOut(object):
    def __init__(self, keep_prob, mode="TRAINING"):
        self.keep_prob = keep_prob
        self.mode = mode
        self.mask = None

    def forward_propagation(self, inputs):
        self.mask = np.random.binomial(1, self.keep_prob, inputs.shape)
        outputs = inputs * self.mask

        if self.mode == "TESTING":
            outputs *= self.keep_prob

        return outputs

    def backward_propagation(self, d_outputs):
        d_inputs = d_outputs * self.mask

        return d_inputs


class FullyConnectedLayer(object):
    def __init__(self, shape, activation_function=None, initializer="ALEXNET_bias1"):
        if initializer == "LENET":
            self.weights = np.random.normal(0, 0.1, shape)
            self.biases = np.ones([1, shape[-1]]) * 0.01
        #AlexNet Initializer
        elif initializer == "ALEXNET_bias1":
            self.weights = np.random.normal(0, 0.01, shape)
            self.biases = np.ones([1, shape[-1]])
        self.v_weights = np.zeros(self.weights.shape)
        self.v_biases = np.zeros(self.biases.shape)
        self.activation_function = activation_function
        self.inputs = None
        self.inputs_activation = None
        self.outputs_activation = None
        self.lr = None

    def forward_propagation(self, inputs):
        self.inputs = inputs
        outputs = np.matmul(inputs, self.weights) + self.biases
        
        self.inputs_activation = outputs
        if self.activation_function == "SIGMOID":
            outputs = 1/(1+np.exp(outputs))
            self.outputs_activation = outputs
        elif self.activation_function == "SQUASHING":
            outputs = 1.7159 * np.tanh(2/3 * outputs)
        elif self.activation_function == "RELU":
            outputs = np.where(outputs < 0, 0, outputs)

        return outputs

    def backward_propagation(self, d_outputs, learning_rate, weight_decay=0, momentum=0):
        if self.activation_function == "SIGMOID":
            d_outputs *= (1-self.outputs_activation) * self.outputs_activation
        elif self.activation_function == "SQUASHING":
            d_outputs *= 1.7159 * 2/3 * (1-np.power(np.tanh(2/3 * self.inputs_activation), 2))
        elif self.activation_function == "RELU":
            d_outputs *= np.where(self.inputs_activation < 0, 0, 1)

        d_inputs = np.matmul(d_outputs, self.weights.T)
        d_weights = np.matmul(self.inputs.T, d_outputs)/self.inputs.shape[0]
        d_biases = np.average(d_outputs, axis=0)

        self.v_weights = momentum * self.v_weights - learning_rate * (d_weights + weight_decay * self.weights)
        self.weights += self.v_weights
        self.v_biases = momentum * self.v_biases - learning_rate * (d_biases + weight_decay * self.biases)
        self.biases += self.v_biases

        return d_inputs

    def SDLM(self, d2_outputs, learning_rate):
        if self.activation_function == "SIGMOID":
            d2_outputs *= self.outputs_activation * (1-self.outputs_activation) * (1-2*self.outputs_activation)
        elif self.activation_function == "SQUASHING":
            d2_outputs *= np.power(1.7159 * 2/3 * (1-np.power(np.tanh(2/3 * self.inputs_activation), 2)), 2)

        d2_inputs = np.matmul(d2_outputs, np.power(self.weights.T, 2))
        d2_weights = np.matmul(np.power(self.inputs.T, 2), d2_outputs)
        h = np.sum(d2_weights)/d2_outputs.shape[0]

        self.lr = learning_rate / (0.02 + h)

        return d2_inputs


class RBFLayer(object):
    def __init__(self, ascii_bitmap):
        self.bitmap = ascii_bitmap
        self.inputs = None
        self.labels = None

    def forward_propagation(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        loss = 0.5 * np.sum(np.average(np.power(inputs - self.bitmap[labels], 2), axis=0))
        outputs = np.argmin(np.sum(np.power(np.expand_dims(inputs, axis=-2) - self.bitmap, 2), axis=2), axis=1)
        return loss, outputs

    def backward_propagation(self):
        d_inputs = self.inputs - self.bitmap[self.labels]
        return d_inputs
    
    def SDLM(self):
        d2_inputs = np.ones(self.inputs.shape)
        return d2_inputs


class Softmax(object):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.labels = None

    def forward_propagation(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.outputs = np.exp(inputs)
        self.outputs /= np.sum(self.outputs, axis=1, keepdims=True)
        labels_pred = np.argmax(self.outputs, axis=1)
        loss = -np.log(self.outputs[np.arange(len(self.outputs)), labels])

        return np.average(loss), labels_pred

    def backward_propagation(self):
        d_inputs = np.copy(self.outputs)
        d_inputs[np.arange(len(d_inputs)), self.labels] -= 1

        return d_inputs



