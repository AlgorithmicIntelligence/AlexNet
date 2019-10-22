import Layer
import numpy as np


class AlexNet(object):
    def __init__(self):
        self.C1 = Layer.ConvolutionalLayer([11, 11, 3, 96], pad="VALID", stride=4, activation_function="RELU", initializer="ALEXNET_bias0")
        self.N1 = Layer.LocalResponseNormalization(depth_radius=2, bias=2, alpha=1e-4, beta=0.75)
        self.S1 = Layer.PoolingLayer([3, 3, 96], stride=2, mode="MAX", activation_function="LINEAR")
        self.C2_1 = Layer.ConvolutionalLayer([5, 5, 48, 128], pad="SAME", activation_function="RELU")
        self.C2_2 = Layer.ConvolutionalLayer([5, 5, 48, 128], pad="SAME", activation_function="RELU")
        self.N2 = Layer.LocalResponseNormalization(depth_radius=2, bias=2, alpha=1e-4, beta=0.75)
        self.S2 = Layer.PoolingLayer([3, 3, 256], stride=2, mode="MAX", activation_function="LINEAR")
        self.C3 = Layer.ConvolutionalLayer([3, 3, 256, 384], pad="SAME", activation_function="RELU", initializer="ALEXNET_bias0")
        self.C4_1 = Layer.ConvolutionalLayer([3, 3, 192, 192], pad="SAME", activation_function="RELU")
        self.C4_2 = Layer.ConvolutionalLayer([3, 3, 192, 192], pad="SAME", activation_function="RELU")
        self.C5_1 = Layer.ConvolutionalLayer([3, 3, 192, 128], pad="SAME", activation_function="RELU")
        self.C5_2 = Layer.ConvolutionalLayer([3, 3, 192, 128], pad="SAME", activation_function="RELU")
        self.S5 = Layer.PoolingLayer([3, 3, 256], stride=2, mode="MAX", activation_function="LINEAR")
        self.F6 = Layer.FullyConnectedLayer([9216, 4096], activation_function="RELU")
        self.D6 = Layer.DropOut(keep_prob=0.5)
        self.F7 = Layer.FullyConnectedLayer([4096, 4096], activation_function="RELU")
        self.D7 = Layer.DropOut(keep_prob=0.5)
        self.F8 = Layer.FullyConnectedLayer([4096, 1000], activation_function="RELU")
        self.Output = Layer.Softmax()

    def forward_propagation(self, inputs, labels):
        C1_outputs = self.C1.forward_propagation(inputs)
        N1_outputs = self.N1.forward_propagation(C1_outputs)
        S1_outputs = self.S1.forward_propagation(N1_outputs)
        C2_1_outputs = self.C2_1.forward_propagation(S1_outputs[..., :48])
        C2_2_outputs = self.C2_2.forward_propagation(S1_outputs[..., 48:])
        N2_outputs = self.N2.forward_propagation(np.concatenate([C2_1_outputs, C2_2_outputs], axis=-1))
        S2_outputs = self.S2.forward_propagation(N2_outputs)
        C3_outputs = self.C3.forward_propagation(S2_outputs)
        C4_1_outputs = self.C4_1.forward_propagation(C3_outputs[..., :192])
        C4_2_outputs = self.C4_2.forward_propagation(C3_outputs[..., 192:])
        C5_1_outputs = self.C5_1.forward_propagation(C4_1_outputs)
        C5_2_outputs = self.C5_2.forward_propagation(C4_2_outputs)
        S5_outputs = self.S5.forward_propagation(np.concatenate([C5_1_outputs, C5_2_outputs], axis=-1))
        F6_outputs = self.F6.forward_propagation(S5_outputs)
        D6_outputs = self.D6.forward_propagation(F6_outputs)
        F7_outputs = self.F7.forward_propagation(D6_outputs)
        D7_outputs = self.D7.forward_propagation(F7_outputs)
        F8_outputs = self.F8.forward_propagation(D7_outputs)
        loss, labels_pred = self.Output.forward_propagation(F8_outputs)

        return loss, labels_pred

    def backward_propagation(self, learning_rate, weight_decay, momentum):
        d_inputs_Output = self.Output.backward_propagation()
        d_inputs_F8 = self.F8.backward_propagation(d_inputs_Output, learning_rate, weight_decay, momentum)
        d_inputs_D7 = self.F8.backward_propagation(d_inputs_F8, learning_rate, weight_decay, momentum)
        d_inputs_F7 = self.F8.backward_propagation(d_inputs_D7, learning_rate, weight_decay, momentum)
        d_inputs_D6 = self.F8.backward_propagation(d_inputs_F7, learning_rate, weight_decay, momentum)
        d_inputs_F6 = self.F8.backward_propagation(d_inputs_D6, learning_rate, weight_decay, momentum)
        d_inputs_S5 = self.F8.backward_propagation(d_inputs_F6, learning_rate, weight_decay, momentum)
        d_inputs_C5_2 = self.F8.backward_propagation(d_inputs_S5[..., d_inputs_S5.shape[-1]/2:], learning_rate, weight_decay, momentum)
        d_inputs_C5_1 = self.F8.backward_propagation(d_inputs_S5[..., :d_inputs_S5.shape[-1]/2], learning_rate, weight_decay, momentum)
        d_inputs_C4_2 = self.F8.backward_propagation(d_inputs_C5_2, learning_rate, weight_decay, momentum)
        d_inputs_C4_1 = self.F8.backward_propagation(d_inputs_C5_1, learning_rate, weight_decay, momentum)
        d_inputs_C3 = self.F8.backward_propagation(np.concatenate([d_inputs_C4_1, d_inputs_C4_2], axis=-1), learning_rate, weight_decay, momentum)
        d_inputs_S2 = self.F8.backward_propagation(d_inputs_C3, learning_rate, weight_decay, momentum)
        d_inputs_N2 = self.F8.backward_propagation(d_inputs_S2, learning_rate, weight_decay, momentum)
        d_inputs_C2_2 = self.F8.backward_propagation(d_inputs_N2[..., d_inputs_N2.shape[-1]/2:], learning_rate, weight_decay, momentum)
        d_inputs_C2_1 = self.F8.backward_propagation(d_inputs_N2[..., :d_inputs_N2.shape[-1]/2], learning_rate, weight_decay, momentum)
        d_inputs_S1 = self.F8.backward_propagation(np.concatenate([d_inputs_C2_1, d_inputs_C2_2], axis=-1), learning_rate, weight_decay, momentum)
        d_inputs_N1 = self.F8.backward_propagation(d_inputs_S1, learning_rate, weight_decay, momentum)
        self.F8.backward_propagation(d_inputs_N1, learning_rate, weight_decay, momentum)