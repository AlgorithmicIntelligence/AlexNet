import numpy as np
import pickle
import AlexNet
import time
import os
import matplotlib.pyplot as plt

# data pre-processing----------------------- to do list
train_data, train_labels, test_data, test_labels = MNIST.load()
train_data = np.pad(train_data, ((0,), (2,), (2,)), "constant")
test_data = np.pad(test_data, ((0,), (2,), (2,)), "constant")

normalize_min, normalize_max = -0.1, 1.175
train_data = train_data / 255 * (normalize_max - normalize_min) + normalize_min
test_data = test_data / 255 * (normalize_max - normalize_min) + normalize_min

train_data = np.expand_dims(train_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)

#data pre-processing------------------------ to do list

#training procedure below------------------------ not debug yet

batch_size_train = 128
weight_decay = 0.0005
momentum = 0.9
learning_rate = np.array([0.01] * 30 + [0.001] * 25 + [0.0001] * 20 + [0.00001] * 15)

restore_weights_path = "./Weights/AlexNetWeightsFinal.pkl"
if os.path.isfile(restore_weights_path):
    with open(restore_weights_path, "rb") as f:
        alexnet = pickle.load(f)
else:
    alexnet = AlexNet.AlexNet()

train_loss_list = list()
train_accuracy_list = list()
test_loss_list = list()
test_accuracy_list = list()
train_data_index = np.arange(len(train_labels))

for epoch in range(len(learning_rate)):
    print("=================================================")
    print("epoch \t", epoch + 1, "\n")
    np.random.shuffle(train_data_index)

    time_start = time.time()

    for i in range(len(train_labels) // batch_size_train):
        loss, labels_ = alexnet.forward_propagation(
            train_data[train_data_index[i * batch_size_train:(i + 1) * batch_size_train]],
            train_labels[train_data_index[i * batch_size_train:(i + 1) * batch_size_train]])
        alexnet.backward_propagation(learning_rate[epoch], weight_decay, momentum)
        time_end = time.time()
        if ((i + 1) % 25) == 0:
            print("Training Data Num ", (i + 1) * batch_size_train)
            print("labels(GT) = ", train_labels[train_data_index[i * batch_size_train:(i + 1) * batch_size_train]])
            print("labels(PD) = ", labels_)
            print("Loss = ", loss, "\tTime Elapsed = ", time_end - time_start, "\n")
            if not os.path.isdir("./Weights/"):
                os.mkdir("./Weights/")
            with open("./Weights/AlexNetWeights_" + str((i + 1) * batch_size_train) + ".pkl", "wb") as f:
                pickle.dump(alexnet, f)

    print("training time = ", time_end - time_start, "\n")

    time_start = time.time()
    loss = 0
    accuracy = 0
    for i in range(len(train_labels) // batch_size_train):
        loss_, label_ = alexnet.forward_propagation(train_data[i * batch_size_train:(i + 1) * batch_size_train],
                                                    train_labels[i * batch_size_train:(i + 1) * batch_size_train])
        loss += loss_
        accuracy += np.sum(np.equal(label_, train_labels[i * batch_size_train:(i + 1) * batch_size_train]))
    loss /= batch_size_train
    accuracy /= len(train_data_index)
    train_loss_list.append(loss)
    train_accuracy_list.append(accuracy)
    time_end = time.time()
    print("training loss = ", loss, "\ttraining accuracy = ", accuracy, "\ttime elapse = ", time_end - time_start, "\n")

    time_start = time.time()
    loss = 0
    accuracy = 0
    for i in range(len(test_labels) // batch_size_train):
        loss_, label_ = alexnet.forward_propagation(test_data[i * batch_size_train:(i + 1) * batch_size_train],
                                                    test_labels[i * batch_size_train:(i + 1) * batch_size_train])
        loss += loss_
        accuracy += np.sum(np.equal(label_, test_labels[i * batch_size_train:(i + 1) * batch_size_train]))
    loss /= batch_size_train
    accuracy /= len(test_labels)
    test_loss_list.append(loss)
    test_accuracy_list.append(accuracy)
    time_end = time.time()
    print("testing loss = ", loss, "\ttesting accuracy = ", accuracy, "\ttime elapse = ", time_end - time_start, "\n")

with open("./Weights/AlexNetWeightsFinal.pkl", "wb") as f:
    pickle.dump(alexnet, f)

x = np.arange(len(learning_rate))
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.plot(x, train_accuracy_list)
plt.plot(x, test_accuracy_list)
plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
plt.show()