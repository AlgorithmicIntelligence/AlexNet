import AlexNet
from DataProcessing import ILSVRC2012
import numpy as np
import pickle
import time
import os
import matplotlib.pyplot as plt

trainingSet = ILSVRC2012('/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_val', './dirname_to_classname')
testingSet = ILSVRC2012('/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_val', './dirname_to_classname')

batch_size_train = 128
batch_size_test = 256
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
train_data_index = np.arange(len(trainingSet))

for epoch in range(len(learning_rate)):
    print("=================================================")
    print("epoch \t", epoch + 1, "\n")
    np.random.shuffle(train_data_index)

    time_start = time.time()

    for i in range(len(trainingSet) // batch_size_train):
        # runtime:270 secs per batch.
        train_imgs, train_labels = trainingSet.__getitem__(train_data_index[i*batch_size_train:(i+1)*batch_size_train])
        loss, labels_ = alexnet.forward_propagation(train_imgs, train_labels)
        alexnet.backward_propagation(learning_rate[epoch], weight_decay, momentum)
        
        time_end = time.time()
        if ((i + 1) % 1) == 0:
            print("Training Data Num ", (i + 1) * batch_size_train)
            print("labels(GT) = ", train_labels[:10])
            print("labels(PD) = ", labels_[:10])
            print("Loss = ", loss, "\tTime Elapsed = ", time_end - time_start, "\n")
        if ((i + 1) % 10) == 0:
            if not os.path.isdir("./Weights/"):
                os.mkdir("./Weights/")
            with open("./Weights/AlexNetWeights_" + str((i + 1) * batch_size_train) + ".pkl", "wb") as f:
                pickle.dump(alexnet, f)

    print("training time = ", time_end - time_start, "\n")

    time_start = time.time()
    loss = 0
    accuracy = 0
    for i in range(len(testingSet) // batch_size_test):
        test_imgs, test_labels = testingSet.__getitem__(range(i * batch_size_test,(i + 1) * batch_size_test))
        loss_, label_ = alexnet.forward_propagation(test_imgs, test_labels)
        loss += loss_
        accuracy += np.sum(np.equal(label_, test_labels))
    loss /= (len(testingSet) // batch_size_test)
    accuracy /= len(testingSet)
    test_loss_list.append(loss)
    test_accuracy_list.append(accuracy)
    time_end = time.time()
    print("training loss = ", loss, "\ttraining accuracy = ", accuracy, "\ttime elapse = ", time_end - time_start, "\n")

with open("./Weights/AlexNetWeightsFinal.pkl", "wb") as f:
    pickle.dump(alexnet, f)

x = np.arange(len(learning_rate))
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.plot(x, train_accuracy_list)
plt.plot(x, test_accuracy_list)
plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
plt.show()