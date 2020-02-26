#AlexNet

Handcraft convolutional NN with AlexNet.

Without using general frameworks (tensorflow, pytorch, keras, etc...), I build AlexNet by numpy(Cupy). If
you are interested on the fundamental of AlexNet, it might help you to figure out how it works.

#Note:

I've tried some package which can help the code on GPU, like NUMBA and CUPY, but I found that I change
all numpy func. into cupy, it became slower than original, and NUMBA seems not support with some of numpy function,
if I need to utilize NUMBA, it will take a lot of time, so...I just give up to run on GPU. I found that if we don't
use general frameworks, it's hard to implement Deep Learning NN on GPU in python. Maybe I will write it with C language
in the future.

#Dataset Download:

Download ILSVRC2012's training set and validation set. [Reference](https://blog.csdn.net/weixin_41043240/article/details/80305311)

#The detail of AlexNet:

Data Pre-processing:
1. Resize the short side of input image to 256, then crop it by the middle, the output image will be 256x256. 
2. Random crop 224x224 from original image 256x256.
3. Flip the image by horizonal side in order to augment the dataset.
4. Normalize the value of images from 0~1.
5. Utilize PCA to the dataset for the color jitter augmentation.
6. Pad the image with (2, 1), (2, 1), the output image will be 227x227.

Architecture of CNN:
1. conv11x11 with 96 filters, pad = 'VALID', stride = 4, activation = ReLU, initializer= bias0
2. Local Response Normalization(k=2, n=4, alpha=0.0001, beta=0.75)
3. MaxPooling, kernel size = 3x3, stride = 2
4. (GPU1) conv5x5 with 128 filters, pad = 'SAME', stride = 1, activation = ReLU, initializer= bias1
4. (GPU2) conv5x5 with 128 filters, pad = 'SAME', stride = 1, activation = ReLU, initializer= bias1
5. Local Response Normalization(k=2, n=4, alpha=0.0001, beta=0.75)
6. MaxPooling, kernel size = 3x3, stride = 2
7. conv3x3 with 384 filters, pad = 'SAME', stride = 1, activation = ReLU, initializer= bias0
8. (GPU1) conv3x3 with 192 filters, pad = 'SAME', stride = 1, activation = ReLU, initializer= bias0
9. (GPU2) conv3x3 with 192 filters, pad = 'SAME', stride = 1, activation = ReLU, initializer= bias0
10. (GPU1) conv3x3 with 128 filters, pad = 'SAME', stride = 1, activation = ReLU, initializer= bias0
11. (GPU2) conv3x3 with 128 filters, pad = 'SAME', stride = 1, activation = ReLU, initializer= bias0
12. MaxPooling, kernel size = 3x3, stride = 2
13. FullyConnected with 4096 filters, activation = ReLU.
14. DropOut with keep prob = 0.5
15. FullyConnected with 4096 filters, activation = ReLU.
16. DropOut with keep prob = 0.5
17. FullyConnected with 1000 filters, activation = ReLU.

Loss Function: crossentropy with softmax

batch_size_train = 128
batch_size_test = 256
weight_decay = 0.0005
momentum = 0.9
learning_rate = [0.01] * 30 + [0.001] * 25 + [0.0001] * 20 + [0.00001] * 15

##Reference

1. [The paper of AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)