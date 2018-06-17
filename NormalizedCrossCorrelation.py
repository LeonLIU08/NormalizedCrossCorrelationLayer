from keras.engine import Layer
from keras import backend as K
import tensorflow as tf
from keras.layers import Input
from keras.models import Model

import numpy as np

class Normlized_Cross_Correlation(Layer):
    # try to speed up
    def __init__(self, k=0, d=1, s1=1, s2=2, **kwargs):
        super(Normlized_Cross_Correlation, self).__init__(**kwargs)

        self.D = 2 * d + 1 # output size
        self.K = 2 * k + 1
        self.d = d
        self.k = k
        self.s1 = s1
        self.s2 = s2
        self.D_stride = self.s2*(self.D-1)+self.K  # make sure the output is DxD

    def build(self, input_shape):
        super(Normlized_Cross_Correlation, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1] / self.s1, input_shape[0][2] / self.s1, self.D ** 2)

    def call(self, x, mask=None):
        input_1, input_2 = x
        input_shape = input_1._keras_shape

        assert input_shape == input_2._keras_shape

        self.H = input_shape[1]
        self.W = input_shape[2]
        self.C = input_shape[3]

        padding1 = K.spatial_2d_padding(input_1,
                                        padding=((self.k, self.k+1), (self.k, self.k+1)),
                                        data_format='channels_last')
        padding2 = K.spatial_2d_padding(input_2,
                                        padding=(
                                            ((self.D_stride-1)/2, (self.D_stride-1)/2+1),
                                            ((self.D_stride-1)/2, (self.D_stride-1)/2+1)),
                                        data_format='channels_last')
        # padding1&2: [nS, w, h, c]

        out = tf.scan(self.single_sample_corr,
                      elems=[padding2, padding1],
                      initializer=(K.zeros((self.H / self.s1, self.W / self.s1, self.D ** 2))))

        return out

    def single_sample_corr(self, previous, features):
        fea1, fea2 = features  # fea1: the displacement, fea2: the kernel

        displaces = []
        kernels = []
        for i in range(self.H / self.s1):
            for j in range(self.W / self.s1):
                slice_h_ker = slice(i * self.s1, i * self.s1 + self.K)
                slice_w_ker = slice(j * self.s1, j * self.s1 + self.K)

                slice_h_dis = slice(i * self.s1, i * self.s1 + self.D_stride)
                slice_w_dis = slice(j * self.s1, j * self.s1 + self.D_stride)

                kernels.append(fea2[slice_h_ker, slice_w_ker, :])
                displaces.append(fea1[slice_h_dis, slice_w_dis, :])

        displaces = K.stack(displaces, axis=0)  # [WH/s1s1, D_stride, D_stride, C]
        kernels = K.stack(kernels, axis=0)
        kernels = K.permute_dimensions(kernels, (1, 2, 3, 0))  # [K, K, C, WH/s1s1]
        corr = self.correlation(kernels, displaces, s2=self.s2)  # [WH/s1s1, D, D, WH/s1s1]

        # get diag
        b = []
        for i in range(self.H*self.W/self.s1**2):
            b.append(corr[i, :, :, i])
        a = K.stack(b, axis=0)  # [WH/s1s1, D, D]

        out = K.reshape(a, (self.H / self.s1, self.W / self.s1, self.D ** 2))

        return out

    def correlation(self, kernel, displace, s2=1):
        dis_std = K.std(displace, axis=[3], keepdims=True)
        ker_std = K.std(kernel, axis=[2], keepdims=True)
        displace = (displace - K.mean(displace, axis=[3], keepdims=True))/(dis_std+0.000001)
        kernel = (kernel - K.mean(kernel, axis=[2], keepdims=True))/(ker_std+0.0000001)
        # print kernel, displace
        return K.conv2d(displace, kernel, strides=(s2, s2), padding='valid', data_format='channels_last')


if __name__ == '__main__':
    from skimage.io import imread

    img = imread('image.bmp')
    print img.shape

    img = img[np.newaxis, :, :, :]
    img2 = np.zeros((32, 64, 64, 3))
    for i in range(32):
        img2[i, :, :, :] = img[:, 164:228, 164:228, :]*(np.random.random()*0.3+0.7)
    img2 /= 255.

    input1 = Input(shape=(64, 64, 3))
    input2 = Input(shape=(64, 64, 3))
    crop = Normlized_Cross_Correlation(k=2, d=5, s1=2, s2=1)([input1, input2])

    model = Model([input1, input2], crop)

    corr = model.predict([img2, img2])

    print(corr.shape)

