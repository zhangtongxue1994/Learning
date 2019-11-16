"""
------------------------------------------------
File Name: mnist_dnn.py
Description: TensorFlow MNIST example
Reference: https://www.tensorflow.org/tutorials/quickstart/beginner?hl=zh_cn
Author: zhangtongxue
Date: 2019/10/25 16:03
-------------------------------------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import cv2
import tensorflow as tf

if __name__ == '__main__':
    # 载入并准备好 MNIST 数据集，将样本从整数转换为浮点数
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 构建模型后，通过调用compile方法配置其训练过程，为训练选择优化器和损失函数
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 模型使用 fit 方法训练数据
    model.fit(x_train, y_train,
              batch_size=256,
              epochs=2,
              verbose=1,
              callbacks=None,
              validation_split=0.0,
              validation_data=None,
              shuffle=True,
              class_weight=None,
              sample_weight=None,
              initial_epoch=0,
              steps_per_epoch=None,
              validation_steps=None,
              validation_freq=1,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False)

    # 测试
    model.evaluate(x_test, y_test, verbose=0)

    # 保存模型
    tf.saved_model.save(model, './')

    # 加载模型
    tf.saved_model.load('./')

    # 选取前两个样本测试
    outputs = model.predict(x_test[0:2, :, :])
    label = tf.argmax(outputs, axis=1).numpy()
    print(y_test[0:2], label[0], label[1])

    # 显示数字和识别结果
    cv2.imshow(str(label[0]), cv2.resize(x_test[0, :, :], (0, 0), fx=20, fy=20))  # 放大图片
    cv2.waitKey()
