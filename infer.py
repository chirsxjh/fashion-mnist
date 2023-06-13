'''
Author: your name
Date: 2023-03-06 16:53:31
LastEditTime: 2023-06-08 15:11:02
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \Fashion-MNIST\net.py
'''
import tensorflow as tf
import os
import argparse
import numpy as np
import gzip

def load_data(data_folder):
    files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder,fname))
        
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    
    return (x_train, y_train), (x_test, y_test)
        


#定义可以接收的训练作业运行参数
parser = argparse.ArgumentParser(description='TensorFlow quick start')
parser.add_argument('--data_url', type=str,default=False, help='path where the dataset is saved')
parser.add_argument('--train_url', type=str, default=False,
                        help='mnist model path')
parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
args = parser.parse_args()

# 加载数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
#data_url  /sssqqqq/tensorflow/data/ train_url /sssqqqq/tensorflow/output/
model_path = args.data_url



(train_images, train_labels), (test_images, test_labels) = load_data(model_path)
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 图像都会被映射到一个标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0

test_images = test_images / 255.0

#构建神经网络需要先配置模型的层，然后再编译模型。
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

#编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#向模型馈送数据
model.fit(train_images, train_labels, epochs=10)

#比较模型在测试数据集上的表现
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
if args.save_model:
    print('training Compeleted ! Start to save model file to obs...')
    tf.saved_model.save(model,args.train_url,)
    print('Successfully save model file to obs!')










