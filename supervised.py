#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import time
from itertools import islice
from sklearn.metrics import f1_score

from dataloader.cora import load_cora
from minibatch import build_batch_from_nodes as build_batch
from graphsage import GraphSageSupervised as GraphSage

# 每层采样个数
SAMPLE_SIZES = [5, 5] 
INTERNAL_DIM = 128
BATCH_SIZE = 256
TRAINING_STEPS = 100
LEARNING_RATE = 0.5

def run_cora():
    # 载入数据，raw_features 是节点特征矩阵，neigh_dict是每个节点的邻居列表
    num_nodes, raw_features, labels, num_classes, neigh_dict = load_cora()
    # 申明GraphSage模型
    graphsage = GraphSage(raw_features, INTERNAL_DIM, len(SAMPLE_SIZES), num_classes)
    # 划分训练集、测试集
    all_nodes = np.random.permutation(num_nodes)
    train_nodes = all_nodes[:2048]
    test_nodes = all_nodes[2048:]

    # 训练数据生成器
    def generate_training_minibatch(nodes_for_training, all_labels, batch_size):
        while True:
            mini_batch_nodes = np.random.choice ( nodes_for_training
                                                , size=batch_size
                                                , replace=False
                                                )
            batch = build_batch(mini_batch_nodes, neigh_dict, SAMPLE_SIZES)
            labels = all_labels[mini_batch_nodes]
            yield (batch, labels)
    # 定义优化器、损失函数
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    minibatch_generator = generate_training_minibatch(train_nodes, labels, BATCH_SIZE)

    times = []
    # islice：对迭代器minibatch_generator进行切片：[0, TRAINING_STEPS]
    for inputs, inputs_labels in islice(minibatch_generator, 0, TRAINING_STEPS):
        start_time = time.time()
        with tf.GradientTape() as tape:
            predicted = graphsage(inputs)
            loss = loss_fn(tf.convert_to_tensor(inputs_labels), predicted)

        grads = tape.gradient(loss, graphsage.trainable_weights)
        optimizer.apply_gradients(zip(grads, graphsage.trainable_weights))
        end_time = time.time()
        times.append(end_time - start_time)
        print("Loss:", loss.numpy())

    # 测试
    results = graphsage(build_batch(test_nodes, neigh_dict, SAMPLE_SIZES))
    score = f1_score(labels[test_nodes], results.numpy().argmax(axis=1), average="micro")
    print("Validation F1: ", score)
    print("Average batch time: ", np.mean(times))

if __name__ == "__main__":
    run_cora()
