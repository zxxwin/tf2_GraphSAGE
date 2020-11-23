#!/usr/bin/env python3

import tensorflow as tf

init_fn = tf.keras.initializers.GlorotUniform


# 从所有特征向量中取出每个batch需要的数据
class RawFeature(tf.keras.layers.Layer):
    def __init__(self, features, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.features = tf.constant(features)
        
    def call(self, nodes):
        return tf.gather(self.features, nodes)

# 平均值聚合器
class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, src_dim, dst_dim, activ=True, **kwargs):
        """
        src_dim: 输入维度
        dst_dim: 输出维度
        """
        super().__init__(**kwargs)
        self.activ_fn = tf.nn.relu if activ else tf.identity
        self.w = self.add_weight( name = kwargs["name"] + "_weight"
                                , shape = (src_dim*2, dst_dim)
                                , dtype = tf.float32
                                , initializer = init_fn
                                , trainable = True
                                )
    
    def call(self, dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat):
        """
        dstsrc_features: 第 K-1 层所有节点的 embedding
        dstsrc2dst: 当前层的目标节点
        dstsrc2src: 当前层的邻居节点
        dif_mat: 归一化矩阵
        """
        # 从当前batch所有节点中取出目标节点
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)
        # 从当前batch所有节点中取出邻居节点
        src_features = tf.gather(dstsrc_features, dstsrc2src)
        # 对邻居节点加权求和，得到邻居节点embedding之和的均值
        # (batch_size, num_neighbors) x (num_neighbors, src_dim)
        aggregated_features = tf.matmul(dif_mat, src_features)
        # 将第k-1层的embedding与聚合结果进行拼接
        concatenated_features = tf.concat([aggregated_features, dst_features], 1)
        # 乘上权重矩阵 w 
        x = tf.matmul(concatenated_features, self.w)
        return self.activ_fn(x)




class GraphSageBase(tf.keras.Model):

    def __init__(self, raw_features, internal_dim, num_layers, last_has_activ):

        assert num_layers > 0, 'illegal parameter "num_layers"'
        assert internal_dim > 0, 'illegal parameter "internal_dim"'

        super().__init__()

        self.input_layer = RawFeature(raw_features, name="raw_feature_layer")

        self.seq_layers = []
        for i in range (1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            input_dim = internal_dim if i > 1 else raw_features.shape[-1]
            has_activ = last_has_activ if i == num_layers else True
            aggregator_layer = MeanAggregator ( input_dim
                                              , internal_dim
                                              , name=layer_name
                                              , activ = has_activ
                                              )
            self.seq_layers.append(aggregator_layer)

    def call(self, minibatch):
        # 取出当前batch中用到的所有节点
        x = self.input_layer(tf.squeeze(minibatch.src_nodes))
        for aggregator_layer in self.seq_layers:
            # 逐层聚合
            x = aggregator_layer ( x
                                 , minibatch.dstsrc2srcs.pop()
                                 , minibatch.dstsrc2dsts.pop()
                                 , minibatch.dif_mats.pop()
                                 )
        return x # shape: (batch_size, src_dim)

class GraphSageSupervised(GraphSageBase):
    def __init__(self, raw_features, internal_dim, num_layers, num_classes):
        super().__init__(raw_features, internal_dim, num_layers, True)
        self.classifier = tf.keras.layers.Dense ( num_classes
                                                , activation = tf.nn.softmax
                                                , use_bias = False
                                                , kernel_initializer = init_fn
                                                , name = "classifier"
                                                )

    def call(self, minibatch):
        return self.classifier( super().call(minibatch) )



# 无监督学习的损失函数
@tf.function 
def compute_uloss(embeddingA, embeddingB, embeddingN, neg_weight):
    # 计算边的两个节点的内积，得到相似度
    # (batch_size, emb_dim) * (batch_size, emb_dim) 
    # -> (batch_size, emb_dim) -> (batch_size, )
    pos_affinity = tf.reduce_sum ( tf.multiply ( embeddingA, embeddingB ), axis=1 )
    # 相当于每个节点都和负样本的 embedding 计算内积，
    # 得到每个节点与每个负样本的相似度
    # (batch_size, emb_dim) x (emb_dim, neg_size) -> (batch_size, neg_size)
    neg_affinity = tf.matmul ( embeddingA, tf.transpose ( embeddingN ) )
    # shape: (batch_size, )
    pos_xent = tf.nn.sigmoid_cross_entropy_with_logits ( tf.ones_like(pos_affinity)
                                                       , pos_affinity
                                                       , "positive_xent" )
    # shape: (batch_size, neg_num)
    neg_xent = tf.nn.sigmoid_cross_entropy_with_logits ( tf.zeros_like(neg_affinity)
                                                       , neg_affinity
                                                       , "negative_xent" )
    # 对neg_xent所有元素求和后乘上权重
    weighted_neg = tf.multiply ( neg_weight, tf.reduce_sum(neg_xent) )
    # 对两个 loss 进行累加
    batch_loss = tf.add ( tf.reduce_sum(pos_xent), weighted_neg )

    # loss 除以样本个数
    return tf.divide ( batch_loss, embeddingA.shape[0] )

class GraphSageUnsupervised(GraphSageBase):
    def __init__(self, raw_features, internal_dim, num_layers, neg_weight):
        super().__init__(raw_features, internal_dim, num_layers, False)
        self.neg_weight = neg_weight

    def call(self, minibatch):
        # 对 embedding 结果进行正则化
        embeddingABN = tf.math.l2_normalize(super().call(minibatch), 1)
        # 损失函数的计算
        self.add_loss (
                compute_uloss ( tf.gather(embeddingABN, minibatch.dst2batchA)
                              , tf.gather(embeddingABN, minibatch.dst2batchB)
                              , tf.boolean_mask(embeddingABN, minibatch.dst2batchN)
                              , self.neg_weight
                              )
                )
        return embeddingABN
