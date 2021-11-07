import tensorflow as tf
import tensorflow.keras


def softmax_loss(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 1e-7)
    softmax_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred),
                                    axis=-1)
    return softmax_loss

def conf_loss(neg_pos_ratio = 7,negatives_for_hard = 100):
    def _conf_loss(y_true, y_pred):
        #-------------------------------#
        #   取出先验框的数量
        #-------------------------------#
        num_boxes = tf.cast((tf.shape(y_true)[1]),tf.float32)
        
        labels         = y_true[:, :, :-1]
        classification = y_pred
        #-------------------------------#
        #   计算所有先验框的损失cls_loss
        #-------------------------------#
        cls_loss = softmax_loss(labels, classification)
        
        #-------------------------------#
        #   获取作为正样本的先验框与损失
        #-------------------------------#
        num_pos = tf.reduce_sum(y_true[:, :, -1], axis=-1)
        pos_conf_loss = tf.reduce_sum(cls_loss * y_true[:, :, -1],
                                      axis=1)
        #-------------------------------#
        #   获取一定的负样本
        #-------------------------------#
        num_neg = tf.minimum(neg_pos_ratio * num_pos, num_boxes - num_pos)

        #-------------------------------#
        #   找到了哪些值是大于0的
        #-------------------------------#
        pos_num_neg_mask = tf.greater(num_neg, 0)
        has_min = tf.cast(tf.reduce_any(pos_num_neg_mask), tf.float32)
        num_neg = tf.concat([num_neg,[(1 - has_min) * negatives_for_hard]], axis=0)

        # --------------------------------------------- #
        #   求整个batch应该的负样本数量总和
        # --------------------------------------------- #
        num_neg_batch = tf.reduce_sum(tf.boolean_mask(num_neg, tf.greater(num_neg, 0)))
        num_neg_batch = tf.cast(num_neg_batch, tf.int32)

        # --------------------------------------------- #
        #   把不是背景的概率求和，求和后的概率越大
        #   代表越难分类。
        # --------------------------------------------- #
        max_confs = tf.reduce_sum(y_pred[:, :, 1:], axis=2)
        # --------------------------------------------------- #
        #   只有没有包含物体的先验框才得到保留
        #   我们在整个batch里面选取最难分类的num_neg_batch个
        #   先验框作为负样本。
        # --------------------------------------------------- #
        max_confs = tf.reshape(max_confs * (1 - y_true[:, :, -1]), [-1])
        _, indices = tf.nn.top_k(max_confs, k=num_neg_batch)

        neg_conf_loss = tf.gather(tf.reshape(cls_loss, [-1]), indices)

        # 进行归一化
        num_pos     = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
        total_loss  = tf.reduce_sum(pos_conf_loss) + tf.reduce_sum(neg_conf_loss)
        total_loss /= tf.reduce_sum(num_pos)
        return total_loss
    return _conf_loss
    
def box_smooth_l1(sigma=1, weights=1):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        #------------------------------------#
        #   取出作为正样本的先验框
        #------------------------------------#
        indices           = tf.where(tensorflow.keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        #------------------------------------#
        #   计算 smooth L1 loss
        #------------------------------------#
        regression_diff = regression - regression_target
        regression_diff = tensorflow.keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            tensorflow.keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tensorflow.keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        normalizer = tensorflow.keras.backend.maximum(1, tensorflow.keras.backend.shape(indices)[0])
        normalizer = tensorflow.keras.backend.cast(normalizer, dtype=tensorflow.keras.backend.floatx())
        loss = tensorflow.keras.backend.sum(regression_loss) / normalizer

        return loss * weights

    return _smooth_l1

def ldm_smooth_l1(sigma=1):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # 找到正样本
        indices           = tf.where(tensorflow.keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = tensorflow.keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            tensorflow.keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tensorflow.keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        normalizer = tensorflow.keras.backend.maximum(1, tensorflow.keras.backend.shape(indices)[0])
        normalizer = tensorflow.keras.backend.cast(normalizer, dtype=tensorflow.keras.backend.floatx())
        loss = tensorflow.keras.backend.sum(regression_loss) / normalizer

        return loss

    return _smooth_l1
