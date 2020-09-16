import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from nets.retinaface import RetinaFace
from nets.retinanet_training import Generator
from nets.retinanet_training import conf_loss, box_smooth_l1, ldm_smooth_l1
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from utils.utils import BBoxUtility
from utils.anchors import Anchors
from utils.config import cfg_re50, cfg_mnet
from functools import partial
from tqdm import tqdm
import tensorflow as tf
import time

@tf.function
def train_step(imgs, targets1, targets2, targets3, net, optimizer):
    with tf.GradientTape() as tape:
        # 计算loss
        prediction = net(imgs, training=True)
        loss_value1 = box_smooth_l1()(targets1, prediction[0])
        loss_value2 = conf_loss()(targets2, prediction[1])
        loss_value3 = ldm_smooth_l1()(targets3, prediction[2])
        loss_value = loss_value1 + loss_value2 + loss_value3
    grads = tape.gradient(loss_value, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss_value

def fit_one_epoch(net, optimizer, epoch, epoch_size, gen, Epoch):
    total_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_size:
                break
            images, targets1, targets2, targets3 = batch[0], tf.convert_to_tensor(batch[1]),  tf.convert_to_tensor(batch[2]), tf.convert_to_tensor(batch[3])
            loss_value = train_step(images, targets1, targets2, targets3, net, optimizer)
            total_loss += loss_value

            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'total_loss': total_loss/(iteration+1) / (iteration + 1), 
                                'step/s'    : waste_time})
            pbar.update(1)

            start_time = time.time()
            

        print('Finish Validation')
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.4f' % (total_loss/(epoch_size+1)))
        net.save_weights('logs/Epoch%d-Total_Loss%.4f.h5'%((epoch+1),total_loss/(epoch_size+1)))
      

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    #-------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet或者resnet50
    #-------------------------------#
    backbone = "mobilenet"
    training_dataset_path = './data/widerface/train/label.txt'

    if backbone == "mobilenet":
        cfg = cfg_mnet
        freeze_layers = 81
    elif backbone == "resnet50":  
        cfg = cfg_re50
        freeze_layers = 173
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    img_dim = cfg['image_size']

    #-------------------------------#
    #   创立模型
    #-------------------------------#
    model = RetinaFace(cfg, backbone=backbone)
    model_path = "model_data/retinaface_mobilenet025.h5"
    model.load_weights(model_path,by_name=True,skip_mismatch=True)
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    #-------------------------------#
    #   获得先验框和工具箱
    #-------------------------------#
    anchors = Anchors(cfg, image_size=(img_dim, img_dim)).get_anchors()
    bbox_util = BBoxUtility(anchors)

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    for i in range(freeze_layers): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))
    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        Init_epoch = 0
        Freeze_epoch = 50
        # batch_size大小，每次喂入多少数据
        batch_size = 8
        # 最大学习率
        learning_rate_base = 1e-3

        gen = Generator(training_dataset_path,img_dim,batch_size,bbox_util)
        epoch_size = gen.get_len()//batch_size

        if Use_Data_Loader:
            gen = partial(gen.generate, eager=True)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32))
            gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

        else:
            gen = gen.generate(eager=True)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate_base,
            decay_steps=epoch_size,
            decay_rate=0.9,
            staircase=True
        )

        print('Train on {} samples, with batch size {}.'.format(epoch_size, batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        for epoch in range(Init_epoch,Freeze_epoch):
            fit_one_epoch(model, optimizer, epoch, epoch_size, gen, Freeze_epoch)

    for i in range(freeze_layers):
        model.layers[i].trainable = True

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        Freeze_epoch = 50
        Epoch = 100
        # batch_size大小，每次喂入多少数据
        batch_size = 4
        # 最大学习率
        learning_rate_base = 1e-4

        gen = Generator(training_dataset_path,img_dim,batch_size,bbox_util)

        epoch_size = gen.get_len()//batch_size

        if Use_Data_Loader:
            gen = partial(gen.generate, eager=True)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32))
            gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

        else:
            gen = gen.generate(eager=True)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate_base,
            decay_steps=epoch_size,
            decay_rate=0.9,
            staircase=True
        )

        print('Train on {} samples, with batch size {}.'.format(epoch_size, batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        for epoch in range(Freeze_epoch,Epoch):
            fit_one_epoch(model, optimizer, epoch, epoch_size, gen, Epoch)