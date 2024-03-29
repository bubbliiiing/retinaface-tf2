import datetime
import os
from functools import partial

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.layers import Conv2D, Dense, DepthwiseConv2D
from tensorflow.keras.optimizers import SGD, Adam

from nets.retinaface import RetinaFace
from nets.retinaface_training import (box_smooth_l1, conf_loss,
                                      get_lr_scheduler, ldm_smooth_l1)
from utils.anchors import Anchors
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint)
from utils.config import cfg_mnet, cfg_re50
from utils.dataloader import Generator
from utils.utils_bbox import BBoxUtility
from utils.utils_fit import fit_one_epoch

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":
    #--------------------------------#
    #   是否使用eager模式训练
    #--------------------------------#
    eager  = False
    #--------------------------------#
    #   获得训练用的人脸标签与坐标
    #--------------------------------#
    training_dataset_path = 'data/widerface/train/label.txt'
    #-------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet或者resnet50
    #-------------------------------#
    backbone = "mobilenet"  
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的训练参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path = "model_data/mobilenet_2_5_224_tf_no_top.h5"
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #      
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从主干网络的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 150，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 150，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（不冻结）
    #       其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合目标检测，需要更多的训练跳出局部最优解。
    #             UnFreeze_Epoch可以在150-300之间调整，YOLOV5和YOLOX均推荐使用300。
    #             Adam相较于SGD收敛的快一些。因此UnFreeze_Epoch理论上可以小一点，但依然推荐更多的Epoch。
    #   （二）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 16
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 150
    Unfreeze_batch_size = 8
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    #------------------------------------------------------------------#
    Freeze_Train        = False
    
    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    #------------------------------------------------------------------#
    save_period         = 1
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------------------#
    num_workers         = 1

    if backbone == "mobilenet":
        cfg             = cfg_mnet
        freeze_layers   = 81
    elif backbone == "resnet50":  
        cfg             = cfg_re50
        freeze_layers   = 173
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    model       = RetinaFace(cfg, backbone=backbone)
    if model_path != '':
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

    #-------------------------------#
    #   获得先验框和工具箱
    #-------------------------------#
    anchors     = Anchors(cfg, image_size=(cfg['train_image_size'], cfg['train_image_size'])).get_anchors()
    bbox_util   = BBoxUtility(anchors)

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        if Freeze_Train:
            for i in range(freeze_layers): model.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        
        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func   = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    
        train_dataloader    = Generator(training_dataset_path, cfg['train_image_size'], batch_size, bbox_util)

        optimizer = {
            'adam'  : Adam(lr = Init_lr_fit, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]

        if eager:
            start_epoch     = Init_Epoch
            end_epoch       = UnFreeze_Epoch
            UnFreeze_flag   = False
            
            gen         = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32))
            gen         = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            loss_history    = LossHistory(log_dir)
            #---------------------------------------#
            #   开始模型训练
            #---------------------------------------#
            for epoch in range(start_epoch, end_epoch):
                #---------------------------------------#
                #   如果模型有冻结学习部分
                #   则解冻，并设置参数
                #---------------------------------------#
                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_size      = Unfreeze_batch_size

                    #-------------------------------------------------------------------#
                    #   判断当前batch_size，自适应调整学习率
                    #-------------------------------------------------------------------#
                    nbs             = 64
                    lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                    lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                    #---------------------------------------#
                    #   获得学习率下降的公式
                    #---------------------------------------#
                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                    for i in range(len(model.layers)): 
                        model.layers[i].trainable = True

                    train_dataloader.batch_size    = batch_size

                    gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32))
                    gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    
                    UnFreeze_flag = True
                    
                lr = lr_scheduler_func(epoch)
                K.set_value(optimizer.lr, lr)

                fit_one_epoch(model, loss_history, optimizer, epoch, train_dataloader.get_len() // batch_size, gen, end_epoch, cfg, save_period, save_dir)
                
                train_dataloader.on_epoch_end()
            
        else:
            start_epoch = Init_Epoch
            end_epoch   = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch
            
            model.compile(optimizer = optimizer, loss={
                        'bbox_reg'  : box_smooth_l1(weights = cfg['loc_weight']),
                        'cls'       : conf_loss(),
                        'ldm_reg'   : ldm_smooth_l1()
                    })
            #-------------------------------------------------------------------------------#
            #   训练参数的设置
            #   logging         用于设置tensorboard的保存地址
            #   checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
            #   lr_scheduler       用于设置学习率下降的方式
            #   early_stopping  用于设定早停，loss多次不下降自动结束训练，表示模型基本收敛
            #-------------------------------------------------------------------------------#
            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            logging         = TensorBoard(log_dir)
            loss_history    = LossHistory(log_dir)
            checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}.h5"), 
                                    monitor = 'loss', save_weights_only = True, save_best_only = False, period = save_period)
            early_stopping  = EarlyStopping(monitor='loss', min_delta = 0, patience = 10, verbose = 1)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            callbacks       = [logging, loss_history, checkpoint, lr_scheduler]

            if start_epoch < end_epoch:
                model.fit_generator(
                    generator           = train_dataloader,
                    steps_per_epoch     = train_dataloader.get_len() // batch_size,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if Freeze_Train:
                batch_size  = Unfreeze_batch_size
                start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
                end_epoch   = UnFreeze_Epoch
                    
                #-------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
                callbacks       = [logging, loss_history, checkpoint, lr_scheduler]
                
                for i in range(len(model.layers)): 
                    model.layers[i].trainable = True
                model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

                train_dataloader.batch_size    = Unfreeze_batch_size

                model.fit_generator(
                    generator           = train_dataloader,
                    steps_per_epoch     = train_dataloader.get_len() // batch_size,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )
