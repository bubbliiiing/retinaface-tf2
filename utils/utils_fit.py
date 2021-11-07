import tensorflow as tf
from nets.retinaface_training import box_smooth_l1, conf_loss, ldm_smooth_l1
from tqdm import tqdm


# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(imgs, targets1, targets2, targets3, net, optimizer, cfg):
        with tf.GradientTape() as tape:
            prediction = net(imgs, training=True)
            loss_value1 = box_smooth_l1(weights = cfg['loc_weight'])(targets1, prediction[0])
            loss_value2 = conf_loss()(targets2, prediction[1])
            loss_value3 = ldm_smooth_l1()(targets3, prediction[2])
            loss_value  = loss_value1 + loss_value2 + loss_value3
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

def fit_one_epoch(net, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cfg):
    train_step  = get_train_step_fn()

    total_loss = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_step:
                break
            images, targets1, targets2, targets3 = batch[0], tf.convert_to_tensor(batch[1]),  tf.convert_to_tensor(batch[2]), tf.convert_to_tensor(batch[3])
            loss_value = train_step(images, targets1, targets2, targets3, net, optimizer, cfg)
            total_loss += loss_value

            pbar.set_postfix(**{'total_loss': total_loss.numpy()/(iteration+1), 
                                'lr'        : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)

    logs = {'loss': total_loss.numpy() / (epoch_step+1)}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f' % (total_loss / epoch_step))
    net.save_weights('logs/ep%03d-loss%.3f.h5' % (epoch + 1, total_loss / epoch_step))
