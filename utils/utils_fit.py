import os

import torch
from tqdm import tqdm

from utils.utils import get_lr
        
def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)
            loss_value      = yolo_loss(outputs, targets, images)

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)
                loss_value      = yolo_loss(outputs, targets, images)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train_eval(images)
            loss_value      = yolo_loss(outputs, targets, images)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))


#---------------------------------------#
#   冻结ACCR模块
#---------------------------------------#
def freeze_parameters_clip(model):
    for name, param in model.named_parameters():
        if 'contrast' in name:
            param.requires_grad = False

def fit_one_epoch_decouple(model_train, model, ema, yolo_loss, decouple_loss, correlation_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss = 0
    val_loss = 0

    yololoss = 0
    decoupleloss = 0
    correlationloss = 0

    # 三个loss的比例系数
    correlation_ratio  = 0.1
    decouple_ratio     = 0.01
    value_ratio        = 10

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):



        if iteration >= epoch_step:
            break
        #冻结clip
        freeze_parameters_clip(model)
        images, targets, captions = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, object_feature, noise_feature, logits_per_image = model_train(images, captions)

            object_feature3, object_feature4 = object_feature
            noise_feature3, noise_feature4 = noise_feature
            logits_per_image3, logits_per_image4 = logits_per_image

            loss_correlation3 = correlation_loss(object_feature3, noise_feature3)
            loss_decouple3 = decouple_loss(logits_per_image3, images.device)

            loss_correlation4 = correlation_loss(object_feature4, noise_feature4)
            loss_decouple4 = decouple_loss(logits_per_image4, images.device)

            # loss_correlation5 = correlation_loss(object_feature5, noise_feature5)
            # loss_decouple5 = decouple_loss(logits_per_image5, images.device)

            loss_correlation = (loss_correlation3 + loss_correlation4) / 2.0
            loss_decouple = (loss_decouple3 + loss_decouple4) / 2.0
            loss_value = yolo_loss(outputs, targets, images)

            loss_correlation = correlation_ratio * loss_correlation
            loss_decouple = decouple_ratio * loss_decouple
            loss_value = value_ratio * loss_value

            # print(loss_value)
            # print(loss_correlation)
            # print(loss_decouple)

            loss_total = loss_value + loss_correlation + loss_decouple
            # print(loss_total)
            # print(loss_total.shape)
            # ----------------------#
            #   反向传播
            # ----------------------#
            loss_total.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs, object_feature, noise_feature, logits_per_image = model_train(images, captions)
                object_feature3, object_feature4 = object_feature
                noise_feature3, noise_feature4 = noise_feature
                logits_per_image3, logits_per_image4 = logits_per_image

                loss_correlation3 = correlation_loss(object_feature3, noise_feature3)
                loss_decouple3 = decouple_loss(logits_per_image3, images.device)

                loss_correlation4 = correlation_loss(object_feature4, noise_feature4)
                loss_decouple4 = decouple_loss(logits_per_image4, images.device)

                loss_correlation = (loss_correlation3 + loss_correlation4)/2.0
                loss_decouple = (loss_decouple3 + loss_decouple4)/2.0
                loss_value = yolo_loss(outputs, targets, images)

                loss_correlation = correlation_ratio * loss_correlation
                loss_decouple = decouple_ratio * loss_decouple
                loss_value = value_ratio * loss_value

                loss_total = loss_value + loss_correlation + loss_decouple
                # loss_total = loss_total.sum()


            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_total.item()
        yololoss += loss_value.item()
        decoupleloss += loss_decouple.item()
        correlationloss += loss_correlation.item()

        if local_rank == 0:
            pbar.set_postfix(**{'totalloss': loss / (iteration + 1),
                                'yololoss': yololoss / (iteration + 1),
                                'decoupleloss': decoupleloss / (iteration + 1),
                                'correlationloss': correlationloss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        # 冻结clip
        freeze_parameters_clip(model)
        images, targets, captions = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, object_feature, noise_feature, logits_per_image = model_train(images, captions)
            object_feature3, object_feature4 = object_feature
            noise_feature3, noise_feature4 = noise_feature
            logits_per_image3, logits_per_image4 = logits_per_image

            loss_correlation3 = correlation_loss(object_feature3, noise_feature3)
            loss_decouple3 = decouple_loss(logits_per_image3, images.device)

            loss_correlation4 = correlation_loss(object_feature4, noise_feature4)
            loss_decouple4 = decouple_loss(logits_per_image4, images.device)

            # loss_correlation5 = correlation_loss(object_feature5, noise_feature5)
            # loss_decouple5 = decouple_loss(logits_per_image5, images.device)

            loss_correlation = (loss_correlation3 + loss_correlation4) / 2.0
            loss_decouple = (loss_decouple3 + loss_decouple4) / 2.0
            loss_value = yolo_loss(outputs, targets, images)

            loss_correlation = correlation_ratio * loss_correlation
            loss_decouple = decouple_ratio * loss_decouple
            loss_value = value_ratio * loss_value

            loss_total = loss_value + loss_correlation + loss_decouple

        val_loss += loss_total.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))