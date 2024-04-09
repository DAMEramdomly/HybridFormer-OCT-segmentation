from loss.utils import *
import tqdm
from torch import optim
from model2.connect_loss import connect_loss

def save_checkpoint(state, filename = './checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)

def train_epoch(args, model, loader, optimizer, epoch, lr):
    losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()
    iou1 = AverageMeter()
    iou2 = AverageMeter()
    iou3 = AverageMeter()
    iou4 = AverageMeter()
    dice1 = AverageMeter()
    dice2 = AverageMeter()
    dice3 = AverageMeter()
    dice4 = AverageMeter()
    sen1 = AverageMeter()
    sen2 = AverageMeter()
    sen3 = AverageMeter()
    sen4 = AverageMeter()
    spe1 = AverageMeter()
    spe2 = AverageMeter()
    spe3 = AverageMeter()
    spe4 = AverageMeter()


    model.train()
    tq = tqdm.tqdm(total=len(loader))#设置进度条
    tq.set_description('epoch %d, lr %f' % (epoch, lr))
    for batch_index, (data, target) in enumerate(loader):
        data = data.cuda()
        target = target.float()
        target = target.cuda()

        if args.loss=='diceloss':
            computeloss=diceloss()
        if args.loss=='diceloss_classloss':
            if epoch>0:                
                computeloss=diceloss_classloss()
            else:
                computeloss=diceloss()

        elif args.loss=='dice_bceloss':
            computeloss=Dice_CrossEntropy_loss()
        elif args.loss == "hybrid_loss":
            computeloss = hybrid_loss()
        elif args.loss == "similarity_loss":
            computeloss = similarity_loss()
        elif args.loss == "ce_loss":
            computeloss = ce_loss()
        hroi_trans = torch.zeros([1, 5, 512, 512])
        vert_trans = torch.zeros([1, 5, 256, 256])
        if args.loss == "connect_loss":
            computeloss1 = connect_loss(args, hori_translation=hroi_trans, verti_translation=vert_trans)
            computeloss2 = connect_loss(args, hori_translation=hroi_trans, verti_translation=vert_trans)
        #output1, output2 = model(data)

        #loss1 = computeloss1.forward(output1, target.long())
        #loss2 = computeloss2.forward(output2, target.long())
        #loss = loss1 + 0.3 * loss2
        output = model(data)
        loss = computeloss.forward(output, target.long())
        pred_seg = torch.argmax(F.softmax(output.detach(), 1), dim=1).int()

        batch_size = target.size(0)
        losses.update(loss.data, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tq.update(1)

        dice, pre, jacc, sen, spe = eval_multi_seg(pred_seg, target.long())

        dice_1 = dice[0]
        dice_2 = dice[1]
        dice_3 = dice[2]
        dice_4 = dice[3]
        acc_1 = pre[0]
        acc_2 = pre[1]
        acc_3 = pre[2]
        acc_4 = pre[3]
        iou_1 = jacc[0]
        iou_2 = jacc[1]
        iou_3 = jacc[2]
        iou_4 = jacc[3]
        sen_1 = sen[0]
        sen_2 = sen[1]
        sen_3 = sen[2]
        sen_4 = sen[3]
        spe_1 = spe[0]
        spe_2 = spe[1]
        spe_3 = spe[2]
        spe_4 = spe[3]
        dice1.update(dice_1)
        dice2.update(dice_2)
        dice3.update(dice_3)
        dice4.update(dice_4)
        acc1.update(acc_1)
        acc2.update(acc_2)
        acc3.update(acc_3)
        acc4.update(acc_4)
        sen1.update(sen_1)
        sen2.update(sen_2)
        sen3.update(sen_3)
        sen4.update(sen_4)
        spe1.update(spe_1)
        spe2.update(spe_2)
        spe3.update(spe_3)
        spe4.update(spe_4)
        iou1.update(iou_1)
        iou2.update(iou_2)
        iou3.update(iou_3)
        iou4.update(iou_4)
        tq.set_postfix(loss='%.4f'% (losses.avg))
    tq.close()
    return losses.avg, acc1.avg, acc2.avg, acc3.avg, acc4.avg,\
           iou1.avg, iou2.avg, iou3.avg, iou4.avg,\
           dice1.avg, dice2.avg, dice3.avg, dice4.avg,\
           sen1.avg, sen2.avg, sen3.avg, sen4.avg,\
           spe1.avg, spe2.avg,spe3.avg, spe4.avg


def train(args, model,train_loader,val_loader,writer,wd=0.0001, momentum=0.9):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    #optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=1e-8)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=momentum,weight_decay=wd)

    best_epoch = 0
    best_dice=0

    if args.lr_mode=='decrease':
        decreasing_lr = list(map(int,args.dlr.split(',')))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr,
                                                     gamma=0.8)
        lr=optimizer.param_groups[0]['lr']

    for epoch in range(args.nepoch):
        if args.lr_mode=='poly' or "cosh":
            lr = poly_learning_rate(args,optimizer,epoch)

        losses, acc1, acc2, acc3, acc4, iou1, iou2, iou3, iou4,\
        dice1, dice2, dice3, dice4, sen1, sen2, sen3, sen4,\
        spe1, spe2, spe3, spe4 = train_epoch(
            args = args,
            loader = train_loader,
            model = model,
            optimizer = optimizer,
            epoch = epoch,
            lr = lr,
            )


        writer.add_scalar('train/losses', losses, epoch)
        writer.add_scalar('train/acc1', acc1, epoch)
        writer.add_scalar('train/acc2', acc2, epoch)
        writer.add_scalar('train/acc3', acc3, epoch)
        writer.add_scalar('train/acc4', acc4, epoch)
        writer.add_scalar('train/iou1', iou1, epoch)
        writer.add_scalar('train/iou2', iou2, epoch)
        writer.add_scalar('train/iou3', iou3, epoch)
        writer.add_scalar('train/iou4', iou4, epoch)
        writer.add_scalar('train/sen1', sen1, epoch)
        writer.add_scalar('train/sen2', sen2, epoch)
        writer.add_scalar('train/sen3', sen3, epoch)
        writer.add_scalar('train/sen4', sen4, epoch)
        writer.add_scalar('train/spe1', spe1, epoch)
        writer.add_scalar('train/spe2', spe2, epoch)
        writer.add_scalar('train/spe3', spe3, epoch)
        writer.add_scalar('train/spe4', spe4, epoch)
        writer.add_scalar('train/dice1', dice1, epoch)
        writer.add_scalar('train/dice2', dice2, epoch)
        writer.add_scalar('train/dice3', dice3, epoch)
        writer.add_scalar('train/dice4', dice4, epoch)
        print('best_loss' + str(losses))

        test_losses, test_acc1, test_acc2, test_acc3, test_acc4,\
        test_iou1, test_iou2, test_iou3, test_iou4,\
        test_dice1, test_dice2, test_dice3, test_dice4,\
        test_sen1, test_sen2, test_sen3, test_sen4,\
        test_spe1, test_spe2, test_spe3, test_spe4 = test_nosave(args, val_loader, model)

        writer.add_scalar('test/losses', test_losses, epoch)
        writer.add_scalar('test/acc1', test_acc1, epoch)
        writer.add_scalar('test/acc2', test_acc2, epoch)
        writer.add_scalar('test/acc3', test_acc3, epoch)
        writer.add_scalar('test/acc4', test_acc4, epoch)
        writer.add_scalar('test/iou1', test_iou1, epoch)
        writer.add_scalar('test/iou2', test_iou2, epoch)
        writer.add_scalar('test/iou3', test_iou3, epoch)
        writer.add_scalar('test/iou4', test_iou4, epoch)
        writer.add_scalar('test/sen1', test_sen1, epoch)
        writer.add_scalar('test/sen2', test_sen2, epoch)
        writer.add_scalar('test/sen3', test_sen3, epoch)
        writer.add_scalar('test/sen4', test_sen4, epoch)
        writer.add_scalar('test/spe1', test_spe1, epoch)
        writer.add_scalar('test/spe2', test_spe2, epoch)
        writer.add_scalar('test/spe3', test_spe3, epoch)
        writer.add_scalar('test/spe4', test_spe4, epoch)
        writer.add_scalar('test/dice1', test_dice1, epoch)
        writer.add_scalar('test/dice2', test_dice2, epoch)
        writer.add_scalar('test/dice3', test_dice3, epoch)
        writer.add_scalar('test/dice4', test_dice4, epoch)
        iou_avg = (test_iou1 + test_iou2 + test_iou3 + test_iou4)/4
        dice_avg = (test_dice1 + test_dice2 + test_dice3 + test_dice4) /4
        sen_avg = (test_sen1 + test_sen2 + test_sen3 + test_sen4)/4
        spe_avg= (test_spe1 + test_spe2 + test_spe3 + test_spe4)/4
        print('dice_avg:%f' %dice_avg)
        if dice_avg > best_dice:
            best_dice = dice_avg
            best_epoch = epoch
            checkpoint_dir = args.dataset_path + "/data"+'/checkpoint'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_latest = os.path.join(checkpoint_dir,args.model + '_checkpoint_best.pth.tar')
            save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),}, filename=checkpoint_latest)
        if (epoch + 1) == args.nepoch:
            checkpoint_dir = args.dataset_path + "/data" + '/checkpoint'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_latest = os.path.join(checkpoint_dir, args.model + '_checkpoint_final.pth.tar')
            save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),},filename = checkpoint_latest)
        print('best_dice_avg:%f' %best_dice)
        print('best_epoch:%f' %best_epoch)
    return best_dice

def test(args, test_loader, model):
    print("start test")
    losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()
    acc5 = AverageMeter()
    iou1 = AverageMeter()
    iou2 = AverageMeter()
    iou3 = AverageMeter()
    iou4 = AverageMeter()
    iou5 = AverageMeter()
    dice1 = AverageMeter()
    dice2 = AverageMeter()
    dice3 = AverageMeter()
    dice4 = AverageMeter()
    dice5 = AverageMeter()
    sen1 = AverageMeter()
    sen2 = AverageMeter()
    sen3 = AverageMeter()
    sen4 = AverageMeter()
    sen5 = AverageMeter()
    spe1 = AverageMeter()
    spe2 = AverageMeter()
    spe3 = AverageMeter()
    spe4 = AverageMeter()
    spe5 = AverageMeter()

    class_acc = AverageMeter()
    tar = tqdm.tqdm(total=len(test_loader))#设置进度条
    with torch.no_grad():
        model.eval()
        for i, (data, target) in enumerate(test_loader):
            tar.update(1)
            data = data.cuda()
            target = target.cuda()
            targetedge = target


            if args.loss == 'diceloss':
                computeloss = diceloss()
            elif args.loss == 'diceloss_classloss':
                computeloss = diceloss_classloss()

            elif args.loss == 'dice_bceloss':
                computeloss = Dice_CrossEntropy_loss()
            elif args.loss == "hybrid_loss":
                computeloss = hybrid_loss()
            elif args.loss == "similarity_loss":
                computeloss = similarity_loss()
            elif args.loss == "ce_loss":
                computeloss = ce_loss()

            hroi_trans = torch.zeros([1, 5, 512, 512])
            vert_trans = torch.zeros([1, 5, 256, 256])

            if args.loss == "connect_loss":
                computeloss1 = connect_loss(args, hori_translation=hroi_trans, verti_translation=vert_trans)
                computeloss2 = connect_loss(args, hori_translation=hroi_trans, verti_translation=vert_trans)

            #output1, output2 = model(data)
            #loss1 = computeloss1.forward(output1, target.long())
            #loss2 = computeloss2.forward(output2, target.long())
            #loss = loss1 + 0.3 * loss2
            output = model(data)
            loss = computeloss.forward(output, target.long())
            pred_seg = torch.argmax(F.softmax(output.detach(), 1), dim=1).int()

            batch_size = target.size(0)
            losses.update(loss.data, batch_size)

            dice, pre, jacc, sen, spe, classacc = eval_multi_seg_haveRS_classacc(pred_seg, target.long())
            dice_1 = dice[0]
            dice_2 = dice[1]
            dice_3 = dice[2]
            dice_4 = dice[3]
            dice_5 = dice[4]
            acc_1 = pre[0]
            acc_2 = pre[1]
            acc_3 = pre[2]
            acc_4 = pre[3]
            acc_5 = pre[4]
            iou_1 = jacc[0]
            iou_2 = jacc[1]
            iou_3 = jacc[2]
            iou_4 = jacc[3]
            iou_5 = jacc[4]
            sen_1 = sen[0]
            sen_2 = sen[1]
            sen_3 = sen[2]
            sen_4 = sen[3]
            sen_5 = sen[4]
            spe_1 = spe[0]
            spe_2 = spe[1]
            spe_3 = spe[2]
            spe_4 = spe[3]
            spe_5 = spe[4]
            dice1.update(dice_1)
            dice2.update(dice_2)
            dice3.update(dice_3)
            dice4.update(dice_4)
            dice5.update(dice_5)
            acc1.update(acc_1)
            acc2.update(acc_2)
            acc3.update(acc_3)
            acc4.update(acc_4)
            acc5.update(acc_5)
            sen1.update(sen_1)
            sen2.update(sen_2)
            sen3.update(sen_3)
            sen4.update(sen_4)
            sen5.update(sen_5)
            spe1.update(spe_1)
            spe2.update(spe_2)
            spe3.update(spe_3)
            spe4.update(spe_4)
            spe5.update(spe_5)
            iou1.update(iou_1)
            iou2.update(iou_2)
            iou3.update(iou_3)
            iou4.update(iou_4)
            iou5.update(iou_5)
            class_acc.update(classacc)

            save_predict(data, pred_seg, targetedge, i, save_path = args.dataset_path,
                         batch_size_test = args.batch_size_test, model = args.model)
            tar.set_postfix(loss = '%.4f'% (losses.avg))
        tar.close()

    return losses.avg, acc1.avg, acc2.avg, acc3.avg, acc4.avg, acc5.avg,\
            iou1.avg, iou2.avg, iou3.avg, iou4.avg, iou5.avg,\
            dice1.avg, dice2.avg, dice3.avg, dice4.avg, dice5.avg,\
            sen1.avg, sen2.avg, sen3.avg, sen4.avg, sen5.avg,\
            spe1.avg, spe2.avg, spe3.avg, spe4.avg, spe5.avg, classacc

def test_nosave(args,val_loader,model):
    print("start test")
    losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()
    iou1 = AverageMeter()
    iou2 = AverageMeter()
    iou3 = AverageMeter()
    iou4 = AverageMeter()
    dice1 = AverageMeter()
    dice2 = AverageMeter()
    dice3 = AverageMeter()
    dice4 = AverageMeter()
    sen1 = AverageMeter()
    sen2 = AverageMeter()
    sen3 = AverageMeter()
    sen4 = AverageMeter()
    spe1 = AverageMeter()
    spe2 = AverageMeter()
    spe3 = AverageMeter()
    spe4 = AverageMeter()

    tar = tqdm.tqdm(total=len(val_loader))#设置进度条
    with torch.no_grad():
        model.eval()
        for i, (data, target) in enumerate(val_loader):
            tar.update(1)
            data = data.cuda()
            target = target.cuda()

            hroi_trans = torch.zeros([1, 5, 512, 512])
            vert_trans = torch.zeros([1, 5, 256, 256])

            if args.loss == 'diceloss':
                computeloss = diceloss()
            elif args.loss == 'diceloss_classloss':
                computeloss = diceloss_classloss()

            elif args.loss == 'dice_bceloss':
                computeloss = Dice_CrossEntropy_loss()
            elif args.loss == "hybrid_loss":
                computeloss = hybrid_loss()
            elif args.loss == "similarity_loss":
                computeloss = similarity_loss()
            elif args.loss == "ce_loss":
                computeloss = ce_loss()

            elif args.loss == "connect_loss":
                computeloss1 = connect_loss(args, hori_translation=hroi_trans, verti_translation=vert_trans)
                computeloss2 = connect_loss(args, hori_translation=hroi_trans, verti_translation=vert_trans)

            #output1, output2 = model(data)
            #loss1 = computeloss1.forward(output1, target.long())
            #loss2 = computeloss2.forward(output2, target.long())
            #loss = loss1 + 0.3 * loss2
            #loss = computeloss1.forward(output1, target.long())
            output = model(data)
            loss = computeloss.forward(output, target.long())
            pred_seg = torch.argmax(F.softmax(output.detach(),1),dim=1).int()

            batch_size = target.size(0)
            losses.update(loss.data, batch_size)

            dice, pre, jacc, sen, spe = eval_multi_seg(pred_seg, target.long())
            dice_1=dice[0]
            dice_2=dice[1]
            dice_3=dice[2]
            dice_4=dice[3]
            acc_1=pre[0]
            acc_2=pre[1]
            acc_3=pre[2]
            acc_4=pre[3]
            iou_1=jacc[0]
            iou_2=jacc[1]
            iou_3=jacc[2]
            iou_4=jacc[3]
            sen_1=sen[0]
            sen_2=sen[1]
            sen_3=sen[2]
            sen_4=sen[3]
            spe_1=spe[0]
            spe_2=spe[1]
            spe_3=spe[2]
            spe_4=spe[3]
            dice1.update(dice_1)
            dice2.update(dice_2)
            dice3.update(dice_3)
            dice4.update(dice_4)
            acc1.update(acc_1)
            acc2.update(acc_2)
            acc3.update(acc_3)
            acc4.update(acc_4)
            sen1.update(sen_1)
            sen2.update(sen_2)
            sen3.update(sen_3)
            sen4.update(sen_4)
            spe1.update(spe_1)
            spe2.update(spe_2)
            spe3.update(spe_3)
            spe4.update(spe_4)
            iou1.update(iou_1)
            iou2.update(iou_2)
            iou3.update(iou_3)
            iou4.update(iou_4)
            tar.set_postfix(loss='%.4f'% (losses.avg))
        tar.close()
    return losses.avg, acc1.avg, acc2.avg, acc3.avg, acc4.avg, \
           iou1.avg, iou2.avg, iou3.avg, iou4.avg, \
           dice1.avg, dice2.avg, dice3.avg, dice4.avg, \
           sen1.avg, sen2.avg, sen3.avg, sen4.avg, \
           spe1.avg, spe2.avg, spe3.avg, spe4.avg




