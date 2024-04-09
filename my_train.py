import argparse
from train import train
from train import test, test_nosave
from loss.utils import get_data
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from tensorboardX import SummaryWriter
from model.pspnet import PSPNet
from model.cenet import CE_Net_
from model.att_unet import AttU_Net
from model.deeplab import deeplabv3
from model.rensnet18_deepsuper import ResNet18_deepsuper
from model.cpfnet import CPFNet
from model.unet import UNet
from model.resnet18_class import ResNet18_classseg, ResNet18_twotask_classseg,ResNet18_twotask_12segclass,ResNet18_twotask_2segencoder
from model.resnet18_oneseg_fiveseg import ResNet18_fiveseg
from model.resnet18fsim import ResNet18_FSim, ResNet18_FSim_FAM,ResNet18_FSim_ETA
from model.resnet18_MS import ResNet18_ag3d
from model.resnet18FAM import ResNet18_FAM,ResNet18_FAM_eta
from model.resnet18 import ResNet18, ResNet34
from model.resnet18_2ga import ResNet18_ETA,ResNet18_CHWA,ResNet18_W3AGpluschuan
from model.Unetpp import Unetpp
from model.r2unet import R2U_Net
from model.segnet import SegNet
from model.resnet18_d2d4 import ResNet18_d2d4,ResNet18d2d4_twotask_12segclass
from model.seg_hrnet import HighResolutionNet
from model.ResTv1 import rest_lite, rest_small, rest_base, rest_large
from model.ResTv2 import restv2_tiny, restv2_small, restv2_base, restv2_large
from model.SwinU import SwinTransformerSys
from model.SwinT import SwinTransformer
from model.SegFormer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from model.PVTv2 import pvt_tiny, pvt_small, pvt_medium, pvt_large
from model.ShuntedT import shunted_t, shunted_s,shunted_b
from model.Mine import mine
from model.mine2 import mine2
from model.mine3 import mine3
from model.mine4 import mine4
from model.mine5 import mine5
from model.mine6 import mine6
from model.mine7 import mine7
from model.SemiVIT import Semiformer
from model.res_unetplus import build_resunetplusplus
from model.multi_resunet import MultiResUnet
from model.CSnet import CSNet
from model.MsTGANet import MsTGANet
from model2.swinu_pro import SwinTransformerSys_pro
from model2.SETR import SETRModel
from model.transattunet import UNet_Attention_Transformer_Multiscale
from model.UTNet import UTNet
from model.transunet import TransUNet
from model.H2former import res34_swin_MS

from model2.DconnNet import DconnNet
def init(args, mode='train'):
       
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('{}:{}'.format('cuda', torch.cuda.is_available()))
    args.cuda = torch.cuda.is_available() #查看cuda是否正常

    #----------------------CNN----------------------

    if args.model == 'resnet18_deepsuper_r':
        model = ResNet18_deepsuper()
    elif args.model == 'resnet18_W3AGpluschuan_noabc':
        model = ResNet18_W3AGpluschuan()
    elif args.model == 'unet':
        model = UNet()

    elif args.model == 'resnet18_twotask_2segencoder2':
        model = ResNet18_twotask_2segencoder()
    elif args.model == 'resnet18_twotask_classseg_crosscon':
        model = ResNet18_twotask_classseg()
    elif args.model == 'cmcnet':
        #所有模块加起来的
        model = ResNet18_twotask_12segclass()
    elif args.model == 'resnet18_oneseg_fiveseg_2cat_g':
        model = ResNet18_fiveseg()
    elif args.model == 'pspnet':
        model = PSPNet()
    elif args.model == 'cenet1':
        model = CE_Net_()
    elif args.model == 'r2unet':
        model = R2U_Net()
    elif args.model == 'segnet':
        model = SegNet()
    elif args.model == 'deeplabv3':
        model = deeplabv3()
    elif args.model == 'unetpp_nodeep6':
        model = Unetpp()
    elif args.model == 'attu':
        model = AttU_Net()
    elif args.model == 'resnet18upcat_twonet_class_seg':
        model = ResNet18_classseg()
    elif args.model == 'resnet18_ag3d':
        model = ResNet18_ag3d()
    elif args.model == 'resnet18':
        model = ResNet18()
    elif args.model == 'resnet34':
        model = ResNet34()

    elif args.model == 'resnet18_edgetargetat':
        model = ResNet18_ETA()
    elif args.model == 'resnet18_w3agjia':
        model = ResNet18_CHWA()
    elif args.model == 'cpfnet':
        model = CPFNet()
    elif args.model == 'resnet18_fsim1' or args.model=='resnet18_fsimdif_center4deep_CHWAgainosig' or args.model=='resnet18_fsim_center4deep_0.1GAN':
        model = ResNet18_FSim()
    elif args.model == 'resnet18_fsim_FAM_r1':
        model = ResNet18_FSim_FAM()
    elif args.model == 'resnet18_fsimdif_center4deep_etagai22':
        model = ResNet18_FSim_ETA()
    elif args.model == 'resnet18_fam':
        model = ResNet18_FAM()
    elif args.model == 'resnet18_fam_eta':
        model = ResNet18_FAM_eta()
    elif args.model == 'ResNet18_d2d4':
        model = ResNet18_d2d4()
    elif args.model == 'resnet18d2d4_twotask_12classseg_lowjiaclasscrosscon_w3agplus+dicelossclassloss0.1':
        model = ResNet18d2d4_twotask_12segclass()
        
    elif args.model == 'hrnet':
        model = HighResolutionNet()
    elif args.model == "res_unet++":
        model = build_resunetplusplus()
    elif args.model == "multi_resunet":
        model = MultiResUnet()
    elif args.model == "csnet":
        model = CSNet()
    #----------------------Transformer Model----------------------

    # ResTransformer v1
    elif args.model == "restv1_lite":
        model = rest_lite()
    elif args.model == "restv1_small":
        model = rest_small()
    elif args.model == "restv1_base":
        model = rest_base()
    elif args.model == "restv1_large":
        model = rest_large()

    # ResTransformer v2
    elif args.model == "restv2_tiny":
        model = restv2_tiny()
    elif args.model == "restv2_small":
        model = restv2_small()
    elif args.model == "restv2_base":
        model = restv2_base()
    elif args.model == "restv2_large":
        model = restv2_large()


    #SegFormer
    elif args.model == "segformer0":
        model = mit_b0()
    elif args.model == "segformer1":
        model = mit_b1()
    elif args.model == "segformer2":
        model = mit_b2()
    elif args.model == "segformer3":
        model = mit_b3()
    elif args.model == "segformer4":
        model = mit_b4()
    elif args.model == "segformer5":
        model = mit_b5()

    #Pyramid Vision Transformer v2
    elif args.model == "pvt_tiny":
        model = pvt_tiny()
    elif args.model == "pvt_small":
        model = pvt_small()
    elif args.model == "pvt_medium":
        model = pvt_medium()
    elif args.model == "pvt_large":
        model = pvt_large()

    elif args.model == "swint":
        model = SwinTransformer()
    elif args.model == "swinu":
        model = SwinTransformerSys()
    elif args.model == "swinu_pro":
        model = SwinTransformerSys_pro()

    #Shunted Transformer
    elif args.model == "shunted_t":
        model = shunted_t()
    elif args.model == "shunted_s":
        model = shunted_s()
    elif args.model == "shunted_b":
        model = shunted_b()


    #elif args.model == "semiformer":
    #    model = Semiformer()
    #没做分割
    elif args.model == "SETR":
        model = SETRModel()


    #Mine
    elif args.model == "mine":
        model = mine()
    elif args.model == "mine2":
        model = mine2()
    elif args.model == "mine3":
        model = mine3()
    elif args.model == "mine4":
        model = mine4()
    elif args.model == "mine5":
        model = mine5()
    elif args.model == "mine6":
        model = mine6()
    elif args.model == "mine7":
        model = mine7()
    elif args.model == "MsTGANet":
        model = MsTGANet()

    elif args.model == "DconnNet":
        model = DconnNet()
    elif args.model == "transattunet":
        model = UNet_Attention_Transformer_Multiscale()
    elif args.model == "utnet":
        model = UTNet()
    elif args.model == "transunet":
        model = TransUNet()
    elif args.model == "h2former":
        model = res34_swin_MS()

        
    model.cuda()
    return model

#C:/Users/lenovo/Desktop/fiveseg
#/data1/ycq/fiveseg

if __name__ == '__main__':
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description="jiang's train")
    parser.add_argument('--batch_size_train', type=int, default=4, help='input batch size for training (default: 4)')
    parser.add_argument('--batch_size_test', type=int, default=4, help='input batch size for testing (default: 4)')
    parser.add_argument('--lr', type=float, default=1.5e-2, help='learning rate (default: 0.01)')#学习率以前是0.01记得改回来
    parser.add_argument('--gpu', default='12', help='index of gpus to use')
    parser.add_argument('--dlr', default='30,70', help='decrease')
    parser.add_argument('--lr_mode', default='cosh', help='decrease or poly')
    parser.add_argument('--model', default='mine3', help='which model (default: resnet18)')
    #parser.add_argument('--model_twoclass', default='resnet34ga_twoclass', help='which model (default: resnet18)')
    #parser.add_argument('--model_oneseg', default='resnet18_oneseg', help='which model (default: resnet18)')
    #parser.add_argument('--model_d', default='Dresnet34', help='which model (default: resnet18)')
    parser.add_argument('--dataset_path', default=r'E:\MY_DATASET\SEGMENTATION', help='dataset_path (default:./)')
    parser.add_argument('--nepoch', type=int, default=1, help='epochs (default: 100)')
    parser.add_argument('--seed', type=int, default='69', help='seed (default: 1)')
    parser.add_argument('--pretrain', type=int, default='1', help='pretrain (default: 1)')
    parser.add_argument('--mode', default='train', help='data_name (default: train)')   
    parser.add_argument('--loss', default='diceloss', help='loss (default: diceloss or diceloss_classloss or dice_edgeloss or dice_unevenloss)')
    args = parser.parse_args()
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.set_num_threads(1) 
    

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_loader = get_data(args.batch_size_test, mode="train", dataset_path=args.dataset_path,
                          mask=False, scale=(256, 512), canny=False, stripup=False)
    val_loader = get_data(args.batch_size_test, mode="val", dataset_path=args.dataset_path,
                        mask=False, scale=(256, 512), canny=False, stripup=False)
    test_loader = get_data(args.batch_size_test, mode="test", dataset_path=args.dataset_path,
                         mask=False, scale=(256, 512), canny=False, stripup=False)


    model = init(args)
    writer = SummaryWriter(comment = (args.model))

    val_bestavgdice = train(args, model, train_loader, val_loader, writer)
    pretrained_model_path = args.dataset_path + "/data" + '/checkpoint'

    #测试验证集最好的模型的结果
    pretrained_model_path = os.path.join(pretrained_model_path, args.model + '_checkpoint_best.pth.tar')
    #pretrained_model_path = r"C:\Users\PC2022\Desktop\jiang\duibi\transformer\swinu\swinu_checkpoint_best.pth.tar"

    print("=> loading pretrained model '{}'".format(pretrained_model_path))
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    test_losses, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5,\
    test_iou1, test_iou2, test_iou3, test_iou4, test_iou5,\
    test_dice1, test_dice2, test_dice3, test_dice4, test_dice5,\
    test_sen1, test_sen2, test_sen3, test_sen4, test_sen5,\
    test_spe1, test_spe2, test_spe3, test_spe4, test_spe5, test_classacc = test(args, test_loader, model)

    iouavg = ((test_iou1 + test_iou2 + test_iou3 + test_iou4) / 4.0)
    diceavg = ((test_dice1 + test_dice2 + test_dice3 + test_dice4) / 4.0)
    senavg = ((test_sen1 + test_sen2 + test_sen3 + test_sen4) / 4.0)
    speavg = ((test_spe1 + test_spe2 + test_spe3 + test_spe4) / 4.0)
    accavg = ((test_acc1 + test_acc2 + test_acc3 + test_acc4) / 4.0)
    

    with open(args.dataset_path + "/data" + "/text" + "/" + args.model + '_classandrs' + 'valbest.txt', 'w') as f:
        
        f.write('  dice1:%f' %(test_dice1 * 100))
        f.write('  iou1:%f' %(test_iou1 * 100))
        f.write('  sen1:%f' %(test_sen1 * 100))
        f.write('  spe1:%f' %(test_spe1 * 100) + "\n")
        
        f.write('  dice2:%f' %(test_dice2 * 100))
        f.write('  iou2:%f' %(test_iou2 * 100))
        f.write('  sen2:%f' %(test_sen2 * 100))
        f.write('  spe2:%f' %(test_spe2 * 100) + "\n")
        
        f.write('  dice3:%f' %(test_dice3 * 100))
        f.write('  iou3:%f' %(test_iou3 * 100))
        f.write('  sen3:%f' %(test_sen3 * 100))
        f.write('  spe3:%f' %(test_spe3 * 100) + "\n")

        f.write('  dice4:%f' %(test_dice4 * 100))
        f.write('  iou4:%f' %(test_iou4 * 100))
        f.write('  sen4:%f' %(test_sen4 * 100))
        f.write('  spe4:%f' %(test_spe4 * 100) + "\n")
    
        f.write('  dice_avg:%f' %(diceavg * 100))
        f.write('  iou_avg:%f' %(iouavg * 100))
        f.write('  sen_avg:%f' %(senavg * 100))
        f.write('  spe_avg:%f' %(speavg * 100) + "\n")
        
        f.write("\n")
        f.write('  dicers:%f' %(test_dice5 * 100))
        f.write('  iours:%f' %(test_iou5 * 100))
        f.write('  senrs:%f' %(test_sen5 * 100))
        f.write('  spers:%f' %(test_spe5 * 100) + "\n")
        f.write('  class_acc:%f' %(test_classacc * 100) + "\n")
       
        #f.write('  val_bestavgdice:%f' %(val_bestavgdice * 100))
        
        
    #测试最后保存的模型结果

    pretrained_model_path = args.dataset_path + "/data" + '/checkpoint'
    pretrained_model_path = os.path.join(pretrained_model_path, args.model + '_checkpoint_final.pth.tar')
    #pretrained_model_path = r"C:\Users\PC2022\Desktop\jiang\duibi\hybrid\mine\mine3_checkpoint_final.pth77.3.tar"
    print("=> loading pretrained model '{}'".format(pretrained_model_path))
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    test_losses, test_acc1, test_acc2, test_acc3, test_acc4,\
    test_iou1, test_iou2, test_iou3, test_iou4,\
    test_dice1, test_dice2, test_dice3, test_dice4,\
    test_sen1, test_sen2, test_sen3, test_sen4,\
    test_spe1, test_spe2, test_spe3, test_spe4 = test_nosave(args, test_loader, model)
             
    iouavg = ((test_iou1 + test_iou2 + test_iou3 + test_iou4) / 4.0)
    diceavg = ((test_dice1 + test_dice2 + test_dice3 + test_dice4) / 4.0)
    senavg = ((test_sen1 + test_sen2 + test_sen3 + test_sen4) / 4.0)
    speavg = ((test_spe1 + test_spe2 + test_spe3 + test_spe4) / 4.0)


    with open(args.dataset_path + "/data" + "/text" + "/" + args.model+'_' + 'valfinal.txt', 'w') as f:
        
        f.write('  dice1:%f' %(test_dice1 * 100))
        f.write('  iou1:%f' %(test_iou1 * 100))
        f.write('  sen1:%f' %(test_sen1 * 100))
        f.write('  spe1:%f' %(test_spe1 * 100) + "\n")
        
        f.write('  dice2:%f' %(test_dice2 * 100))
        f.write('  iou2:%f' %(test_iou2 * 100))
        f.write('  sen2:%f' %(test_sen2 * 100))
        f.write('  spe2:%f' %(test_spe2 * 100) + "\n")
        
        f.write('  dice3:%f' %(test_dice3 * 100))
        f.write('  iou3:%f' %(test_iou3 * 100))
        f.write('  sen3:%f' %(test_sen3 * 100))
        f.write('  spe3:%f' %(test_spe3 * 100) + "\n")
        
        f.write('  dice4:%f' %(test_dice4 * 100))
        f.write('  iou4:%f' %(test_iou4 * 100))
        f.write('  sen4:%f' %(test_sen4 * 100))
        f.write('  spe4:%f' %(test_spe4 * 100) + "\n")
    
        f.write('  dice_avg:%f' %(diceavg * 100))
        f.write('  iou_avg:%f' %(iouavg * 100))
        f.write('  sen_avg:%f' %(senavg * 100))
        f.write('  spe_avg:%f' %(speavg * 100) + "\n")
        
        f.write("\n")
        #f.write('  val_bestavgdice:%f' %(val_bestavgdice * 100))

    print('hhh')
    print('Done!')

