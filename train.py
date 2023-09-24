import warnings
import datetime
import torch
import os
import numpy as np
from dataloader import Dataset_train, Dataset_val, Dataset_test

from torch.optim import Adam
from torch.utils.data import DataLoader
from network import Network
from utils import SWL
from medpy import metric
import random


# 定义超参数
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_GPU else "cpu")


def seed_torch(seed=828):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, fold, filename, SAVE=False):
    if SAVE:
        torch.save(state, os.path.join(fold, filename))
        print('save model in specific epoch ', filename)


def dice_coef(pred_label, gt_label):
    # list of classes
    c_list = [0,1]

    dice_c = []
    for c in range(1,len(c_list)): # dice not for bg
        # intersection
        ints = np.sum(((pred_label == c_list[c]) * 1) * ((gt_label == c_list[c]) * 1))
        # sum
        sums = np.sum(((pred_label == c_list[c]) * 1) + ((gt_label == c_list[c]) * 1))+1e-5
        dice_c.append((2.0 * ints +1e-5) / sums)

    return dice_c


def train(param_set, model):
    folder = param_set['model']+datetime.datetime.now().strftime('%Y-%m-%d-%H')
    save_dir = param_set['result_dir'] + folder
    ckpt_dir = save_dir + '/checkpoint'
    log_dir = save_dir + '/log'
    test_result_dir = save_dir + '/testResult'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.mkdir(ckpt_dir)
        os.mkdir(test_result_dir)
    for file in os.listdir(log_dir):
        print('removing ' + os.path.join(log_dir, file))
        os.remove(os.path.join(log_dir, file))

    criterion = SWL()
    optimizer = Adam(model.parameters(),lr = 5e-4)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400])

    epoch_save = param_set['epoch_save']
    num_epochs = param_set['epoch']

    iter_count = 0
    train_loader = DataLoader(Dataset_train(param_set['traindir']),num_workers=param_set['num_workers'], batch_size=param_set['batch_size'], shuffle=True, pin_memory=True, drop_last=True)
    print('steps per epoch:', len(train_loader))
    val_loader = DataLoader(Dataset_test(param_set['valdir']), num_workers=2, batch_size=1, shuffle=False, drop_last=True)
    test_loader = DataLoader(Dataset_test(param_set['testdir']), num_workers=2, batch_size=1, shuffle=False, drop_last=True)
    #####train#####
    model.train()
    best_valDPC_result = 0
    best_testDPC_result = 0
    for epoch in range(num_epochs+1):
        iter_loss = 0
        iter_dice = 0
        lr_decay.step()
        for step, (Pre_image, AP_image, VP_image, DP_image, _, labels) in enumerate(train_loader):
            model.train()
            if USE_GPU:
                Pre_image = Pre_image.cuda(non_blocking=True)
                AP_image = AP_image.cuda(non_blocking=True)
                VP_image = VP_image.cuda(non_blocking=True)
                DP_image = DP_image.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            outputs, seg5, seg4, seg3, seg2 = model(DP_image, VP_image, AP_image, Pre_image)

            tumor_map = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            out_dice = dice_coef(tumor_map, labels.cpu().detach().numpy())
            iter_dice += out_dice[0]

            loss = criterion(outputs, labels, seg5, seg4, seg3, seg2)
            iter_loss += float(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_count += 1

        print("epoch {epoch}, training dice {train_dice}, training loss {train_loss}".format(epoch=epoch, train_loss=iter_loss / len(train_loader), train_dice=iter_dice / len(train_loader)))  #for debug

        ####eval####
        valDG, valDPC, val_result = eval_predict(val_loader, model, epoch, log_dir)
        print("epoch {epoch}, val DG {val_DG:6.3f}, val DPC {val_DPC:6.3f}".format(epoch=epoch, val_DG=valDG*100, val_DPC=valDPC*100))
        
        ####save checkpoint###
        filename = "{model}-{epoch:03}-{val_DPC:6.3f}.pth".format(model=param_set['model'], epoch=epoch, val_DPC=valDPC*100)
        save_checkpoint(model.state_dict(),
                            ckpt_dir,
                            filename,
                            SAVE=(epoch % epoch_save == 0 or epoch==num_epochs))
        

def eval_predict(val_loader, model, epoch, log_dir):
    model.cuda()
    model.eval()
    with torch.no_grad():
        DG = 0
        DPC = 0
        dice_score = []
        dice_intersection, dice_union = 0, 0

        for step, (Pre_image, AP_image, VP_image, DP_image, _, labels) in enumerate(val_loader):
            if USE_GPU:
                Pre_image = Pre_image.squeeze()
                AP_image = AP_image.squeeze()
                VP_image = VP_image.squeeze()
                DP_image = DP_image.squeeze()
                labels = labels.squeeze()

            # 计算并保存每个CT的patch的坐标
            pred_seg_array = np.zeros(DP_image.shape)
            for z in range(DP_image.shape[0]):
                patch_array1 = Pre_image[z, :,:]
                patch_array2 = AP_image[z, :,:]
                patch_array3 = VP_image[z, :,:]
                patch_array4 = DP_image[z, :,:]

                with torch.no_grad():
                    patch_tensor1 = torch.FloatTensor(patch_array1).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
                    patch_tensor2 = torch.FloatTensor(patch_array2).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
                    patch_tensor3 = torch.FloatTensor(patch_array3).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
                    patch_tensor4 = torch.FloatTensor(patch_array4).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
                    patch_outputs,_,_,_,_ = model(patch_tensor4,patch_tensor3,patch_tensor2,patch_tensor1)
                    patch_outputs = patch_outputs.squeeze()
                    patch_pred_seg = patch_outputs.cpu().detach().numpy()
                    patch_pred_seg = np.argmax(patch_pred_seg, axis=0)
                
                pred_seg_array[z, :, :] = patch_pred_seg

            # 计算分割评价指标
            labels = labels.detach().numpy()
            current_dice = metric.binary.dc(pred_seg_array, labels)
            dice_score.append(current_dice)
            
            dice_intersection += 2 * (labels * pred_seg_array).sum()
            dice_union += labels.sum() + pred_seg_array.sum()

        # 将评价指标写入到exel中
        DG = dice_intersection / dice_union
        DPC = np.mean(dice_score)
        
        return DG, DPC, dice_score


def main(param_set):
    seed_torch(seed=828)
    warnings.filterwarnings('ignore')
    print('====== Phase >>> %s <<< ======' % param_set['mode'])
    NUM_CLASSES = param_set['nclass']
    model = Network(NUM_CLASSES)
    model = torch.nn.DataParallel(model)
    model.to(DEVICE)

    pre_train = param_set['pre_train']
    if param_set['mode']=='train':
        if pre_train:
            ckpt_dir = param_set['result_dir'] + param_set['folder'] + '/checkpoint/' + param_set['model_loc']
            model.load_state_dict(torch.load(ckpt_dir))
            model_name = ckpt_dir.split('/')[-1]
            print('load pretrained model ', model_name)
        train(param_set, model)

if __name__ == '__main__':
    param_set = dict(numChannels=1,
                      mode='train', # 'train' or 'test'
                      height=256,     #input resolution height
                      width=352,      #input resolution width
                      nclass=2,     #class number
                      batch_size=4,
                      num_workers=4,
                      result_dir='./Result/',  #save results to this dir
                      folder='',
                      traindir='',      #data path
                      valdir='',
                      testdir='',  #test path
                      pre_train=False,
                      model_loc='',
                      epoch=500,
                      epoch_save=50,
                      model='MCDA')
    main(param_set)
