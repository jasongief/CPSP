import os
import time
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder

from losses import video_contrastive_loss, loss_vpsa_infoNCE_version, LabelFreeSelfSupervisedNCELoss
import pdb


# configs
dataset_configs = get_and_save_args(parser)
parser.set_defaults(**dataset_configs)
args = parser.parse_args()
# select GPUs
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

 # =================================  seed config ============================
SEED = args.seed
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =============================================================================


'''Create snapshot_pred dir for copying code and saving model '''
if not os.path.exists(args.snapshot_pref):
    os.makedirs(args.snapshot_pref, exist_ok=True)

# if os.path.isfile(args.resume):
#     args.snapshot_pref = os.path.dirname(args.resume)

logger = Prepare_logger(args, eval=args.evaluate)

if not args.evaluate:
    logger.info(f'\nCreating folder: {args.snapshot_pref}')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
else:
    logger.info(f'\nLog file will be save in {args.snapshot_pref}/Eval.log.')

# '''Tensorboard and Code backup'''
# writer = SummaryWriter(args.snapshot_pref)
# recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
# recorder.writeopt(args)


"""dataset selection"""
if args.dataset_name == 'ave':
    if args.vis_fea_type == 'vgg':
        from dataset.AVE_dataset_weak import AVEDataset_VGG as AVEDataset
    elif args.vis_fea_type == 'resnet':
        from dataset.AVE_dataset_weak import AVEDataset_ResNet as AVEDataset
        print("[Warning] The path of visual feature should be given in args.")
        print("Current visual feature path is {}\n".format(args.vis_maps_path))
        print("[Warning] visual dimension in the Network should also match with the features.")
elif args.dataset_name =='vggsound':
    from dataset.vggsound_avel_dataset import VGGSoundAVELDatasetWeak as AVEDataset
else: 
    raise NotImplementedError


def main():
    '''Dataloader selection'''
    if args.dataset_name == 'ave':
        data_root = 'please change this path to the path of AVE data'
        train_dataloader = DataLoader(
            AVEDataset(data_root, args, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            AVEDataset(data_root, args, split='val'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            AVEDataset(data_root, args, split='test'),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
    elif args.dataset_name == 'vggsound':
        meta_csv_path = 'please change this to your path of the vggsound-avel100k.csv'
        audio_fea_base_path = 'please change this to your path of the audio feature'
        video_fea_base_path = 'please change this to your path of the video feature'
        avc_label_base_path = 'please change this to your path of labels'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='val'),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='test'),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
    else:
        raise NotImplementedError


    '''model setting'''
    if 'psp' in args.model:
        print(f'Load {args.model} network')
        if args.vis_fea_type == 'vgg':
            from model.psp_family import weakly_psp_net as main_model
            mainModel = main_model(vis_fea_type=args.vis_fea_type, flag=args.model, thr_val=args.threshold_value, category_num=args.category_num)
        elif args.vis_fea_type == 'resnet':
            from model.psp_family import weakly_resnet_psp_net as main_model
            mainModel = main_model(vis_fea_type=args.vis_fea_type, flag=args.model, last_layer=3, thr_val=args.threshold_value, category_num=args.category_num)
    elif args.model == 'cmran': # for CMRAN, the returns of this model should be modified
        print(f'Load CMRAN network')
        from model.cmran import weak_main_model as main_model # for CMRAN
        mainModel = main_model(vis_fea_type=args.vis_fea_type, category_num=args.category_num)
    elif args.model == 'avel':
        print(f'Load AVEL network')
        from model.avel import TBMRF_Net as main_model
        mainModel = main_model(vis_fea_type=args.vis_fea_type, flag='weakly', category_num=args.category_num)
    elif args.model == 'avsdn':
        print(f'Load AVSDN network')
        from model.avsdn import avsdn_net as main_model
        mainModel = main_model(vis_fea_type=args.vis_fea_type, flag='weakly', category_num=args.category_num) 
    
    '''optimizer and criterion''' 
    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[5, 20, 40], gamma=0.5)
    
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.MultiLabelSoftMarginLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError

    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Testing..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    best_accuracy, best_accuracy_epoch = 0, 0
    '''Training and Evaluation'''
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, val_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                logger.info(f'best accuracy at epoch-{epoch}: {best_accuracy:.4f}')
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='WeakSupervised',
                    epoch=epoch + 1,
                    seed=SEED,
                )
        scheduler.step()


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        # labels = labels.cuda()
        visual_feature = visual_feature.double()
        audio_feature = audio_feature.double()
        labels = labels.double().cuda() # [B, 29]
        if args.model == 'cmran': # should be one of ['avel', 'avsdn', 'cmran', 'psp', 'cpsp', 'sspsp']
            is_event_scores, raw_logits, event_scores = model(visual_feature, audio_feature) # for CMRAN
            # is_event_scores = is_event_scores.transpose(1, 0).squeeze().contiguous()
        elif args.model == 'sspsp':
            final_a_fea, final_v_fea, is_event_scores, raw_logits, event_scores = model(visual_feature, audio_feature)
        else:
            fusion, is_event_scores, raw_logits, event_scores = model(visual_feature, audio_feature)

        '''compute losses'''
        loss_event_class = criterion_event(event_scores, labels)
        loss = loss_event_class
        loss_vcon, loss_sscon = torch.tensor(0).cuda(), torch.tensor(0).cuda()
        if args.vcon_flag:
            _, event_class_flag = labels.max(-1) # [B]
            loss_vcon = video_contrastive_loss(fusion, event_class_flag, margin=args.margin, neg_num=args.neg_num)
            loss += args.lambda_vcon * loss_vcon
            # pdb.set_trace()
        if args.vcon_infoNCE_flag:
            _, event_class_flag = labels.max(-1) # [B]
            loss_vcon = loss_vpsa_infoNCE_version(fusion, event_class_flag, t=args.eta)
            loss += args.lambda_vcon * loss_vcon
        if args.model == 'sspsp':
            loss_sscon = LabelFreeSelfSupervisedNCELoss(final_a_fea, final_v_fea)
            loss += args.lambda_sscon * loss_sscon
    
        loss.backward()

        '''Compute Accuracy'''
        # acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        acc = torch.tensor([0])
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # '''Add loss of a iteration in Tensorboard'''
        # writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg:.3f})\t'
                f'loss_event_class {loss_event_class.item():.3f}\t'
                f'loss_sscon {loss_sscon.item():.3f}'
                # f'loss_vcon {loss_vcon.item():.3f}'
            )

        # '''Add loss of an epoch in Tensorboard'''
        # writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg



@torch.no_grad()
def validate_epoch(model, val_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()
    model.double()

    for n_iter, batch_data in enumerate(val_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        # labels = labels.cuda()
        visual_feature = visual_feature.double()
        audio_feature = audio_feature.double()
        labels = labels.double().cuda()
        bs = visual_feature.size(0)
        
        if args.model == 'cmran':
            is_event_scores, raw_logits, event_scores = model(visual_feature, audio_feature) # for CMRAN
            # is_event_scores = is_event_scores.transpose(1, 0).squeeze()
        elif args.model == 'sspsp':
            _, _, is_event_scores, raw_logits, event_scores = model(visual_feature, audio_feature)
        else:
            _, is_event_scores, raw_logits, event_scores = model(visual_feature, audio_feature)
        

        acc = compute_accuracy_supervised(is_event_scores, raw_logits, labels, bg_flag=args.category_num)
        accuracy.update(acc.item(), bs)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Print logs in Terminal'''
        if n_iter % args.val_print_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(val_dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
            )

    # if not eval_only:
    #     writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    logger.info(
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
    )

    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels, bg_flag=29):
    # is_event_scores: [B, 10]
    # event_scores: [B, 29]
    # labels: [B, 10, 29]
    # labels = labels[:, :, :-1]  # 28 denote background
    _, targets = labels.max(-1) # [B, 10]
    # pos pred
    is_event_scores = is_event_scores.sigmoid() # [B, 10]
    scores_pos_ind = is_event_scores > 0.5 # [B, 10]
    scores_mask = scores_pos_ind == 0 # [B, 10]
    # bg_mask = scores_mask * 28 # 28 denotes bg
    _, event_class = event_scores.max(-1) # foreground classification, # [B]
    pred = scores_pos_ind.long()
    pred *= event_class[:, None]
    # add mask
    pred[scores_mask] = bg_flag
    # pred += bg_mask
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())
    return acc



def save_checkpoint(state_dict, top1, task, epoch, seed):
    model_name = f'{args.snapshot_pref}/model_seed_{seed}_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


if __name__ == '__main__':
    main()
