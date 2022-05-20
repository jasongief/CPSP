from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import *
from nets.psp_net import PSP_MMIL_Net as MMIL_Net
from utils.eval_metrics import segment_level, event_level
import pandas as pd
import pickle as pkl
import pdb

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    criterion2 = LabelSmoothingLoss(10, smoothing=0.1)
    criterion_vcon = VconInfoLoss(ratio=args.ratio, eta=args.eta)
    accs = []
    num = 0

    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['video_st'].to('cuda'), sample['label'].type(torch.FloatTensor).to('cuda')
        # audio2 = sample['audio2'].to('cuda')
        data_idx = sample['idx']
        optimizer.zero_grad()
        # output, a_prob, v_prob, _, sims, mask = model(audio, video, video_st, audio2)
        av_fea, output, a_prob, v_prob, _, sims, mask = model(audio, video, video_st)
        output.clamp_(min=1e-7, max=1 - 1e-7)
        a_prob.clamp_(min=1e-7, max=1 - 1e-7)
        v_prob.clamp_(min=1e-7, max=1 - 1e-7)

        Pa = sample['pa'].type(torch.FloatTensor).to('cuda')
        Pv = sample['pv'].type(torch.FloatTensor).to('cuda')
        
        b=audio.size(0)
        loss1 =  criterion(a_prob, Pa) 
        loss2 =  criterion(v_prob, Pv) 
        loss3 =  criterion(output, target) 

        loss4 = criterion2(sims, mask)
        loss = loss1 + loss2 + loss3 + loss4 

        loss5 = torch.Tensor([0])
        if args.vcon_flag:
            loss5 = criterion_vcon(av_fea, target)
            if not torch.isnan(loss5):
                loss += args.lambda_v * loss5


        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tLoss: {:.3f}\tLoss: {:.3f}\tLoss: {:.3f}\tLoss: {:.3f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss1.item(),  loss2.item(), loss3.item(), loss4.item(), loss5.item()))


def eval(model, val_loader, set):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    model.eval()

    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv("data/AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv("data/AVVP_eval_visual.csv", header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'),sample['video_st'].to('cuda'), sample['label'].to('cuda')
            # output, a_prob, v_prob, frame_prob, a, is_real = model(audio, video, video_st, audio)
            _, output, a_prob, v_prob, frame_prob, a, is_real = model(audio, video, video_st)
            o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)
            oa = (a_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
            ov = (v_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
            
            Pa = frame_prob[0, :, 0, :].cpu().detach().numpy()
            Pv = frame_prob[0, :, 1, :].cpu().detach().numpy()

            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(oa, repeats=10, axis=0)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(ov, repeats=10, axis=0)

            # extract audio GT labels
            GT_a = np.zeros((25, 10))
            GT_v =np.zeros((25, 10))
            GT_aa = np.zeros(25, dtype=np.int)
            GT_vv = np.zeros(25, dtype=np.int)

            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num >0:
                for i in range(num):

                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1
                    GT_aa[idx] = 1

            # extract visual GT labels
            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1
                    GT_vv[idx]=1
            GT_av = GT_a * GT_v

            # obtain prediction matrices
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)

    print("\n")    
    print('Audio  \t {:.1f} \t {:.1f}'.format( 100 * np.mean(np.array(F_seg_a)),  100 * np.mean(np.array(F_event_a))))
    print('Visual \t {:.1f} \t {:.1f}'.format( 100 * np.mean(np.array(F_seg_v)),  100 * np.mean(np.array(F_event_v))))
    print('AudVis \t {:.1f} \t {:.1f}'.format( 100 * np.mean(np.array(F_seg_av)),  100 * np.mean(np.array(F_event_av))))

    avg_type = (100 * np.mean(np.array(F_seg_av))+100 * np.mean(np.array(F_seg_a))+100 * np.mean(np.array(F_seg_v)))/3.
    avg_event = 100 * np.mean(np.array(F_seg))
    print('Segment-levelType@Avg. F1: {:.1f}'.format(avg_type))
    print('Segment-level Event@Avg. F1: {:.1f}'.format( avg_event))

    avg_type_event = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(
        np.array(F_event_v))) / 3.
    avg_event_level = 100 * np.mean(np.array(F_event))
    print('Event-level Type@Avg. F1: {:.1f}'.format( avg_type_event))
    print('Event-level Event@Avg. F1: {:.1f}'.format(avg_event_level))
    print("\n")
    return avg_type


def main():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    parser.add_argument(
        "--audio_dir", type=str, default='data/feats/vggish/', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='data/feats/res152/',
        help="video dir")
    parser.add_argument(
        "--st_dir", type=str, default='data/feats/r2plus1d_18/',
        help="video dir")
    parser.add_argument(
        "--label_train", type=str, default="data/AVVP_train.csv", help="weak train csv file")
    parser.add_argument(
        "--label_val", type=str, default="data/AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument(
        "--label_test", type=str, default="data/AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=12, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='MMIL_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='PSP_MMIL_Net',
        help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')
    parser.add_argument(
        '--threshold', type=float, default=0.099, help='threshold value for psp module'
    )
    parser.add_argument(
        '--vcon_flag', action="store_true", help="video contrastive loss for cpsp"
    )
    parser.add_argument(
        '--ratio', type=float, default=1.0, help='ratio for vcon_loss'
    )
    parser.add_argument(
        '--eta', type=float, default=1.0, help='eta for vcon_loss'
    )
    parser.add_argument(
        '--lambda_v', type=float, default=1.0, help='weight for vcon_loss'
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # model = MMIL_Net().to('cuda')
    model = MMIL_Net(args).to('cuda')

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir, exist_ok=True)
    model_save_path = args.model_save_dir + args.checkpoint + "_bs_%d_threshold_%f_ratio_%.2f_eta_%.2f"%(args.batch_size, args.threshold, args.ratio, args.eta) + ".pt"

    if args.mode == 'retrain':
        train_dataset = LLP_dataset(train=True, label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([ToTensor()]))
        val_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([ToTensor()]))
        test_dataset = LLP_dataset(train=False, label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,
            st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.BCELoss()
        best_F = 0
 
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step()
            print("Validation Performance of Epoch {}:".format(epoch))
            F = eval(model, val_loader, args.label_val)
                
        print("Test the latest model:")
        F=eval(model, test_loader, args.label_test)

        # save model
        torch.save(model.state_dict(), model_save_path)
        print('model is saved to %s'%model_save_path)
            
    elif args.mode == 'val':
        test_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                    st_dir=args.st_dir, transform=transforms.Compose([
                ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(model_save_path))
        eval(model, test_loader, args.label_val)
    else:
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,  st_dir=args.st_dir, transform = transforms.Compose([
                                               ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(model_save_path))
        eval(model, test_loader, args.label_test)
if __name__ == '__main__':
    main()
