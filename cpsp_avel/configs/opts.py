import argparse

parser = argparse.ArgumentParser(description="A project implemented in pyTorch")

# =========================== Learning Configs ============================
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--n_epoch', type=int)
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('--test_batch_size', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--gpu', type=str)
parser.add_argument('--snapshot_pref', type=str)
parser.add_argument('--split', type=str)
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--clip_gradient', type=float)
parser.add_argument('--loss_weights', type=float)
parser.add_argument('--lambda_avps', type=float)
parser.add_argument('--lambda_scon', type=float)
parser.add_argument('--lambda_vcon', type=float)
parser.add_argument('--lambda_sscon', type=float)
parser.add_argument('--avps_flag', action='store_true')
parser.add_argument('--scon_flag', action='store_true')
parser.add_argument('--vcon_flag', action='store_true')
parser.add_argument('--vcon_infoNCE_flag', action='store_true')
parser.add_argument('--start_epoch', type=int)
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
parser.add_argument('--weight_decay', '--wd', type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

# =========================== Data Configs ==============================
parser.add_argument('--vis_maps_path', type=str)

# =========================== Model Configs ==============================
parser.add_argument('--vis_fea_type', type=str)
parser.add_argument('--threshold_value', type=float)
parser.add_argument('--category_num', type=int)

parser.add_argument('--margin', type=float)
parser.add_argument('--neg_num', type=int)
parser.add_argument('--eta', type=float)

# =========================== Display Configs ============================
parser.add_argument('--print_freq', type=int)
parser.add_argument('--val_print_freq', type=int)
parser.add_argument('--save_freq', type=int)
parser.add_argument('--eval_freq', type=int)



