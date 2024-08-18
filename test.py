import argparse
import logging
import os
import time
import torch.utils.data as Data
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from UNet import Unet
from attention_unet import AttU_Net
from config import get_config
from datasets import ISIC2018_dataset, LiverDataset, covid19, DriveEyeDataset
from diceloss import val_dice_isic, Intersection_over_Union_isic, get_soft_label
from transform import ISIC2018_transform
from unetpp import NestedUNet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vision_transformer import SwinUnet as ViT_segswin

from configtransdeeplab import get_configdeeplab
import importlib
from model.swin_deeplab import SwinDeepLab

from models.HiFormer import HiFormer
import configs.HiFormer_configs as configs
from utils import assd

# Setup logging configuration
logging.basicConfig(filename='test_log.log', level=logging.INFO, format='%(asctime)s %(message)s')
Test_Model = {
    'Unet': Unet,
    'Attention_UNet': AttU_Net,
    'NestedUNet': NestedUNet,
    'ViT_seg': ViT_seg,
    'ViT_segswin': ViT_segswin,
    'hiformer': HiFormer,
    'transdeeplab': SwinDeepLab
}

Test_Dataset = {'ISIC2018': ISIC2018_dataset,
                'liver': LiverDataset,
                'covid19': covid19,
                'eye': DriveEyeDataset}

Test_Transform = {'ISIC2018': ISIC2018_transform}


def create_model(model_id, num_input, num_classes, img_size, args=None, config1=None, config2=None):
    if model_id == 'ViT_seg':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3
        config_vit.pretrained_path = r'.\pretrained_ckp0\imagenet21k_R50+ViT-B_16.npz'
        if isinstance(config_vit.patches.size, tuple):
            patch_size = config_vit.patches.size[0]
        else:
            patch_size = config_vit.patches.size
        config_vit.patches.grid = (img_size // patch_size, img_size // patch_size)
        model = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes).cuda()

        # Load pretrained weights if available
        pretrained_path = config_vit.pretrained_path

        def load_pretrained_weights(model, weights):
            """Load pretrained weights, ignoring missing or incompatible keys."""
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in weights.items() if k in model_dict and model_dict[k].size() == v.size()}
            missing_keys = [k for k in model_dict.keys() if k not in pretrained_dict.keys()]
            incompatible_keys = [k for k in weights.keys() if k not in pretrained_dict.keys()]
            print(f"Missing keys: {missing_keys}")
            print(f"Incompatible keys: {incompatible_keys}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        if os.path.exists(pretrained_path):
            weights = np.load(pretrained_path, allow_pickle=True)
            print("Loaded pre-trained weights:", weights.keys())
            load_pretrained_weights(model, weights)
        else:
            print(f"Pre-trained weights file {pretrained_path} not found, starting from scratch.")
    elif model_id == 'ViT_segswin':
        if config1 is None:
            raise ValueError("Configuration for SwinUnet is required")
        model = ViT_segswin(config1, img_size=img_size, num_classes=num_classes).cuda()
    elif model_id == 'hiformer':
        CONFIGS = {

            'hiformer-b': configs.get_hiformer_b_configs(),

        }
        config = CONFIGS['hiformer-b']  # hiformer-b
        model = HiFormer(config=config, img_size=img_size, n_classes=num_classes).cuda()
    elif model_id == 'transdeeplab':
        config = config2
        model_config = importlib.import_module(f'model.configs.{args.config_file}')
        model = SwinDeepLab(
            model_config.EncoderConfig,
            model_config.ASPPConfig,
            model_config.DecoderConfig
        ).cuda()
    elif model_id == 'NestedUNet':
        model = Test_Model[model_id](num_input, num_classes).cuda()
    else:
        model = Test_Model[model_id](num_input, num_classes).cuda()  # unet++args
    return model

def test(test_loader, model, args):
    model.eval()
    isic_dice = []
    isic_iou = []
    isic_assd = []
    infer_time = []

    save_dir = os.path.join('./results', args.id, args.data, args.optimize, 'predictions')
    os.makedirs(save_dir, exist_ok=True)

    # Load best model checkpoint
    modelname = r'.\saved_models\liver\Unet\adam\min_loss_liver_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print(f"=> Loading checkpoint '{modelname}'")
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
    else:
        print(f"=> No checkpoint found at '{modelname}'")
        return

    with torch.no_grad():
        save_count = 0  # Initialize save_count outside the loop

        for step, (img, lab) in enumerate(test_loader):
            image = img.float().cuda()
            target = lab.float().cuda()

            # Timing the inference
            begin_time = time.time()
            output = model(image)
            end_time = time.time()
            infer_time.append(end_time - begin_time)

            # Post-process the output and target
            output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
            output_soft = get_soft_label(output_dis, args.num_classes)
            target_soft = get_soft_label(target, args.num_classes)

            label_arr = target_soft.cpu().numpy().astype(np.uint8)
            output_arr = output_soft.cpu().byte().numpy().astype(np.uint8)

            # Calculate metrics
            isic_b_dice = val_dice_isic(output_soft, target_soft, args.num_classes)
            isic_b_iou = Intersection_over_Union_isic(output_soft, target_soft, args.num_classes)
            isic_b_asd = assd(output_arr[:, :, :, 1], label_arr[:, :, :, 1])

            isic_dice.append(isic_b_dice.data.cpu().numpy())
            isic_iou.append(isic_b_iou.data.cpu().numpy())
            isic_assd.append(isic_b_asd)
            # 使用 `torchvision.transforms.Resize` 调整大小
            resize_transform = transforms.Resize((target.size(2), target.size(3)))
            img_resized = resize_transform(img)
            output_resized = resize_transform(output_dis.float())
            target_resized = resize_transform(target.float())
            # Save the first 10 images, labels, and predictions
            if save_count < 10:
                img_save_dir = os.path.join(save_dir, f'sample_{save_count}')
                os.makedirs(img_save_dir, exist_ok=True)

                img_name = os.path.join(img_save_dir, 'image.png')
                label_name = os.path.join(img_save_dir, 'label.png')
                pred_name = os.path.join(img_save_dir, 'prediction.png')

                # Save the images, labels, and predictions
                torchvision.utils.save_image(img_resized.cpu(), img_name)
                torchvision.utils.save_image(target_resized.cpu(), label_name)
                torchvision.utils.save_image(output_resized.cpu(), pred_name)

                save_count += 1

        # Summary of results
        isic_dice_mean = np.mean(isic_dice)
        isic_dice_std = np.std(isic_dice)
        isic_iou_mean = np.mean(isic_iou)
        isic_iou_std = np.std(isic_iou)
        isic_assd_mean = np.mean(isic_assd)
        isic_assd_std = np.std(isic_assd)
        all_time = np.sum(infer_time)

        print(f'The ISIC mean Accuracy: {isic_dice_mean:.4f}; The ISIC Accuracy std: {isic_dice_std:.4f}')
        print(f'The ISIC mean IoU: {isic_iou_mean:.4f}; The ISIC IoU std: {isic_iou_std:.4f}')
        print(f'The ISIC mean ASSD: {isic_assd_mean:.4f}; The ISIC ASSD std: {isic_assd_std:.4f}')
        print(f'Total inference time: {all_time:.4f} seconds')

        print('Testing Done!')



def main_test(args):
    print('loading the test dataset ...')

    if args.data == 'eye':
        x_transforms = transforms.Compose([
            transforms.ToTensor(),  # -> [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
        ])
        y_transforms = transforms.ToTensor()
        test_dataset = DriveEyeDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        testloader = DataLoader(test_dataset, batch_size=1)
    elif args.data =='covid19':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        root = r".\dataset\COVID-19_Radiography_Dataset\COVID"
        test_dataset = covid19(root, state='test', transform=transform, mask_transform=mask_transform)
        testloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)#16？
    elif args.data == 'liver':
        x_transforms = transforms.Compose([
            transforms.ToTensor(),  # -> [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
        ])
        # mask只需要转换为tensor
        y_transforms = transforms.ToTensor()
        # x_transforms = transforms.Compose([
        #     transforms.Resize((256, 256)),  # 首先调整图像大小以适应中心裁剪
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])
        # y_transforms = transforms.Compose([
        #     transforms.Resize((256, 256)),  # 首先调整图像大小以适应中心裁剪
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        # ])
        test_dataset = LiverDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        testloader = DataLoader(test_dataset, batch_size=1)
    else:
        testset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test',
                                           transform=Test_Transform[args.data])

        testloader = Data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    print('Loading is done\n')

    # Define model
    if args.data == 'covid19':
        args.num_input = 1
        args.num_classes = 2
        args.out_size = (224, 224)
        args.out_size = (224, 224)
    elif args.data == 'ISIC2018':
        args.num_input = 3
        args.num_classes = 2
        args.out_size = (224, 224)
    elif args.data == 'eye':
        args.num_input = 3
        args.num_classes = 2
        args.out_size = (576, 576)
    else:
        args.num_input = 3
        args.num_classes = 2
        args.out_size = (512, 512)

    configswin = get_config(args)
    configdeeplab = get_configdeeplab(args)
    model = create_model(args.id, args.num_input, args.num_classes, args.img_size, args=args, config1=configswin,
                         config2=configdeeplab)

    if torch.cuda.is_available():
        print('We can use', torch.cuda.device_count(), 'GPUs to test the network')
        model = model.cuda()

    print('Start testing ...')
    test(testloader, model, args)
    print('Testing Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='segmentation network for biomedical Dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='cache mode options')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')
    parser.add_argument('--cfg', type=str,
                        default=r'.\config\swin_tiny_patch4_window7_224_lite.yaml',
                        metavar="FILE", help='path to config file')
    parser.add_argument('--id', default='Unet',
                        help='a name for identitying the model. Choose from the following options: ViT_segswin')
    parser.add_argument('--volume_path', type=str,
                        default=r'D:\lyytest\UNET-ZOO-master\dataset\Synapse\test_vol_h5', help='root dir for validation volume data')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--root_path', default=r".\data\ISIC2018_Task1_npy_all",
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes2')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=200, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--data', default='liver', help='choose the dataset eye liver lung ISIC2018 Synapse covid19')
    parser.add_argument('--out_size', default=(512, 512), help='the output image size')
    parser.add_argument('--val_folder', default=r"\Datasets\folder1", type=str,
                        help='which cross validation folder')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--img_size', type=int, default=512,help='input patch size of network input512')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--deepsupervision', default=0)
    parser.add_argument('--optimize', default='adam', type=str)
    parser.add_argument('--config_file', type=str,
                        default='swin_224_7_1level', help='config file name w/o suffix')
    parser.add_argument('--eval_interval', type=int,
                        default=99, help='evaluation epoch')
    args = parser.parse_args()
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    main_test(args)
