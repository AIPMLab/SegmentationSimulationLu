import argparse
import logging
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from distutils.version import LooseVersion
from UNet import Unet
from attention_unet import AttU_Net
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vision_transformer import SwinUnet as ViT_segswin
from config import get_config
from datasets import ISIC2018_dataset, LiverDataset, covid19, DriveEyeDataset
from diceloss import val_dice_isic, Intersection_over_Union_isic, get_soft_label, SoftDiceLoss, CombinedLoss
from eval import AverageMeter
from transform import ISIC2018_transform
from unetpp import NestedUNet
import torch
from configtransdeeplab import get_configdeeplab
import importlib
from model.swin_deeplab import SwinDeepLab

from models.HiFormer import HiFormer
import configs.HiFormer_configs as configs

# Define model configurations
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


# Create model function
def create_model(model_id, num_input, num_classes, img_size, args=None, config1=None, config2=None):
    if model_id == 'ViT_seg':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3
        config_vit.pretrained_path = r'.\pretrained_ckpt\imagenet21k_R50+ViT-B_16.npz'
        if isinstance(config_vit.patches.size, tuple):
            patch_size = config_vit.patches.size[0]
        else:
            patch_size = config_vit.patches.size
        config_vit.patches.grid = (img_size // patch_size, img_size // patch_size)
        model = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes).cuda()

        # Load pretrained weights if available
        pretrained_path = config_vit.pretrained_path

        def load_pretrained_weights(model, weights):
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in weights.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        if os.path.exists(pretrained_path):
            weights = np.load(pretrained_path, allow_pickle=True)
            load_pretrained_weights(model, weights)
        else:
            print(f"Pre-trained weights file {pretrained_path} not found, starting from scratch.")

    elif model_id == 'ViT_segswin':
        if config1 is None:
            raise ValueError("Configuration for SwinUnet is required")
        model = ViT_segswin(config1, img_size=img_size, num_classes=num_classes).cuda()
        pretrained_path = config1.MODEL.PRETRAIN_CKPT
        if os.path.exists(pretrained_path):
            model.load_from(config1)
        else:
            print(f"Pre-trained weights file {pretrained_path} not found, starting from scratch.")

    elif model_id == 'hiformer':
        config = configs.get_hiformer_b_configs()
        model = HiFormer(config=config, img_size=img_size, n_classes=num_classes).cuda()
        if config.swin_pretrained_path:
            try:
                model.load_from(config.swin_pretrained_path)
                print("LOADING FINISHED")
            except FileNotFoundError:
                print("keep training")

    elif model_id == 'transdeeplab':
        config = config2
        model_config = importlib.import_module(f'model.configs.{args.config_file}')
        model = SwinDeepLab(
            model_config.EncoderConfig,
            model_config.ASPPConfig,
            model_config.DecoderConfig
        ).cuda()
        try:
            if model_config.EncoderConfig.encoder_name == 'swin' and model_config.EncoderConfig.load_pretrained:
                model.encoder.load_from('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
        except FileNotFoundError:
            print("Pretrained encoder file not found, continuing without loading.")

        try:
            if model_config.ASPPConfig.aspp_name == 'swin' and model_config.ASPPConfig.load_pretrained:
                model.aspp.load_from('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
        except FileNotFoundError:
            print("Pretrained ASPP file not found, continuing without loading.")

        try:
            if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and not model_config.DecoderConfig.extended_load:
                model.decoder.load_from('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
        except FileNotFoundError:
            print("Pretrained decoder file not found, continuing without loading.")

        try:
            if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and model_config.DecoderConfig.extended_load:
                model.decoder.load_from_extended('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
        except FileNotFoundError:
            print("Extended pretrained decoder file not found, continuing without loading.")

    elif model_id == 'NestedUNet':
        model = Test_Model[model_id](num_input, num_classes).cuda()
    else:
        model = Test_Model[model_id](num_input, num_classes).cuda()

    return model

def validate(model, valid_loader, criterion, args, epoch, min_loss, optimizer):
    val_losses = AverageMeter()
    val_dice = AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, (image, target) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            image = image.float().cuda()
            target = target.float().cuda()

            output = model(image)

            output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
            output_soft = get_soft_label(output_dis, args.num_classes)
            target_soft = get_soft_label(target, args.num_classes)
            # val_loss = criterion(output, target, args.num_classes)
            # val_losses.update(val_loss.data, image.size(0))
            val_loss = criterion(output, target_soft, args.num_classes)
            val_losses.update(val_loss.data, image.size(0))

            isic = val_dice_isic(output_soft, target_soft, args.num_classes)
            val_dice.update(isic.data, image.size(0))

            # 在每个 epoch 完成后打印日志
        logging.info(f'Epoch {epoch}: Validation average Dice: {val_dice.avg:.4f}, average Loss: {val_losses.avg:.4f}')
        # print('The Mean Average Dice score: {:.4f}; The Average Loss score: {:.4f}'.format(val_dice.avg,
        #                                                                                 val_losses.avg))

    if val_losses.avg < min(min_loss):
        min_loss.append(val_losses.avg)
        print(min_loss)
        modelname = os.path.join(args.ckpt, args.optimize, f'min_loss_{args.data}_checkpoint.pth.tar')
        directory = os.path.dirname(modelname)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory {directory}")
        print('The best model will be saved at {}'.format(modelname))
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        torch.save(state, modelname)
        logging.info("Save best model to {}".format(modelname))


def main(args):
    min_loss = [1.0]
    start_epoch = args.start_epoch

    # Loading the dataset
    print(f"Loading {args.data} dataset...")

    if args.data == 'eye':
        x_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        y_transforms = transforms.ToTensor()
        train_dataset = DriveEyeDataset('train', transform=x_transforms, target_transform=y_transforms)
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = DriveEyeDataset('val', transform=x_transforms, target_transform=y_transforms)
        validloader = DataLoader(val_dataset, batch_size=1)
        test_dataset = DriveEyeDataset('test', transform=x_transforms, target_transform=y_transforms)
        testloader = DataLoader(test_dataset, batch_size=1)

    elif args.data == 'covid19':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        root = r"D:\paperl\UNET-ZOO-master\dataset\COVID-19_Radiography_Dataset\COVID"
        train_dataset = covid19(root, state='train', transform=transform, mask_transform=mask_transform)
        trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        valid_dataset = covid19(root, state='val', transform=transform, mask_transform=mask_transform)
        validloader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=2)

    elif args.data == 'liver':
        x_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        y_transforms = transforms.ToTensor()
        train_dataset = LiverDataset('train', transform=x_transforms, target_transform=y_transforms)
        trainloader = DataLoader(train_dataset, batch_size=2)
        val_dataset = LiverDataset('val', transform=x_transforms, target_transform=y_transforms)
        validloader = DataLoader(val_dataset, batch_size=2)
        test_dataset = LiverDataset('test', transform=x_transforms, target_transform=y_transforms)
        testloader = DataLoader(test_dataset, batch_size=4)

    else:
        trainset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='train',
                                           transform=Test_Transform[args.data])
        validset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder,
                                           train_type='validation', transform=Test_Transform[args.data])
        trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        validloader = DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        print('Loading is done\n')

    # Define model
    if args.data == 'covid19':
        args.num_input = 1
        args.num_classes = 2
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
    model = create_model(args.id, args.num_input, args.num_classes, args.img_size, args=args, config1=configswin, config2=configdeeplab)

    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = model.cuda()

    print(f"Model architecture for {args.id}:")
    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Set up optimizer and learning rate
    base_lr = args.lr_rate
    if args.optimize == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), base_lr, weight_decay=1e-4)
    elif args.optimize == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), base_lr, weight_decay=1e-4, momentum=0.9)
    elif args.optimize == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), base_lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimize}")

    max_iterations = args.epochs * len(trainloader)
    # 使用CombinedLoss
    # criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
    criterion = SoftDiceLoss()
    save_path = r'D:\unetandvarients\saved_modelall'
    log_directory = os.path.join(save_path, args.id, args.optimize, args.data)
    os.makedirs(log_directory, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_directory, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.resume and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
    else:
        print("=> No checkpoint found, training from scratch.")

    print("Starting training...")

    iter_num = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        losses = AverageMeter()
        train_dice = AverageMeter()  # Initialize the AverageMeter for Dice

        for step, (x, y) in tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}"):
            image = x.float().cuda()
            target = y.float().cuda()

            output = model(image)
            target_soft = get_soft_label(target, args.num_classes)
            # loss = criterion(output, target, args.num_classes)
            # losses.update(loss.data, image.size(0))
            loss = criterion(output, target_soft, args.num_classes)
            losses.update(loss.data, image.size(0))

            # Calculate Dice for the training data
            output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
            output_soft = get_soft_label(output_dis, args.num_classes)
            isic = val_dice_isic(output_soft, target_soft, args.num_classes)
            train_dice.update(isic.data, image.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            # print(f"Epoch: {epoch}, Step: {step}, Learning Rate: {lr_:.6f}")  # 打印学习率的变化
            logging.info(f"Epoch: {epoch}, Step: {step}, Learning Rate: {lr_:.6f}")
            iter_num += 1

        # Log average loss and dice for the training epoch
        logging.info(f'Epoch {epoch} : Loss : {losses.avg:.4f} : Dice : {train_dice.avg:.4f}')

        validate(model, validloader, criterion, args, epoch, min_loss, optimizer)

        if epoch % args.save_epochs_steps == 0:
            filename = os.path.join(args.ckpt, args.optimize, f'{epoch}_{args.data}_checkpoint.pth.tar')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f"Saving model to {filename}")
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, filename)
            logging.info(f"Model saved to {filename}")

    print("Training complete. Starting testing...")


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='segmentation network for biomedical Dataset')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--root_pathsynapse', type=str, default=r'dataset\Synapse\train_npz', help='root dir for data')
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
    parser.add_argument('--cfg', type=str, default=r'.\config\swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE",
                        help='path to config file')
    parser.add_argument('--id', default='Unet', help='a name for identifying the model.')
    # Test_Model = {
    #     'Unet': Unet,
    #     'NestedUNet': NestedUNet,
    #     'Attention_UNet': AttU_Net,
    #     'ViT_seg': ViT_seg,
    #     'ViT_segswin': ViT_segswin,
    #     'hiformer': HiFormer,  # 添加HiFormer模型
    #     'transdeeplab': SwinDeepLab  # 添加transdeeplab模型
    # }
    parser.add_argument('--root_path', default=r"D:\paperl\UNET-ZOO-master\data\ISIC2018_Task1_npy_all",
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models', help='folder to output checkpoints')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continuing from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--num_input', default=1, type=int, help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=99, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=200, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--data', default='liver',
                        help='choose the dataset eye Synapse ISIC2018 liver512 lung covid19 braintumor')
    parser.add_argument('--out_size', default=(224, 224), help='the output image size')
    parser.add_argument('--val_folder', default=r"D:\paperl\UNET-ZOO-master\Datasets\folder1", type=str,
                        help='which cross validation folder')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--deepsupervision', default=0)
    parser.add_argument('--optimize', default='adam', type=str)
    parser.add_argument('--config_file', type=str, default='swin_224_7_1level', help='config file name w/o suffix')
    parser.add_argument('--eval_interval', type=int, default=99, help='evaluation epoch')

    args = parser.parse_args()
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    args.ckpt = os.path.join(args.ckpt, args.data, args.id)
    print('Models are saved at %s' % (args.ckpt))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_' + args.data + '_checkpoint.pth.tar'

    main(args)
