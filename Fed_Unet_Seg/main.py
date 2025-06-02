# -*- coding: utf-8 -*-
# @Description: 
# @Author: Yanhan Hu
# @Date: 2025-05-15
# @LastEditTime: 2025-05-15

import argparse
import logging
import torch
from train import train_model, k_fold_train, TrainingLogger, FederatedTraining
from model.unet_model import UNet

def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train UNet model for image segmentation')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Image downscaling factor')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0, help='Percentage of data for validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision training')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--kfold', '-k', action='store_true', default=True, help='Use 5-fold cross-validation')
    parser.add_argument('--log_file', type=str, default='log/ours_g1020.xlsx', help='Training log file path')
    # FedAvg
    parser.add_argument('--federated', action='store_true', default=True, help='Use federated learning')
    parser.add_argument('--client-config', type=str, default="client_config.json", help='Client configuration file path (JSON format)')
    parser.add_argument('--global-epochs', type=int, default=10, help='Number of global training epochs in federated learning')
    parser.add_argument('--local-epochs', type=int, default=10, help='Number of local training epochs on clients in federated learning')

    return parser.parse_args()

def load_client_config(config_path):
    """Load federated learning configuration file"""
    import json
    with open(config_path) as f:
        config = json.load(f)
    
    required_keys = ['img_dir', 'mask_dir']
    client_configs = []
    
    for client in config['clients']:
        if not all(k in client for k in required_keys):
            raise ValueError(f"Each client configuration must include {required_keys}")
        client_configs.append(client)
    
    server_config = config['server']
    
    return client_configs, server_config

def main():
    args = get_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Initialize UNet model
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network architecture:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed Conv"} upsampling')

    # Load pretrained model
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    torch.cuda.empty_cache()  # Clear cache immediately after initialization
    
    # Initialize training logger
    logger = TrainingLogger(args.log_file)
    
    # ============================= FedAvg learning =========================
    if args.federated:
        if not args.client_config:
            raise ValueError("Federated learning mode requires specifying --client-config parameter")
        
        # Load client configurations
        client_configs, server_config = load_client_config(args.client_config)
        
        logging.info(f'Starting federated learning training with {len(client_configs)} clients')
        for i, config in enumerate(client_configs):
            logging.info(f'Client {i+1} ({config.get("name", "Unnamed")}):')
            logging.info(f'  Image path: {config["img_dir"]}')
            logging.info(f'  Mask path: {config["mask_dir"]}')
        
        federated_trainer = FederatedTraining(
            client_data_configs=client_configs,
            server_data_config=server_config,
            device=device,
            logger=logger
        )
        federated_trainer.initialize_models(model)
        federated_trainer.federated_train(
            global_epochs=args.global_epochs,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_percent=args.val / 100,
            amp=args.amp,
            weight_decay=1e-5,
            gradient_clipping=1.0,
            img_scale=args.scale,
            save_checkpoint=True
        )
    # ============================= FedAvg learning =========================
    else:
        # ============== Train only UNET segmentation model ================== 
        if args.kfold:
            # Use 5-fold cross-validation training
            k_fold_train(
                model=model,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                img_scale=args.scale,
                amp=args.amp,
                logger=logger
            )
        else:
            # Regular training (single training-validation split)
            train_model(
                model=model,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                val_percent=args.val / 100,
                img_scale=args.scale,
                amp=args.amp,
                logger=logger
            )
        # ============== Train only UNET segmentation model ================== 

if __name__ == '__main__':
    main()