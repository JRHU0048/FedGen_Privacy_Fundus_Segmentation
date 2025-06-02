import argparse
import logging
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.unet_model import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from train import evaluate

# Define function to get test arguments
def get_test_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks using a pretrained model')
    parser.add_argument('--pretrained_model_path', '-p', type=str, required=False, help='Path to the pretrained model')
    parser.add_argument('--test_dir_img', type=str, required=False, help='Path to the test image directory')
    parser.add_argument('--test_dir_mask', type=str, required=False, help='Path to the test mask directory')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    return parser.parse_args()

# Define function to test the model
def test_model(dataset, model, device, amp):
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)

    test_results = evaluate(model, test_loader, device, amp)

    # Print all evaluation metrics
    print(f"Validation Dice: {test_results['dice']:.4f}")
    print(f"Validation Accuracy: {test_results['accuracy']:.4f}")
    print(f"Validation Precision: {test_results['precision']:.4f}")
    print(f"Validation Recall: {test_results['recall']:.4f}")
    print(f"Validation F1 Score: {test_results['f1_score']:.4f}")
    print(f"Validation Specificity: {test_results['specificity']:.4f}")
    print(f"Validation IoU: {test_results['iou']:.4f}")
    
    return test_results

if __name__ == '__main__':
    args = get_test_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    ########################################################################
    args.test_dir_img = "path/to/test/images/"  
    args.test_dir_mask = "path/to/test/masks/" 
    args.pretrained_model_path = "checkpoints/best_model.pth" 
    ########################################################################


    model = UNet(n_channels=3, n_classes=args.classes)

    state_dict = torch.load(args.pretrained_model_path, map_location=device)

    if 'model' in state_dict:
        state_dict = state_dict['model']  
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    model.load_state_dict(state_dict)
    model.to(device=device)

    # Prepare test dataset
    try:
        test_dataset = CarvanaDataset(Path(args.test_dir_img), Path(args.test_dir_mask), args.scale)
    except (AssertionError, RuntimeError, IndexError):
        test_dataset = BasicDataset(Path(args.test_dir_img), Path(args.test_dir_mask), args.scale)

    # Perform testing
    test_results = test_model(test_dataset, model, device, args.amp)
    with open('test_result.txt', 'w') as f:
        f.write(f"Final Test Dice score: {test_results['dice']:.4f}\n")
        f.write(f"Final Test accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"Validation Precision: {test_results['precision']:.4f}\n")
        f.write(f"Validation Recall: {test_results['recall']:.4f}\n")
        f.write(f"Validation F1 Score: {test_results['f1_score']:.4f}\n")
        f.write(f"Validation Specificity: {test_results['specificity']:.4f}\n")
        f.write(f"Validation IoU: {test_results['iou']:.4f}\n")