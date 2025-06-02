import os
import random
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, JaccardIndex
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss

random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

# ==================================================
# Only the variables that need to be modified when training the UNET segmentation model
dir_img = Path('/home/tangzhiri/yanhanhu/dataset/dataset-preprocessed/G1020-Image-denoised')
dir_mask = Path('/home/tangzhiri/yanhanhu/dataset/dataset-preprocessed/G1020-mask')
dir_checkpoint = Path('checkpoints/')
checkpoint_name = 'best_model.pth'   # The name of the saved model
# ==================================================

class TrainingLogger:
    """ Training logger, used to save training metrics to an Excel file """

    def __init__(self, file_path='training_log.xlsx'):
        self.file_path = file_path
        self.columns = [
            'epoch', 'fold', 'train_loss', 'val_dice', 'val_accuracy', 
            'val_precision', 'val_recall', 'val_f1_score', 
            'val_specificity', 'val_iou'
        ]
        self.df = pd.DataFrame(columns=self.columns)
        
    def add_record(self, record_dict):
        new_row = pd.DataFrame([record_dict])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
    def save_to_excel(self):
        self.df.to_excel(self.file_path, index=False)
        print(f"The training log has been saved to: {self.file_path}")

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    """Evaluate model performance

    Args:
        net: Model to be evaluated
        dataloader: Data loader
        device: Computing device
        amp: Whether to use mixed precision

    Returns:
        Dictionary containing various metrics
    """
    net.eval()
    num_val_batches = len(dataloader)
    
    # Initialize evaluation metrics
    dice_metric = 0
    acc_metric = Accuracy(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes).to(device)
    prec_metric = Precision(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes, average='macro').to(device)
    recall_metric = Recall(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes, average='macro').to(device)
    f1_metric = F1Score(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes, average='macro').to(device)
    spec_metric = Specificity(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes, average='macro').to(device)
    iou_metric = JaccardIndex(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes).to(device)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='验证轮次', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # Move the images and labels to the specified device
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict masks
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'The true mask values should be within the range of [0, 1]'
                # mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float().squeeze(1)
                # Calculate the Dice score
                dice_metric += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                
                # Update binary classification metrics
                mask_true_float = mask_true.float()
                acc_metric.update(mask_pred, mask_true_float)
                prec_metric.update(mask_pred, mask_true_float)
                recall_metric.update(mask_pred, mask_true_float)
                f1_metric.update(mask_pred, mask_true_float)
                spec_metric.update(mask_pred, mask_true_float)
                iou_metric.update(mask_pred, mask_true_float)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'The true mask values should be within the range of [0, classes]'
                mask_true_one_hot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_one_hot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # Calculate the Dice score, ignoring the background
                dice_metric += multiclass_dice_coeff(mask_pred_one_hot[:, 1:], mask_true_one_hot[:, 1:], reduce_batch_first=False)
                
                # Update multi-classification metrics
                acc_metric.update(mask_pred.argmax(dim=1), mask_true)
                prec_metric.update(mask_pred.argmax(dim=1), mask_true)
                recall_metric.update(mask_pred.argmax(dim=1), mask_true)
                f1_metric.update(mask_pred.argmax(dim=1), mask_true)
                spec_metric.update(mask_pred.argmax(dim=1), mask_true)
                iou_metric.update(mask_pred.argmax(dim=1), mask_true)

    net.train()

    # Calculate final metrics
    avg_dice = dice_metric / max(num_val_batches, 1)
    avg_acc = acc_metric.compute()
    avg_prec = prec_metric.compute()
    avg_recall = recall_metric.compute()
    avg_f1 = f1_metric.compute()
    avg_spec = spec_metric.compute()
    avg_iou = iou_metric.compute()

    acc_metric.reset()
    prec_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    spec_metric.reset()
    iou_metric.reset()

    return {
        'dice': avg_dice,
        'accuracy': avg_acc,
        'precision': avg_prec,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'specificity': avg_spec,
        'iou': avg_iou
    }

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.2,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-4,
        gradient_clipping: float = 1.0,
        fold_num: int = 0,
        logger: TrainingLogger = None
):

    """Training model function

    Parameters:
        model: Model to be trained
        device: Training device (cpu/cuda)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        val_percent: Proportion of validation set
        save_checkpoint: Whether to save checkpoints
        img_scale: Image scaling ratio
        amp: Whether to use mixed-precision training
        weight_decay: Weight decay coefficient
        gradient_clipping: Gradient clipping threshold
        fold_num: Identification of cross-validation fold number
        logger: Training log recorder
    """
    
    from utils.data_loading import BasicDataset, CarvanaDataset
    
    # 1. Create a dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split the dataset into training set and validation set
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(random_seed))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up optimizer, loss function, etc.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Start training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs} (Fold {fold_num + 1})', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # Use mixed-precision training
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                # Backpropagation and parameter update
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'Loss (batch)': loss.item()})

        # Validation evaluation
        val_metrics = evaluate(model, val_loader, device, amp)
        scheduler.step(val_metrics['dice'])
        
        # Log training metrics
        if logger is not None:
            log_record = {
                'epoch': epoch,
                'fold': fold_num + 1,
                'train_loss': epoch_loss / len(train_loader),
                'val_dice': val_metrics['dice'].item(),
                'val_accuracy': val_metrics['accuracy'].item(),
                'val_precision': val_metrics['precision'].item(),
                'val_recall': val_metrics['recall'].item(),
                'val_f1_score': val_metrics['f1_score'].item(),
                'val_specificity': val_metrics['specificity'].item(),
                'val_iou': val_metrics['iou'].item()
            }
            logger.add_record(log_record)

        # Save checkpoint
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            checkpoint_path = dir_checkpoint / f'fold{fold_num + 1}_epoch{epoch}.pth'
            torch.save(state_dict, str(checkpoint_path))

def k_fold_train(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        n_splits: int = 5,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        logger: TrainingLogger = None
):
    """Perform k-fold cross-validation training
    
    Args:
        model: The model to train
        device: Training device (cpu/cuda)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        n_splits: Number of cross-validation folds
        save_checkpoint: Whether to save checkpoints
        img_scale: Image scaling ratio
        amp: Whether to use mixed precision training
        weight_decay: Weight decay coefficient
        gradient_clipping: Gradient clipping threshold
        logger: Training logger
    """
    
    from utils.data_loading import BasicDataset, CarvanaDataset
    
    # Load dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # Initialize k-fold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Record the best score for each fold
    fold_scores = []
    
    current_best_model = 0.0   # Track the best model across all folds and epochs

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        # Create data subsets
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        # Create data loaders
        loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
        train_loader = DataLoader(dataset, sampler=train_subsampler, **loader_args)
        val_loader = DataLoader(dataset, sampler=val_subsampler, **loader_args)
        
        # Copy the model for training in the new fold
        model_copy = type(model)(n_channels=model.n_channels, n_classes=model.n_classes, bilinear=model.bilinear)
        model_copy.to(device=device, memory_format=torch.channels_last)
        
        # Set up optimizer, loss function, etc.
        optimizer = optim.Adam(model_copy.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
        criterion = nn.CrossEntropyLoss() if model_copy.n_classes > 1 else nn.BCEWithLogitsLoss()
        
        best_score = 0.0 

        # Training loop
        for epoch in range(1, epochs + 1):
            model_copy.train()
            epoch_loss = 0
            
            with tqdm(total=len(train_ids), desc=f'Epoch {epoch}/{epochs} (Fold {fold + 1})', unit='img') as pbar:
                for batch in train_loader:
                    images, true_masks = batch['image'], batch['mask']
                    
                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)
                    
                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        masks_pred = model_copy(images)
                        if model_copy.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model_copy.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                    
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_copy.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    
                    pbar.update(images.shape[0])
                    epoch_loss += loss.item()
                    pbar.set_postfix(**{'Loss (batch)': loss.item()})
            
            # Validation evaluation
            val_metrics = evaluate(model_copy, val_loader, device, amp)
            scheduler.step(val_metrics['dice'])

            # Log training metrics
            if logger is not None:
                log_record = {
                    'epoch': epoch,
                    'fold': fold + 1,
                    'train_loss': epoch_loss / len(train_loader),
                    'val_dice': val_metrics['dice'].item(),
                    'val_accuracy': val_metrics['accuracy'].item(),
                    'val_precision': val_metrics['precision'].item(),
                    'val_recall': val_metrics['recall'].item(),
                    'val_f1_score': val_metrics['f1_score'].item(),
                    'val_specificity': val_metrics['specificity'].item(),
                    'val_iou': val_metrics['iou'].item()
                }
                logger.add_record(log_record)

            # Update best score
            if val_metrics['dice'] > best_score:
                best_score = val_metrics['dice']

            # Save checkpoint
            if save_checkpoint and best_score > current_best_model:
                current_best_model = best_score
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model_copy.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                checkpoint_path = dir_checkpoint / checkpoint_name
                torch.save(state_dict, str(checkpoint_path))

        # Record best score
        fold_scores.append(best_score)
        print(f'Fold {fold + 1} completed with best Dice score: {best_score:.4f}')

    # Output cross-validation results
    print('Cross-validation completed. Dice scores for each fold:')
    for i, score in enumerate(fold_scores):
        print(f'Fold {i + 1}: {score:.4f}')
    print(f'Average Dice score: {sum(fold_scores)/len(fold_scores):.4f}')
    
    # Save training logs
    if logger is not None:
        logger.save_to_excel()    



class FederatedTraining:
    def __init__(self, client_data_configs, server_data_config, device, logger=None):
        self.num_clients = len(client_data_configs)
        self.client_configs = client_data_configs
        self.server_config = server_data_config
        self.device = device
        self.logger = logger
        self.global_model = None
        
    def initialize_models(self, model_template):
        """Initialize the global model (without preallocating client models)"""
        self.global_model = model_template
        torch.cuda.empty_cache()
        
    def _create_client_model(self):
        """Dynamically create client models"""
        model = type(self.global_model)(
            n_channels=self.global_model.n_channels,
            n_classes=self.global_model.n_classes,
            bilinear=self.global_model.bilinear
        ).to(self.device)
        model.load_state_dict(self.global_model.state_dict())
        return model
        
    def client_train(self, client_idx, epochs=1, batch_size=1, lr=1e-5, 
                    val_percent=0.2, amp=False, weight_decay=1e-8, 
                    gradient_clipping=1.0, img_scale=0.5):
        from utils.data_loading import BasicDataset, CarvanaDataset
        
        try:
            # 1.
            config = self.client_configs[client_idx]
            dataset = CarvanaDataset(Path(config['img_dir']), Path(config['mask_dir']), img_scale)
        except:
            dataset = BasicDataset(Path(config['img_dir']), Path(config['mask_dir']), img_scale)
            
        # 2. 
        n_val = int(len(dataset) * val_percent)
        train_set, val_set = random_split(dataset, [len(dataset)-n_val, n_val], 
                                        generator=torch.Generator().manual_seed(random_seed))
        
        # 3. 
        loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
        
        # 4. Create client models
        model = self._create_client_model()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        grad_scaler = torch.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        

        # Distinguish between single-class and multi-class scenarios
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        
        # Distinguish between single-class and multi-class loss calculations
        if model.n_classes == 1:
            # Binary classification scenario: Use sigmoid and binary Dice loss
            loss_fn = lambda preds, masks: criterion(preds.squeeze(1), masks.float()) + \
                dice_loss(F.sigmoid(preds.squeeze(1)), masks.float(), multiclass=False)
        else:
            # Multi-class scenario: Use softmax and multi-class Dice loss
            loss_fn = lambda preds, masks: criterion(preds, masks) + \
                dice_loss(F.softmax(preds, dim=1), 
                        F.one_hot(masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True)

        # 5. Training loop
        accumulation_steps = 4  # Gradient accumulation steps
        best_score = 0.0
        
        try:
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                epoch_loss = 0
                
                for i, batch in enumerate(train_loader):
                    with torch.autocast(self.device.type, enabled=amp, dtype=torch.float16):
                        images = batch['image'].to(dtype=torch.float16, device=self.device)
                        masks = batch['mask'].to(device=self.device)
                        
                        preds = model(images)
                        loss = loss_fn(preds, masks)  # Use a unified loss function interface
                        
                        # Gradient accumulation
                        loss = loss / accumulation_steps
                        grad_scaler.scale(loss).backward()
                    
                    # Update parameters periodically
                    if (i+1) % accumulation_steps == 0:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        
                        del images, masks, preds
                        torch.cuda.empty_cache()
                    
                    epoch_loss += loss.item() * accumulation_steps
                
                # validation
                val_metrics = evaluate(model, val_loader, self.device, amp)
                current_score = val_metrics['dice'].item()
                if current_score > best_score:
                    best_score = current_score
                
                logging.info(f'Client {client_idx+1} Epoch {epoch+1}/{epochs} - '
                            f'Loss: {epoch_loss/len(train_loader):.4f}, '
                            f'Dice: {current_score:.4f}')
                
        finally:
            state_dict = model.state_dict()
            del model, optimizer, grad_scaler
            torch.cuda.empty_cache()
            return state_dict, len(train_set), best_score
    
    def aggregate(self, client_updates, client_sizes):
        total_size = sum(client_sizes)
        averaged = {}
        
        # Aggregate on the CPU
        with torch.no_grad():
            for key in client_updates[0].keys():
                averaged[key] = torch.zeros_like(client_updates[0][key].cpu())
                
            for update, size in zip(client_updates, client_sizes):
                weight = size / total_size
                for key in averaged.keys():
                    averaged[key] += update[key].cpu() * weight
            
            for key in averaged.keys():
                averaged[key] = averaged[key].to(self.device)
                
        self.global_model.load_state_dict(averaged)
        torch.cuda.empty_cache()
    
    def federated_train(self, global_epochs=10, local_epochs=10, 
                       batch_size=1, lr=1e-5, val_percent=0.2, amp=False, 
                       weight_decay=1e-8, gradient_clipping=1.0, img_scale=0.5,
                       save_checkpoint=True):
        """Federated training main loop"""
        best_global_score = 0.0
        
        for global_epoch in range(1, global_epochs+1):
            logging.info(f'\n=== Global Epoch {global_epoch}/{global_epochs} ===')
            
            # Train all clients
            updates, sizes, scores = [], [], []
            for client_idx in range(self.num_clients):
                torch.cuda.empty_cache()
                
                update, size, score = self.client_train(
                    client_idx=client_idx,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    val_percent=val_percent,
                    amp=amp,
                    weight_decay=weight_decay,
                    gradient_clipping=gradient_clipping,
                    img_scale=img_scale
                )
                updates.append(update)
                sizes.append(size)
                scores.append(score)
                
            self.aggregate(updates, sizes)
            
            # Server evaluation
            server_metrics = self._evaluate_on_server(
                batch_size=batch_size,
                img_scale=img_scale,
                amp=amp
            )
            
            if self.logger:
                self._log_metrics(global_epoch, scores, server_metrics)
            
            # Save the best model
            current_score = server_metrics['dice'].item()
            if save_checkpoint and current_score > best_global_score:
                best_global_score = current_score
                self._save_checkpoint(global_epoch, best_global_score)

        # Save training logs
        if self.logger is not None:
            self.logger.save_to_excel()
    
    def _evaluate_on_server(self, batch_size, img_scale, amp):
        """Evaluate on server data"""
        from utils.data_loading import BasicDataset, CarvanaDataset
        config = self.server_config[0]
        
        try:
            dataset = CarvanaDataset(
                Path(config['img_dir']),
                Path(config['mask_dir']),
                img_scale
            )
        except:
            dataset = BasicDataset(
                Path(config['img_dir']),
                Path(config['mask_dir']),
                img_scale
            )
            
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return evaluate(self.global_model, loader, self.device, amp)
    
    def _log_metrics(self, epoch, client_scores, server_metrics):
        """Record training metrics"""
        log_record = {
            'epoch': epoch,
            'fold': 0, 
            'train_loss': sum(client_scores)/len(client_scores), 
            'val_dice': server_metrics['dice'].item(),
            'val_accuracy': server_metrics['accuracy'].item(),
            'val_precision': server_metrics['precision'].item(),
            'val_recall': server_metrics['recall'].item(),
            'val_f1_score': server_metrics['f1_score'].item(),
            'val_specificity': server_metrics['specificity'].item(),
            'val_iou': server_metrics['iou'].item(),
            'client_scores': '|'.join(f'{s:.4f}' for s in client_scores)
        }
        self.logger.add_record(log_record)

    def _save_checkpoint(self, epoch, score):
        Path('checkpoints').mkdir(exist_ok=True)
        state_dict = {
            'model': self.global_model.state_dict(),
            'mask_values': getattr(self.global_model, 'mask_values', None),
            'score': score
        }
        torch.save(state_dict, f'checkpoints/{checkpoint_name}.pth')
        logging.info(f'Saved checkpoint at epoch {epoch} with score {score:.4f}')
