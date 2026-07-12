import os
import time
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Set environment variables to reduce warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='No device id is provided')
warnings.filterwarnings('ignore', category=UserWarning, message='.*OMP_NUM_THREADS.*')

# Import our modules
from src.utils import (
    set_seed, get_device, print_gpu_info, load_config, save_config,
    create_output_dirs, setup_logging, save_checkpoint, load_checkpoint,
    AverageMeter, accuracy, TrainingState, TimeEstimator, EarlyStopping,
    print_model_info, get_lr, create_optimizer_and_scheduler
)
from src.models import create_model
from src.data import create_imagenet1k_dataloaders as create_dataloaders
from src.data.pretrain.dataloader import validate_dataset


class SoftTargetCrossEntropy(nn.Module):
    """Soft target cross entropy for mixup/cutmix"""
    def forward(self, logits, soft_targets):
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()


def setup_ddp():
    """Initialize distributed training with robust environment variable handling."""
    # Support multiple launcher formats (torchrun, slurm, etc.)
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
    
    # Set device before initializing process group to avoid warnings
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        # Initialize without device_id parameter (newer PyTorch versions handle this automatically)
        dist.init_process_group(backend='nccl')
    
    return local_rank


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """Get world size."""
    return dist.get_world_size() if dist.is_initialized() else 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MTA-Swin Transformer Multi-GPU Pretraining on ImageNet')
    parser.add_argument('--config', type=str, default='configs/pretrain/mta_swin_ddp_v2.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate the model')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs (for testing)')
    parser.add_argument('--local-rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    return parser.parse_args()


def safe_save_checkpoint(checkpoint_data, save_dir, filename):
    """Safely save checkpoint with atomic write operation."""
    filepath = os.path.join(save_dir, filename)
    temp_filepath = filepath + '.tmp'
    
    try:
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Save to temporary file first
        torch.save(checkpoint_data, temp_filepath)
        
        # Atomic move to final location
        if os.path.exists(temp_filepath):
            os.rename(temp_filepath, filepath)
            return True
        else:
            raise RuntimeError(f"Temporary file {temp_filepath} was not created")
            
    except Exception as e:
        # Clean up temporary file if it exists
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except:
                pass
        raise e


def train_one_epoch(
    model, train_loader, optimizer, criterion, device, epoch, config, writer, logger, 
    scaler=None, local_rank=0, mixup_fn=None, scheduler=None, is_step_based=False
):
    """Train model for one epoch with AMP and DDP support."""
    model.train()
    
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    total_steps = len(train_loader)
    print_freq = config.get('print_freq', 100)
    use_amp = config.get('performance', {}).get('amp', False)
    amp_dtype = config.get('performance', {}).get('amp_dtype', 'bf16')
    
    # Set autocast dtype
    if use_amp:
        autocast_dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
    
    # Set epoch for distributed sampler
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    
    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Apply Mixup/CutMix augmentation
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP support
        if use_amp:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Backward pass with AMP support
        if use_amp and amp_dtype == 'fp16' and scaler is not None:
            # fp16 needs gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping with AMP
            if config.get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # bf16 or no AMP
            loss.backward()
            
            # Standard gradient clipping
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
        
        # Step-based scheduler update
        if scheduler and is_step_based:
            scheduler.step()
        
        # Measure accuracy and record loss
        # For mixup, use argmax for logging purposes (accuracy is approximate)
        if mixup_fn is not None:
            targets_hard = targets.argmax(dim=1)
            acc1, acc5 = accuracy(outputs, targets_hard, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        
        # Log progress (only on main process)
        if is_main_process() and step % print_freq == 0:
            current_lr = get_lr(optimizer)
            amp_status = f"AMP-{amp_dtype.upper()}" if use_amp else "FP32"
            world_size = get_world_size()
            mixup_status = "+Mixup" if mixup_fn else ""
            logger.info(
                f'Epoch: [{epoch}][{step}/{total_steps}] ({world_size}GPU-{amp_status}{mixup_status}) '
                f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                f'Acc@5 {top5.val:.3f} ({top5.avg:.3f}) '
                f'LR {current_lr:.6f}'
            )
            
            # TensorBoard logging
            if writer:
                global_step = epoch * total_steps + step
                writer.add_scalar('Train/Loss', losses.val, global_step)
                writer.add_scalar('Train/Acc1', top1.val, global_step)
                writer.add_scalar('Train/LR', current_lr, global_step)
    
    if is_main_process():
        logger.info(
            f'Train Epoch {epoch}: '
            f'Loss {losses.avg:.4f} '
            f'Acc@1 {top1.avg:.3f} '
            f'Acc@5 {top5.avg:.3f}'
        )
    
    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device, epoch, logger):
    """Validate model with distributed metric aggregation."""
    model.eval()
    
    total_loss = torch.tensor(0.0, device=device)
    total_acc1 = torch.tensor(0.0, device=device)
    total_acc5 = torch.tensor(0.0, device=device)
    total_num  = torch.tensor(0.0, device=device)
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            bs = torch.tensor(images.size(0), device=device, dtype=torch.float32)
            total_loss += loss.detach() * bs
            total_acc1 += acc1[0].detach() * bs
            total_acc5 += acc5[0].detach() * bs
            total_num  += bs
    
    # Aggregate across all processes
    if dist.is_initialized():
        for t in (total_loss, total_acc1, total_acc5, total_num):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    
    loss_avg = (total_loss / total_num).item()
    acc1_avg = (total_acc1 / total_num).item()
    acc5_avg = (total_acc5 / total_num).item()
    
    if is_main_process():
        logger.info(
            f'Val Epoch {epoch}: '
            f'Loss {loss_avg:.4f} '
            f'Acc@1 {acc1_avg:.3f} '
            f'Acc@5 {acc5_avg:.3f}'
        )
    
    return loss_avg, acc1_avg, acc5_avg


def main():
    """Main training function."""
    args = parse_args()

    # DDP initialization
    local_rank = setup_ddp()
    world_size = get_world_size()
    if is_main_process():
        print("MTA-Swin Transformer Multi-GPU Pretraining on ImageNet")
        print(f"Config: {args.config}")
        print(f"World Size: {world_size} GPUs")

    # Config / seed / device setup
    config = load_config(args.config)
    if args.epochs is not None:
        config['epochs'] = args.epochs
        if is_main_process():
            print(f"Testing mode: Training for {args.epochs} epochs")

    set_seed(config.get('seed', 42))
    device = torch.device(f'cuda:{local_rank}')
    if is_main_process():
        print_gpu_info()

    # I/O directories and logging (only rank 0)
    if is_main_process():
        model_name = config['model']['name']
        paths = create_output_dirs(config['output_dir'], model_name, 'pretrain')
        logger = setup_logging(paths['log_file'])
        logger.info("Starting MTA-Swin Transformer multi-GPU pretraining")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"World Size: {world_size} GPUs")
        save_config(config, os.path.join(paths['log_dir'], 'config.yaml'))
    else:
        logger, paths = None, None

    # Dataset validation (only rank 0) then barrier sync
    if is_main_process():
        if not validate_dataset(config['data']['data_dir']):
            logger.error("Dataset validation failed!")
            cleanup_ddp()
            return
    if dist.is_initialized():
        dist.barrier()

    # Create dataloaders
    if is_main_process():
        logger.info("Creating data loaders with distributed sampling...")

    total_batch_size = config['data']['batch_size']
    per_gpu_batch_size = total_batch_size // world_size
    assert total_batch_size % world_size == 0, \
        f"data.batch_size ({total_batch_size}) must be divisible by world_size ({world_size})"

    if is_main_process():
        logger.info(f"Total batch size: {total_batch_size}")
        logger.info(f"Per GPU batch size: {per_gpu_batch_size}")
    train_loader, val_loader, mixup_fn = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=per_gpu_batch_size,
        num_workers=config['data'].get('num_workers', 4),
        img_size=config['data'].get('img_size', 224),
        pin_memory=config['data'].get('pin_memory', True),
        augment_train=True,
        config=config,
        distributed=True
    )
    if is_main_process():
        logger.info(f"Train batches per GPU: {len(train_loader)}, Val batches per GPU: {len(val_loader)}")
        if mixup_fn: logger.info("Mixup/CutMix augmentation enabled")

    # Dataset info (only rank 0)
    if is_main_process():
        from src.data.pretrain.dataloader import get_dataset_info
        dataset_info = get_dataset_info(config['data']['data_dir'])
        logger.info(f"Dataset: {dataset_info['dataset_name']}")
        logger.info(f"Classes: {dataset_info['num_classes']}")
        logger.info(f"Train samples: {dataset_info['train_size']}")
        logger.info(f"Val samples: {dataset_info['val_size']}")
        if config['model']['num_classes'] != dataset_info['num_classes']:
            logger.warning(f"Updating num_classes from {config['model']['num_classes']} to {dataset_info['num_classes']}")
            config['model']['num_classes'] = dataset_info['num_classes']

    # Create model
    if is_main_process():
        logger.info("Creating MTA-Swin model...")
    model_config = config['model']
    model = create_model(
        model_name=model_config['name'],
        num_classes=model_config['num_classes'],
        img_size=config['data'].get('img_size', 224),
        drop_path_rate=model_config.get('drop_path_rate', 0.1),
        stage_qk_conv=model_config.get('stage_qk_conv', [True, True, False, False]),
        stage_head_mixing=model_config.get('stage_head_mixing', [False, False, True, True]),
        stage_group_norm=model_config.get('stage_group_norm', [True, True, False, False]),
        cq=model_config.get('cq', 6),
        ck=model_config.get('ck', 11),
        ch=model_config.get('ch', 2),
        fused_window_process=model_config.get('fused_window_process', False)
    ).to(device)

    # Print full configuration (only rank 0)
    if is_main_process():
        logger.info("=" * 60)
        logger.info("FULL CONFIGURATION:")
        logger.info("=" * 60)
        try:
            import yaml
            for line in yaml.dump(config, default_flow_style=False, indent=2).split('\n'):
                if line.strip(): logger.info(line)
        except Exception:
            import pprint
            logger.info(pprint.pformat(config, indent=2, width=80))
        logger.info("=" * 60)

    # Enable gradient checkpointing
    if config.get('performance', {}).get('gradient_checkpointing', False):
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
            if is_main_process(): logger.info("Gradient checkpointing enabled")

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False, broadcast_buffers=True)

    if is_main_process():
        print_model_info(model.module)
        logger.info(f"Model created: {config['model']['name']}")
        logger.info("Model wrapped with DistributedDataParallel")

    # Create optimizer and scheduler
    steps_per_epoch = len(train_loader)
    optimizer, scheduler, is_step_scheduler = create_optimizer_and_scheduler(
        model.module, config, steps_per_epoch
    )

    # Setup loss functions
    if mixup_fn is not None:
        criterion_train = SoftTargetCrossEntropy()
        criterion_val = nn.CrossEntropyLoss()
        if is_main_process():
            logger.info("Using SoftTargetCrossEntropy for Mixup (train)")
            logger.info("Using CrossEntropyLoss for evaluation (val)")
    else:
        criterion_train = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))
        criterion_val = nn.CrossEntropyLoss()
        if is_main_process():
            ls = config.get('label_smoothing', 0.0)
            logger.info(f"Using CrossEntropyLoss with label smoothing: {ls} (train)" if ls > 0
                        else "Using CrossEntropyLoss without label smoothing (train)")
            logger.info("Using CrossEntropyLoss for evaluation (val)")

    # Setup AMP
    scaler = None
    use_amp = config.get('performance', {}).get('amp', False)
    amp_dtype = config.get('performance', {}).get('amp_dtype', 'bf16')
    if use_amp and amp_dtype == 'fp16':
        scaler = torch.cuda.amp.GradScaler()
        if is_main_process(): logger.info("AMP enabled (fp16)")
    elif use_amp and amp_dtype == 'bf16':
        if is_main_process(): logger.info("AMP enabled (bf16)")

    # Setup CUDNN benchmark
    if config.get('performance', {}).get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
        if is_main_process(): logger.info("CUDNN benchmark enabled")

    # Training utilities (only rank 0)
    if is_main_process():
        state = TrainingState()
        timer = TimeEstimator(config['epochs'])
        early_stopping = None
        if config.get('early_stopping', {}).get('enabled', False):
            es = config['early_stopping']
            early_stopping = EarlyStopping(
                patience=es['patience'], min_delta=es['min_delta'],
                mode=es['mode'], restore_best_weights=es['restore_best_weights']
            )
            logger.info(f"Early stopping enabled (patience: {es['patience']})")
        writer = SummaryWriter(log_dir=paths['tensorboard_dir']) if config.get('tensorboard', True) else None
        if writer: logger.info(f"TensorBoard logging enabled: {paths['tensorboard_dir']}")
    else:
        state = timer = early_stopping = writer = None

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_main_process(): logger.info("Loaded optimizer state")
        except Exception as e:
            if is_main_process():
                logger.warning(f"Could not load optimizer state: {e}")
        if scheduler and checkpoint.get('scheduler_state_dict') is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if is_main_process(): logger.info("Loaded scheduler state")
            except Exception as e:
                if is_main_process(): logger.warning(f"Could not load scheduler state: {e}")
        if scaler is not None and amp_dtype == 'fp16' and 'scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                if is_main_process(): logger.info("Loaded AMP scaler state")
            except Exception as e:
                if is_main_process(): logger.warning(f"Could not load AMP scaler state: {e}")

        # Move optimizer tensors to current device
        for st in optimizer.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(device)

        start_epoch = checkpoint['epoch'] + 1
        if is_main_process():
            state.best_acc = checkpoint.get('best_acc', 0.0)
            logger.info(f"Resumed from {args.resume} (epoch {start_epoch}, best acc {state.best_acc:.3f}%)")

    # Evaluation only mode
    if args.eval_only:
        if is_main_process(): logger.info("Running evaluation only...")
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion_val, device, 0, logger)
        if is_main_process():
            logger.info(f"Eval - Loss: {val_loss:.4f}, Acc@1: {val_acc1:.3f}%, Acc@5: {val_acc5:.3f}%")
        cleanup_ddp()
        return

    # Start training
    if is_main_process():
        logger.info("Starting MTA-Swin pretraining...")
        timer.start_training()

    try:
        for epoch in range(start_epoch, config['epochs']):
            if is_main_process():
                epoch_start_time = time.time()
                timer.start_epoch()

            # Training
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model, train_loader, optimizer, criterion_train, device, epoch, config,
                writer, logger, scaler, local_rank, mixup_fn, scheduler, is_step_scheduler
            )
            # Validation
            val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion_val, device, epoch, logger)

            # Warmup logging and state update (only rank 0)
            if is_main_process():
                current_lr_before_step = get_lr(optimizer)
                warmup_epochs = config.get('scheduler', {}).get('warmup_epochs', 0)
                if warmup_epochs > 0:
                    if epoch < warmup_epochs:
                        logger.info(f"Warmup {epoch+1}/{warmup_epochs}, LR {current_lr_before_step:.6f}")
                    elif epoch == warmup_epochs:
                        logger.info(f"Warmup complete, LR {current_lr_before_step:.6f}")
                current_lr = get_lr(optimizer)
                is_best = state.update(epoch, train_loss, val_loss, val_acc1, current_lr)
                torch.cuda.empty_cache()

            # Epoch-based scheduler update (all ranks)
            if scheduler and not is_step_scheduler:
                scheduler.step()

            # Early stopping: rank 0 decision then broadcast to all ranks
            should_stop = False
            if early_stopping and is_main_process():
                should_stop = early_stopping(val_acc1, model.module.state_dict(), epoch)
                if should_stop and early_stopping.restore_best_weights and early_stopping.get_best_weights():
                    model.module.load_state_dict(early_stopping.get_best_weights())
                    logger.info("Best weights restored")
            if dist.is_initialized():
                stop_flag = torch.tensor(1 if should_stop else 0, device=device, dtype=torch.int)
                dist.broadcast(stop_flag, src=0)
                should_stop = bool(stop_flag.item())
            if should_stop:
                if is_main_process(): logger.info("Early stopping triggered!")
                # Sync all ranks before exiting training loop
                if dist.is_initialized(): dist.barrier()
                break

            # Synchronization before saving: ensure all GPUs reach this point
            if dist.is_initialized():
                dist.barrier()

            # I/O operations (TensorBoard and checkpoints) - only rank 0
            if is_main_process():
                if writer:
                    writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
                    writer.add_scalar('Epoch/Train_Acc1', train_acc1, epoch)
                    writer.add_scalar('Epoch/Train_Acc5', train_acc5, epoch)
                    writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                    writer.add_scalar('Epoch/Val_Acc1', val_acc1, epoch)
                    writer.add_scalar('Epoch/Val_Acc5', val_acc5, epoch)
                    writer.add_scalar('Epoch/LR', current_lr, epoch)

                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_acc': state.best_acc,
                    'config': config,
                    'world_size': world_size,
                    'mta_config': {
                        'stage_qk_conv': model_config.get('stage_qk_conv', [True, True, False, False]),
                        'stage_head_mixing': model_config.get('stage_head_mixing', [False, False, True, True]),
                        'stage_group_norm': model_config.get('stage_group_norm', [True, True, False, False]),
                        'cq': model_config.get('cq', 6),
                        'ck': model_config.get('ck', 11),
                        'ch': model_config.get('ch', 2)
                    }
                }

                try:
                    if (epoch + 1) % config.get('save_freq', 10) == 0:
                        safe_save_checkpoint(checkpoint_data, paths['checkpoint_dir'],
                                             f'checkpoint_epoch_{epoch:03d}.pth')
                        logger.info(f"Saved checkpoint for epoch {epoch}")
                    if is_best:
                        safe_save_checkpoint(checkpoint_data, paths['checkpoint_dir'], 'best_model.pth')
                        logger.info(f"New best accuracy: {state.best_acc:.3f}%")
                    safe_save_checkpoint(checkpoint_data, paths['checkpoint_dir'], 'latest_model.pth')
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at epoch {epoch}: {e}")

                # Record timing and summary
                epoch_end_time = time.time()
                minutes = int((epoch_end_time - epoch_start_time) // 60)
                seconds = int((epoch_end_time - epoch_start_time) % 60)
                logger.info(f"Epoch {epoch} completed in {minutes}m {seconds}s ({world_size} GPUs)")
                timer.end_epoch(epoch)
                summary = state.get_summary()
                logger.info(
                    f"Epoch {epoch} Summary: "
                    f"Train {train_loss:.4f}, Val {val_loss:.4f}, "
                    f"Val@1 {val_acc1:.3f}%, Best {summary['best_acc']:.3f}% (Ep {summary['best_epoch']}), "
                    f"LR {current_lr:.6f}"
                )

            # Synchronization after I/O: prevent next iteration from starting early
            if dist.is_initialized():
                torch.cuda.synchronize()   # optional but recommended
                dist.barrier()

        # Training completion
        if is_main_process():
            logger.info("MTA-Swin Multi-GPU Pretraining completed!")
            final_summary = state.get_summary()
            logger.info(f"Final Results:")
            logger.info(f"  Best Accuracy: {final_summary['best_acc']:.3f}% (Epoch {final_summary['best_epoch']})")
            logger.info(f"  Total Epochs: {final_summary['total_epochs']}")
            history_path = os.path.join(paths['log_dir'], 'training_history.json')
            state.save_history(history_path)
            speed_stats = timer.get_speed_stats()
            if speed_stats:
                logger.info("Training Speed Stats:")
                for k, v in speed_stats.items():
                    logger.info(f"  {k}: {v:.2f}s")

    except KeyboardInterrupt:
        if is_main_process(): logger.info("Training interrupted by user")
    except Exception as e:
        if is_main_process(): logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Sync all ranks before exit to avoid resource competition
        if dist.is_initialized():
            dist.barrier()
        if is_main_process() and writer:
            writer.close()
        cleanup_ddp()


if __name__ == '__main__':
    main()