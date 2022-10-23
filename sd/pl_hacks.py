import os
import shutil

import pytorch_lightning as pl
from weakref import proxy

reset_lr = False
reset_optimizers = False
reset_callbacks = False

# Make PL accept a checkpoint without optimizer states.
# If pl_hacks.reset_optimizers is set to True, optimizers and LR schedulers from the checkpoint are not restored
# If pl_hacks.reset_lr is set to True, LR and LR schedulers from the checkpoint are not restored
def restore_optimizers_and_schedulers(self) -> None:
    """Restores the optimizers and learning rate scheduler states from the pre-loaded checkpoint."""
    if not self._loaded_checkpoint:
        return

    no_opt = False
    if self.trainer.strategy.lightning_restore_optimizer:
        if 'optimizer_states' not in self._loaded_checkpoint:
            print('Optimizer states not present in checkpoint')
            no_opt = True
        elif reset_optimizers:
            print('Optimizer states present in checkpoint but not restoring because pl_hacks.reset_optimizers == True') 
        else:
            print('Optimizer states present in checkpoint and restoring') 
            if reset_lr:
                print('Resetting LR in checkpoint because pl_hacks.reset_lr == True')
                optimizer_states = self._loaded_checkpoint['optimizer_states']
                for optimizer, opt_state in zip(self.trainer.strategy.optimizers, optimizer_states):
                    for group, group_state in zip(optimizer.param_groups, opt_state['param_groups']):
                        group_state['lr'] = group['lr']
                        if 'initial_lr' in group:
                            group_state['initial_lr'] = group['initial_lr']
            self.restore_optimizers()
    elif reset_lr:
        print('Warning: pl_hacks.reset_lr == True, but PL training strategy uses its own checkpoint loading, LR may not be reset!')

    if 'lr_schedulers' not in self._loaded_checkpoint:
        print('LR schedulers not present in checkpoint')
    elif no_opt:
        print('LR schedulers present in checkpoint but not restoring because there are no optimizer states')
    elif reset_optimizers:
        print('LR schedulers present in checkpoint but not restoring because pl_hacks.reset_optimizers == True')
    elif reset_lr:
        print('LR schedulers present in checkpoint but not restoring because pl_hacks.reset_lr == True')
    else:
        print('LR schedulers present in checkpoint and restoring')
        self.restore_lr_schedulers()

pl.trainer.connectors.checkpoint_connector.CheckpointConnector.restore_optimizers_and_schedulers = restore_optimizers_and_schedulers

# If pl_hacks.reset_callbacks is set to True, do not load callback states from the checkpoint
orig_reset_callbacks = pl.trainer.connectors.checkpoint_connector.CheckpointConnector.restore_callbacks
def restore_callbacks(self):
    if 'callbacks' in self._loaded_checkpoint:
        for k,v in self._loaded_checkpoint['callbacks'].items():
            if k == pl.callbacks.model_checkpoint.ModelCheckpoint:
                if 'dirpath' in v and 'stable-diffusion' in v['dirpath']:
                    print('Detected stable-diffusion checkpoint: Ignoring callbacks')
                    return
    
    if not reset_callbacks:
        orig_reset_callbacks(self)

pl.trainer.connectors.checkpoint_connector.CheckpointConnector.restore_callbacks = restore_callbacks


# ModelCheckpoint where save_last only triggers on the specified every_n_epochs interval, instead of every epoch (useful for short epochs)
# Also uses temporary files to prevent partial overwrites of large SD checkpoints and only saves the checkpoint once if
# it is set up to store multiples (e.g. both topk and last)
class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, use_temporary_file=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_temporary_file = use_temporary_file
    
    def on_train_epoch_end(self, trainer, pl_module):
        if not self._should_skip_saving_checkpoint(trainer) and self._save_on_train_epoch_end:
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
                self._save_last_checkpoint(trainer, monitor_candidates)
    
    def on_validation_end(self, trainer, pl_module):
        if not self._should_skip_saving_checkpoint(trainer) and not self._save_on_train_epoch_end:
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
                self._save_last_checkpoint(trainer, monitor_candidates)
                
    def _save_checkpoint(self, trainer, filepath):
        parts = os.path.split(filepath)
        tmp_path = os.path.join(parts[0], '_temp.ckpt')
        
        if self._last_global_step_saved == trainer.global_step:
            # Copy checkpoint if it already saved to disk this step (e.g. topk + last)
            if self.use_temporary_file:
                shutil.copy(self._last_filename, tmp_path)
                shutil.move(tmp_path, filepath)
            else:
                shutil.copy(self._last_filename, filepath)
            return
        
        if self.use_temporary_file:
            # Save checkpoint to temporary file first, the move to the final filename to avoid overwriting ckpt with a partial one
            trainer.save_checkpoint(tmp_path, self.save_weights_only)
            shutil.move(tmp_path, filepath)
        else:
            trainer.save_checkpoint(filepath, self.save_weights_only)
        self._last_filename = filepath

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
