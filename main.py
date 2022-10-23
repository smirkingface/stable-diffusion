import argparse
import os
import sys
import datetime
import shutil
import sd.pl_hacks
import yaml

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from sd.util import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        default='',
        help='postfix for logdir',
    )
    parser.add_argument(
        '-r',
        '--resume',
        type=str,
        help='Path to logging directory to resume',
    )
    parser.add_argument(
        '-c',
        '--config',
        help='Configuration filename'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Enable test',
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=23,
        help='seed for seed_everything',
    )
    parser.add_argument(
        '-f',
        '--postfix',
        type=str,
        default='',
        help='post-postfix for default name',
    )
    parser.add_argument(
        '-l',
        '--logdir',
        type=str,
        default='logs',
        help='Base directory for experiment logs',
    )
    parser.add_argument(
        '--disable_scale_lr',
        action='store_true',
        help='Scale base learning rate by: number of gpus * batch_size * number of gradient accumulations',
    )
    parser.add_argument(
        '--reset_optimizer',
        action='store_true',
        help='When resuming training, reset optimizer from config (implies --reset_lr)',
    )
    parser.add_argument(
        '--reset_lr',
        action='store_true',
        help='When resuming training, reset LR and LR scheduler from config',
    )
    parser.add_argument(
        '--reset_callbacks',
        action='store_true',
        help='When resuming training, reset callback states',
    )
    parser.add_argument(
        '--reset_ema',
        action='store_true',
        help='When resuming training, reset EMA (automatically true if no EMA weights are in the checkpoint)',
    )
    
    return parser

if __name__ == '__main__':
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    # Add current working directory to path for convenience and to make classes in this directory available when running as `python main.py`
    sys.path.append(os.getcwd())

    # Parse commandline arguments
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt = parser.parse_args()
    if opt.name and opt.resume:
        raise ValueError(
            '-n/--name and -r/--resume cannot be specified both.'
            'If you want to resume training in a new log folder, '
            'use -n/--name in combination with --resume_from_checkpoint'
        )

    # When resuming training, use commandline option to determine if LR and LR schedulers, optimizer, and/or callbacks state need to be reset
    if opt.resume or opt.resume_from_checkpoint:
        sd.pl_hacks.reset_lr = opt.reset_lr
        sd.pl_hacks.reset_optimizers = opt.reset_optimizer
        sd.pl_hacks.reset_callbacks = opt.reset_callbacks

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError('Cannot find {}'.format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split('/')
            logdir = '/'.join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip('/')
            ckpt = os.path.join(logdir, 'checkpoints', 'last.ckpt')

        opt.resume_from_checkpoint = ckpt
        _tmp = logdir.split('/')
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = '_' + opt.name
        elif opt.config:
            cfg_fname = os.path.split(opt.config)[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = '_' + cfg_name
        else:
            name = ''
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, 'checkpoints')
    cfgdir = os.path.join(logdir, 'configs')
    seed_everything(opt.seed, workers=True)

    # Load config
    config = yaml.safe_load(open(opt.config, 'r'))
    if 'lightning' not in config:
        config['lightning'] = {}
    trainer_kwargs = dict(config['lightning']['trainer'])

    # Set up logger
    if 'logger' in config['lightning']:
        config['lightning']['logger']['params']['save_dir'] = logdir
        trainer_kwargs['logger'] = instantiate_from_config(config['lightning']['logger'])
    else:
        trainer_kwargs['logger'] = instantiate_from_config({
            'target': 'pytorch_lightning.loggers.CSVLogger',
            'params': {
                'name': 'csv',
                'save_dir': logdir,
            }
        })

    # Set up modelcheckpoint callback
    if 'modelcheckpoint' in config['lightning']:
        config['lightning']['modelcheckpoint']['params']['dirpath'] = ckptdir
        modelckpt_cfg = config['lightning']['modelcheckpoint']
    else:
        modelckpt_cfg = {
            'target': 'sd.pl_hacks.ModelCheckpoint',
            'params': {
                'dirpath': ckptdir,
                'filename': '{epoch:06}',
                'verbose': True,
                'save_last': True,
                'save_top_k': 0
            }
        }
    
    # Set up callbacks
    callbacks = {
        'learning_rate_logger': {
            'target': 'pytorch_lightning.callbacks.LearningRateMonitor',
            'params': {
                'logging_interval': 'step'
            }
        },
        'cuda': {'target': 'sd.callbacks.cuda.CUDACallback'}
    }

    if 'callbacks' in config['lightning']:
        callbacks.update(config['lightning']['callbacks'])
    callbacks['checkpoint'] = modelckpt_cfg
    trainer_kwargs['callbacks'] = [instantiate_from_config(callbacks[k]) for k in callbacks]

    # Default to GPU accelerator
    if opt.accelerator == None:
        opt.accelerator = 'gpu'

    # Disable find_unused_parameters when DDP is used (conflicts with gradient checkpoints)
    if not config['lightning'].get('find_unused_parameters', False):
        if hasattr(opt, 'strategy') and opt.strategy == 'ddp':
            opt.strategy = 'ddp_find_unused_parameters_false'
        if 'strategy' in trainer_kwargs and trainer_kwargs['strategy'] == 'ddp':
            trainer_kwargs['strategy'] = 'ddp_find_unused_parameters_false'

    # Default trainer benchmark option to True
    trainer_kwargs['benchmark'] = trainer_kwargs.get('benchmark', True)

    # Avoid deprecation warning about --resume_from_checkpoint
    ckpt_path = opt.resume_from_checkpoint
    opt.resume_from_checkpoint = None
    
    # Create trainer
    trainer = Trainer.from_argparse_args(opt, **trainer_kwargs)

    # Create data
    data = instantiate_from_config(config['data'])

    # Create model
    if opt.reset_ema:
        config['model']['params']['reset_ema'] = True
    model = instantiate_from_config(config['model'])

    # Configure learning rate
    bs = data.batch_size
    base_lr = model.learning_rate
    ngpu = trainer.num_devices
    accumulate_grad_batches = trainer.accumulate_grad_batches

    print(f'accumulate_grad_batches = {accumulate_grad_batches}')
    if not opt.disable_scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print('Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)'.format(model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        print('++++ NOT USING LR SCALING ++++')
        print(f'Setting learning rate to {model.learning_rate:.2e}')

    # Save config to file
    if trainer.is_global_zero:
        os.makedirs(cfgdir, exist_ok=True)
        yaml.dump(config, open(os.path.join(cfgdir, "{}-project.yaml".format(now)), 'w'), sort_keys=False)
    
    # Run training
    trainer.fit(model, data, ckpt_path=ckpt_path)

    if trainer.is_global_zero:
        print('Saving checkpoint' + (' (fit() interrupted)' if trainer.interrupted else ''))
        
        tmp_path = os.path.join(ckptdir, '_temp.ckpt')
        trainer.save_checkpoint(tmp_path)
        shutil.move(tmp_path, os.path.join(ckptdir, 'last.ckpt'))

    # Test (would we ever use this?)
    if opt.test and not trainer.interrupted:
        trainer.test(model, data)

