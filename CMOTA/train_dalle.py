import argparse, yaml
import glob
import os
import datetime

import torch
from cmota.models.vqvae import VQVAE, EMAVQVAE, GumbelVQVAE
from cmota.models.cond_transformer import Net2NetTransformer

from cmota.loader import StoryTextImageDataModule 
from cmota.modules.dalle.tokenizer import HugTokenizer, YttmTokenizer#, tokenizer
from cmota.modules.dalle.tokenizer_mindalle import build_tokenizer

from cmota.callbacks import DalleGenerativeStorySampler

from torchvision import transforms as T

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import XLAStatsMonitor, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if (epoch+1) % 10 == 0 and global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

if __name__ == "__main__":

    # argument parsing
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


    parser = argparse.ArgumentParser(description='DALL-E Training for Pytorch')

    # Added configuration for recurrent transformer
    parser.add_argument('--n_memory_cells', type=int, default=1, help='n_memory_cells')  
    parser.add_argument('--load_images', action='store_true', default=False, help='load images in ram') 

    parser.add_argument('--special_case_img_seq_len', action='store_true', default=False, help='64 image size to 16x16 token size')

    # Cyclic Pseudo-Texts training
    parser.add_argument('--pseudo_i2t', action='store_true', default=False, help='Pseudo text to image generation training for supporting contextual information')
    parser.add_argument('--pseudo_filtering', action='store_true', default=False, help='64 for 128')

    parser.add_argument('--pretrained_from_small', action='store_true', default=False, help='64 for 128')

    #path configuration
    parser.add_argument('--data_dir', type=str, default='dataset/ducogan_pororo/', help='path to train/val dataset')               
    parser.add_argument('--log_dir', type=str, default='results/', help='path to save logs')
    parser.add_argument('--output_imgs_dir', type=str, default='output_imgs/test', help='path to save logs')
                    

    parser.add_argument('--vae_path', type=str, help='path to your trained VAE')
    parser.add_argument('--bpe_path', type=str, help='path to your BPE json file')
    parser.add_argument('--backup_dir', type=str, default='backups/', help='path to save backups for sudden crash') 
    parser.add_argument('--ckpt_path', type=str,default='results/checkpoints/last.ckpt', help='path to previous checkpoint') 
    parser.add_argument('--gpu_dist', action='store_true', default=False, help='distributed training with gpus') 

    parser.add_argument('--top_p', type=float, default=0.9, help="top p sampling")
    parser.add_argument('--save_step_frequency', type=int, default=500)

    parser.add_argument('--hybrid_mask', action = 'store_true')   

    #training configuration
    parser.add_argument('--backup', action='store_true', default=False,
                    help='save backup and load from backup if restart happens')      
    parser.add_argument('--backup_steps', type =int, default = 1000,
                    help='saves backup every n training steps')                   
    parser.add_argument('--image_log_steps', type=int, default=1000,
                    help='log image outputs for every n step. not recommended for tpus')   
    parser.add_argument('--refresh_rate', type=int, default=1,
                    help='progress bar refresh rate')    
    parser.add_argument('--precision', type=int, default=32,
                    help='precision for training')                                                                             
    parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume from checkpoint')                   
    parser.add_argument('--seed', type=int, default=42,
                    help='random seed')  
    parser.add_argument('--gpus', type=int, default=16,
                    help='number of gpus')                                   
    parser.add_argument('--num_sanity_val_steps', type=int, default=0,
                    help='num_sanity_val_steps') 

    parser.add_argument('--batch_size', type=int, default=12,
                    help='training settings')  
    parser.add_argument('--epochs', type=int, default=100,
                    help='training settings')  
    parser.add_argument('--learning_rate', default=4.5e-6, type=float,
                    help='base learning rate')
    parser.add_argument('--lr_decay', action = 'store_true')                                   
    parser.add_argument('--num_workers', type=int, default=16,
                    help='training settings')   
    parser.add_argument('--img_size', type=int, default=256,
                    help='training settings')  
    parser.add_argument('--clip_grad_norm', default = 0.5, type = float, help = 'Clip gradient norm')
    parser.add_argument('--hug', dest='hug', action='store_true')
    parser.add_argument('--resize_ratio', type=float, default=0.75,
                    help='Random resized crop lower ratio')
    parser.add_argument('--ga_steps', default = 1, type = int, 
                    help = 'Number of steps to accumulate gradients across per each iteration.')                
    parser.add_argument('--truncate_captions', dest='truncate_captions', action='store_true',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')

    parser.add_argument('--test', action='store_true', default=False,
                    help='test run')    
    parser.add_argument('--debug', action='store_true', default=False,
                    help='debug run') 
    parser.add_argument('--xla_stat', action='store_true', default=False,
                    help='print out tpu related stat')     
    parser.add_argument('--dataset_size', nargs='+', type=int, default=[1e9],
                    help='training settings')                                       
    #VAE configuration
    parser.add_argument('--vae', type=str, default='evqvae')
    parser.add_argument('--finetuned', action='store_true', default=False,
                    help='whether vae is finetuned')                 
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,2,2,4],
                    help='resnet channel multiplier') 
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=[32],
                    help='model settings')                          
    parser.add_argument('--loss_type', type=str, default='mse')

    #Transformer configuration
    parser.add_argument('--hidden_dim', default = 512, type = int, 
                    help = 'Model dimension')
    parser.add_argument('--text_seq_len', default = 128, type = int, 
                    help = 'Text sequence length')                
    parser.add_argument('--depth', default = 32, type = int, 
                    help = 'Model depth')
    parser.add_argument('--heads', default = 16, type = int, 
                    help = 'Model number of heads')
    parser.add_argument('--loss_img_weight', default = 7.0, type = float, 
                    help = 'Image loss weight')
    parser.add_argument('--keep_prob', default = 1.0, type = float, 
                    help = 'token keep prob')
    parser.add_argument('--wandb', action='store_true', default=False, help='use wandb for logging')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.

    args = parser.parse_args(remaining)

    print("====="*10)
    print("args")
    print(args)
    print("====="*10)
        
    #random seed fix
    seed_everything(args.seed)   

    gpus = args.gpus      
    if args.gpu_dist:
        torch.distributed.init_process_group(backend='nccl') 
        args.world_size = torch.distributed.get_world_size()
    else:
        args.world_size = args.gpus     


    args.base_lr = args.learning_rate              
    args.learning_rate = args.learning_rate * args.world_size * args.batch_size

    # tokenizer
    if exists(args.bpe_path):
        klass = HugTokenizer if args.hug else YttmTokenizer
        tokenizer = klass(args.bpe_path)  

    tokenizer = build_tokenizer('tokenizer', context_length=80, lowercase=True, dropout=None)
    # pororo
    tokenizer.add_tokens(['pororo', 'loopy', 'eddy', 'harry', 'poby', 'tongtong', 'crong', 'rody', 'petty'])
    # Flintstone
    #tokenizer.add_tokens(['fred', 'barney', 'wilma', 'betty', 'pebbles', 'dino', 'slate'])

    args.num_text_tokens = tokenizer.get_vocab_size()

    print(f'Using BPE model with vocab size: {args.num_text_tokens}')
    default_root_dir = args.log_dir

    if args.resume:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = None

    checkpoint_callback = ModelCheckpoint(monitor="val/total_loss", save_last=True)

    if args.backup:
        args.backup_dir = os.path.join(args.backup_dir, f'dalle/{args.vae}')
        backup_callback = ModelCheckpoint(
                                    dirpath=args.backup_dir,
                                    every_n_train_steps = args.backup_steps,
                                    filename='{epoch}_{step}'
                                    )
        
        if len(glob.glob(os.path.join(args.backup_dir,'*.ckpt'))) != 0 :
            ckpt_path = sorted(glob.glob(os.path.join(args.backup_dir,'*.ckpt')))[-1]
            if args.resume:
                print("Setting default ckpt to {}. If this is unexpected behavior, remove {}".format(ckpt_path, ckpt_path))
                
    limit_train_batches = 1.0
    limit_test_batches = 1.0   

    # model
    if args.vae == 'vqvae':
        if args.finetuned:
            vae = VQVAE.load_from_checkpoint(args.vae_path, finetuned=True, 
                                                ft_attn_resolutions=args.attn_resolutions, 
                                                ft_loss_type = args.loss_type, 
                                                ft_args = args)
        else:
            vae = VQVAE.load_from_checkpoint(args.vae_path)
        vae.setup_eval()

    elif args.vae == 'evqvae': 
        if args.finetuned:
            vae = EMAVQVAE.load_from_checkpoint(args.vae_path, finetuned=True, 
                                                ft_attn_resolutions=args.attn_resolutions, 
                                                ft_loss_type = args.loss_type, 
                                                ft_args = args)
        else:
            vae = EMAVQVAE.load_from_checkpoint(args.vae_path)      
        vae.setup_eval()                       
    elif args.vae == 'gvqvae': 
        if args.finetuned:
            vae = GumbelVQVAE.load_from_checkpoint(args.vae_path, finetuned=True, 
                                                ft_attn_resolutions=args.attn_resolutions, 
                                                ft_loss_type = args.loss_type, 
                                                ft_args = args)
        else:
            vae = GumbelVQVAE.load_from_checkpoint(args.vae_path)      
        vae.setup_eval()


    model = Net2NetTransformer(args, args.batch_size, args.learning_rate, vae=vae, keep_prob = args.keep_prob, cond_stage_key='text', special_case_img_seq_len=args.special_case_img_seq_len)

    if args.pretrained_from_small:
        m = torch.load(args.ckpt_path)['state_dict']
        model_dict = model.state_dict()

        for k in m.keys():
            if 'first_stage_model' in k:
                continue

            if k in model_dict:
                pname = k
                pval = m[k]
                model_dict[pname] = pval.clone().to(model_dict[pname].device)

        model.load_state_dict(model_dict, strict=False)
        ckpt_path = None

    datamodule = StoryTextImageDataModule(args.data_dir,
                                          args.batch_size, args.num_workers, 
                                          args.img_size, args.text_seq_len, 
                                          args.resize_ratio,args.truncate_captions, 
                                          tokenizer,
                                          world_size = args.world_size,
                                          dataset_size = args.dataset_size,
                                          load_images=args.load_images,
                                          debug_mode=args.debug)
                
                         
    logger = pl.loggers.tensorboard.TensorBoardLogger(args.log_dir, name='dalle')    
                    

    trainer = Trainer(gpus= gpus, default_root_dir=default_root_dir,
                        max_epochs=args.epochs, progress_bar_refresh_rate=args.refresh_rate,precision=args.precision,
                        accelerator='ddp', accumulate_grad_batches=args.ga_steps,
                        gradient_clip_val=args.clip_grad_norm,
                        num_sanity_val_steps=args.num_sanity_val_steps,
                        limit_train_batches=limit_train_batches,limit_test_batches=limit_test_batches,                          
                        resume_from_checkpoint = ckpt_path, callbacks=[checkpoint_callback],
                        logger=logger)

    trainer.callbacks.append(LearningRateMonitor())
    if args.backup:
        trainer.callbacks.append(backup_callback)      
    if args.resume:    
        trainer.callbacks.append(ModelCheckpoint()) 
    
    # Text to Image generation logging
    trainer.callbacks.append(DalleGenerativeStorySampler(every_n_steps=args.image_log_steps, tokenizer=tokenizer, top_p=args.top_p, img_size=args.img_size, depth=args.depth))

    print("Setting batch size: {} learning rate: {:.2e} * {} * {} = {:.2e}".format(model.hparams.batch_size,args.base_lr,args.world_size,args.batch_size, model.hparams.learning_rate))
    
    if not args.test:    
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule)
