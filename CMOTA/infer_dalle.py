import argparse, yaml
import glob
import os
import datetime

import torch
from cmota.models.vqvae import VQVAE, EMAVQVAE, GumbelVQVAE
from cmota.models.cond_transformer import Net2NetTransformer

from cmota.loader import StoryTextImageDataModule
from cmota.modules.dalle.tokenizer import HugTokenizer, YttmTokenizer #tokenizer
from cmota.modules.dalle.tokenizer_mindalle import build_tokenizer

from torchvision import transforms as T


import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

if __name__ == "__main__":

    # argument parsing
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


    parser = argparse.ArgumentParser(description='DALL-E Training for Pytorch GPU')

    # Added configuration for recurrent transformer
    parser.add_argument('--n_memory_cells', type=int, default=1, help='n_memory_cells')  
    parser.add_argument('--load_images', action='store_true', default=False, help='load images in ram') 
    parser.add_argument('--memory', action='store_true', default=False, help='use memory') 
    parser.add_argument('--gru_updater', action='store_true', default=False, help='gru memory updater')
    
    parser.add_argument('--special_case_img_seq_len', action='store_true', default=False, help='64 image size to 16x16 token size')
    parser.add_argument('--bidirection', action='store_true', default=False, help='Whether or not to use bi-directional training')
    parser.add_argument('--last_layer_memory', action='store_true', default=False, help='Last layer memory update')
    parser.add_argument('--first_layer_memory', action='store_true', default=False, help='First layer memory update')
    parser.add_argument('--nth_markov', action='store_true', default=False, help='N-th order memory propagation')
    parser.add_argument('--nth_markov_w_mlp', action='store_true', default=False, help='N-th order memory propagation w/ mlp layer')

    parser.add_argument('--backward_flow', action='store_true', default=False, help='backward flow generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='backward flow generation')

    #path configuration
    parser.add_argument('--data_dir', type=str, default='dataset/ducogan_pororo', help='path to train/val dataset')            
    parser.add_argument('--log_dir', type=str, default='results/',
                    help='path to save logs')

    parser.add_argument('--vae_path', type=str,
                   help='path to your trained VAE')

    parser.add_argument('--bpe_path', type=str, 
                    help='path to your BPE json file')

    parser.add_argument('--backup_dir', type=str, default='backups/',
                    help='path to save backups for sudden crash') 

    parser.add_argument('--ckpt_path', type=str,default='results/checkpoints/last.ckpt',
                    help='path to previous checkpoint') 

    parser.add_argument('--gpu_dist', action='store_true', default=False,
                    help='distributed training with gpus')

    parser.add_argument('--infer_mode', type=str, help='val or test', default='test') 

    parser.add_argument('--infer_name', type=str, help='val or test', default='pororo_128_tmp') 

    #training configuration
    parser.add_argument('--backup', action='store_true', default=False,
                    help='save backup and load from backup if restart happens')      
    parser.add_argument('--backup_steps', type =int, default = 1000,
                    help='saves backup every n training steps')             
    parser.add_argument('--image_log_steps', type=int, default=1000,
                    help='log image outputs for every n step.')   
    parser.add_argument('--refresh_rate', type=int, default=1,
                    help='progress bar refresh rate')    
    parser.add_argument('--precision', type=int, default=32,
                    help='precision for training')                     
    parser.add_argument('--fake_data', action='store_true', default=False,
                    help='using fake_data for debugging')                                                              
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
    parser.add_argument('--num_workers', type=int, default=64,
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
    parser.add_argument('--web_dataset',action='store_true', default=False,
                    help='enable web_dataset') 
    parser.add_argument('--wds_keys', type=str, default='img,cap',
                    help='web_dataset keys')   
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
    parser.add_argument('--p_loss_weight', type = float, default=0.1,
                    help = 'Perceptual loss weight')   

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
    args.world_size = args.gpus

    args.base_lr = args.learning_rate              
    args.learning_rate = args.learning_rate * args.world_size * args.batch_size
    

    # tokenizer
    if exists(args.bpe_path):
        klass = HugTokenizer if args.hug else YttmTokenizer
        tokenizer = klass(args.bpe_path)  
    #args.num_text_tokens = tokenizer.vocab_size
    tokenizer = build_tokenizer('tokenizer', context_length=80, lowercase=True, dropout=None)
    # pororo
    tokenizer.add_tokens(['pororo', 'loopy', 'eddy', 'harry', 'poby', 'tongtong', 'crong', 'rody', 'petty'])
    args.num_text_tokens = tokenizer.get_vocab_size()
    
    print(f'Using BPE model with vocab size: {args.num_text_tokens}')
    default_root_dir = args.log_dir

    limit_train_batches = 1.0
    limit_test_batches = 1.0   

    # model (EVQVAE)
    vae = EMAVQVAE.load_from_checkpoint(args.vae_path)      
    vae.setup_eval()                       


    print(f'Loaded VAE with codebook size: {vae.num_tokens}')

    model = Net2NetTransformer(args=args, batch_size=args.batch_size, learning_rate=args.learning_rate, vae=vae, keep_prob = args.keep_prob, cond_stage_key='text', special_case_img_seq_len=args.special_case_img_seq_len, infer_name=args.infer_name)
    model = model.load_from_checkpoint(vae=vae, checkpoint_path=args.ckpt_path, infer_name=args.infer_name)

    datamodule = StoryTextImageDataModule(args.data_dir, 
                                        args.batch_size, args.num_workers, 
                                        args.img_size, args.text_seq_len, 
                                        args.resize_ratio,args.truncate_captions,
                                        tokenizer, 
                                        dataset_size = args.dataset_size)   

    trainer = Trainer(gpus= gpus, default_root_dir=default_root_dir,
                      max_epochs=args.epochs, progress_bar_refresh_rate=args.refresh_rate,precision=args.precision,
                      accelerator='ddp', accumulate_grad_batches=args.ga_steps,
                      gradient_clip_val=args.clip_grad_norm,
                      num_sanity_val_steps=args.num_sanity_val_steps,
                      limit_train_batches=limit_train_batches,limit_test_batches=limit_test_batches,                          
                      resume_from_checkpoint = args.ckpt_path)


    trainer.test(model, datamodule=datamodule)
