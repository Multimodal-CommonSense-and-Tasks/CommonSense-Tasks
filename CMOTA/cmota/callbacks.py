#borrowed from https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/callbacks/vision/image_generation.py#L15-L97
from typing import Any, Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer

from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import wandb

import copy


class ReconstructedImageLogger(Callback):
    def __init__(
        self,
        every_n_steps: int = 1000,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        use_wandb: bool = False,
        multi_optim = False,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``True``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.multi_optim = multi_optim
        self.use_wandb = use_wandb

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        if trainer.global_step % self.every_n_steps == 0:
            if self.multi_optim:
                x = outputs[0]['x']
                xrec = outputs[0]['xrec']
            else:
                x = outputs['x']
                xrec = outputs['xrec']

            x_grid = torchvision.utils.make_grid(
                    tensor=x,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )           
            xrec_grid = torchvision.utils.make_grid(
                    tensor=xrec,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )  
            if self.use_wandb:
                trainer.logger.experiment.log({
                "train/input": wandb.Image(x_grid),
                "train/reconstruction": wandb.Image(xrec_grid),                
                "global_step": trainer.global_step
            })
            else:  
                x_title = "train/input"
                trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
                xrec_title = "train/reconstruction"
                trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""
        x = outputs['x']
        xrec = outputs['xrec']
        x_grid = torchvision.utils.make_grid(
                    tensor=x,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
            )           
        xrec_grid = torchvision.utils.make_grid(
                    tensor=xrec,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
            )  
        if self.use_wandb:
            trainer.logger.experiment.log({
            "val/input": wandb.Image(x_grid),
            "val/reconstruction": wandb.Image(xrec_grid),                
            "global_step": trainer.global_step
        })
        else:  
            x_title = "val/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            xrec_title = "val/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)



class DalleGenerativeStorySampler(Callback):
    
    def __init__(
        self,
        every_n_steps: int = 1000,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        tokenizer = None,
        top_p = None,
        img_size=128,
        depth=8
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``True``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """

        super().__init__()
        self.every_n_steps = every_n_steps
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.tokenizer = tokenizer
        self.top_p = top_p
        self.img_size = img_size
        self.depth = depth

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        if trainer.global_step % self.every_n_steps == 0:          
            print("training batch end")

            self.nrow = 5
            txt_ori_list = []
            img_ori_list = []
            img_tgt_list = []
            img_rec_list = []
            img_gen_list = []
            txt_list = []
            txt_full_list = []

            # prev_ms
            prev_ms_img = [None] * self.depth * 2
            prev_ms_sent = [None] * self.depth * 2
            
            text_list = []
            for i, story in enumerate(batch):
                text, x = story['tokenized_text'], story['image']

                tmp_text_list = []
                for j in range(len(text)):
                    sent = text[j]
                    tmp_decoded_text = self.tokenizer.decode(sent.tolist(), skip_special_tokens=True)
                    tmp_text_list.append(tmp_decoded_text)
                tmp = copy.deepcopy(tmp_text_list)
                text_list.append(tmp)
      
                text = text.to(pl_module.device)
                x = x.to(pl_module.device)    
                logits = outputs['logits'][i]
                target = outputs['target'][i] 

                with torch.no_grad():
                    pl_module.eval()
                    
                    z_indices = pl_module.encode_to_z(x)
                    c_indices = pl_module.encode_to_c(text)

                    #target sample
                    x_target = pl_module.decode_to_img(target)
                    #reconstruction sample                  
                    x_rec = pl_module.decode_to_img(logits)


                    #generate sample
                    z_start_indices = z_indices[:,:1]
                    full_steps = z_indices.shape[1] - 2    
                    x_gen_idx, prev_ms_img = pl_module.sample(c = c_indices, x=z_start_indices, 
                                                          prev_ms=prev_ms_img,
                                                          steps=full_steps, 
                                                          sample=True,
                                                          threshold=0.9,
                                                          top_p=self.top_p)  
                    x_gen = pl_module.decode_to_img(x_gen_idx[:,1:-1])

                    # generate captions
                    c_start_indices = c_indices[:,:1]
                    full_steps = c_indices.shape[1] - 1
                    c_gen_idx, prev_ms_sent = pl_module.sample_cond(c = z_indices, x=c_start_indices, prev_ms=prev_ms_sent,
                                        steps=full_steps, 
                                        sample=True,
                                        threshold=0.9,
                                        top_p=self.top_p) 

                    c_gen_list = []
                    for c_idx in c_gen_idx:
                        c_idx = self.tokenizer.decode(c_idx.tolist(), skip_special_tokens=True)
                        c_gen_list.append(c_idx)

                    pl_module.train()

                    c_gen_tmp = 0
                    c_gen_tmp = copy.deepcopy(c_gen_list)
                    img_ori_list.append(x.unsqueeze(1))
                    img_tgt_list.append(x_target.unsqueeze(1))
                    img_rec_list.append(x_rec.unsqueeze(1))
                    img_gen_list.append(x_gen.unsqueeze(1))
                    txt_list.append(c_gen_tmp)

            for i in range(x.size()[0]):
                tmp_list = []
                for j in range(5):
                    tmp_list.append(text_list[j][i] + " | sep | ")
                tmp_list.append("| end |  ")
                txt_ori_list.append(''.join(tmp_list))

            for i in range(len(txt_list[0])):
                tmp_list = []
                for j in range(5):
                    tmp_list.append(txt_list[j][i] + " | sep | ")
                tmp_list.append(" | end | ")
                txt_full_list.append(''.join(tmp_list))

            x = torch.cat(img_ori_list, dim=1).view(-1,3,self.img_size,self.img_size) 
            x_target = torch.cat(img_tgt_list, dim=1).view(-1,3,self.img_size,self.img_size)
            x_rec = torch.cat(img_rec_list, dim=1).view(-1,3,self.img_size,self.img_size) 
            x_gen = torch.cat(img_gen_list, dim=1).view(-1,3,self.img_size,self.img_size)

            x_grid = torchvision.utils.make_grid(
                tensor=x,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            ) 
            xtarget_grid = torchvision.utils.make_grid(
                tensor=x_target,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )                        
            xrec_grid = torchvision.utils.make_grid(
                tensor=x_rec,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )                      
            xgen_grid = torchvision.utils.make_grid(
                tensor=x_gen,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )                
            text_title = "train/text"
            txt_ori_list = ''.join(txt_ori_list)
            trainer.logger.experiment.add_text(text_title, txt_ori_list, global_step=trainer.global_step)

            text_gen_title = 'train/gen_text'
            txt_full_list = ''.join(txt_full_list)
            trainer.logger.experiment.add_text(text_gen_title, txt_full_list, global_step=trainer.global_step)

            x_title = "train/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            xtar_title = "train/target"
            trainer.logger.experiment.add_image(xtar_title, xtarget_grid, global_step=trainer.global_step)            
            xrec_title = "train/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)            
            xgen_title = "train/generation"
            trainer.logger.experiment.add_image(xgen_title, xgen_grid, global_step=trainer.global_step)
    
    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        """Called when the validation batch ends."""        
        if batch_idx == 0:          
            
            self.nrow = 5
            img_ori_list = []
            img_tgt_list = []
            img_rec_list = []
            img_gen_list = []
            txt_ori_list = []
            txt_list = []
            txt_full_list = []

            #print("validation: ", batch_idx, dataloader_idx)


            # prev_ms
            prev_ms_img = [None] * self.depth * 2
            prev_ms_sent = [None] * self.depth * 2

            text_list = []
            for i, story in enumerate(batch):
                text, x = story['tokenized_text'], story['image']
                
                tmp_text_list = []
                for j in range(len(text)):
                    sent = text[j]
                    tmp_decoded_text = self.tokenizer.decode(sent.tolist(), skip_special_tokens=True)
                    tmp_text_list.append(tmp_decoded_text)
                tmp = copy.deepcopy(tmp_text_list)
                text_list.append(tmp)

                text = text.to(pl_module.device)
                x = x.to(pl_module.device)    
                logits = outputs['logits'][i]
                target = outputs['target'][i] 

                with torch.no_grad():
                    pl_module.eval()

                    z_indices = pl_module.encode_to_z(x)
                    c_indices = pl_module.encode_to_c(text)

                    #target sample
                    x_target = pl_module.decode_to_img(target)
                    #reconstruction sample                  
                    x_rec = pl_module.decode_to_img(logits)

                    #generate sample
                    z_start_indices = z_indices[:,:1]
                    full_steps = z_indices.shape[1] - 2    
                    x_gen_idx, prev_ms_img = pl_module.sample(c = c_indices, x=z_start_indices, prev_ms=prev_ms_img,
                                                          steps=full_steps, 
                                                          sample=True,
                                                          threshold=0.9,
                                                          top_p=self.top_p)                                                                
                    x_gen = pl_module.decode_to_img(x_gen_idx[:,1:-1])

                    # generate caption
                    c_start_indices = c_indices[:,:1]
                    full_steps = c_indices.shape[1] - 1
                    c_gen_idx, prev_ms_sent = pl_module.sample_cond(c = z_indices, x=c_start_indices, prev_ms=prev_ms_sent,
                                        steps=full_steps, 
                                        sample=True,
                                        threshold=0.9,
                                        top_p=self.top_p) 

                    c_gen_list = []
                    for c_idx in c_gen_idx:
                        c_idx = self.tokenizer.decode(c_idx.tolist(), skip_special_tokens=True)
                        c_gen_list.append(c_idx)

                    pl_module.train()

                    c_gen_tmp = 0
                    c_gen_tmp = copy.deepcopy(c_gen_list)
                    img_ori_list.append(x.unsqueeze(1))
                    img_tgt_list.append(x_target.unsqueeze(1))
                    img_rec_list.append(x_rec.unsqueeze(1))
                    img_gen_list.append(x_gen.unsqueeze(1))
                    txt_list.append(c_gen_tmp)


            for i in range(x.size()[0]):
                tmp_list = []
                for j in range(5):
                    tmp_list.append(text_list[j][i] + " | sep | ")
                tmp_list.append("| end |  ")
                txt_ori_list.append(''.join(tmp_list))

            for i in range(len(txt_list[0])):
                tmp_list = []
                for j in range(5):
                    tmp_list.append(txt_list[j][i] + " | sep | ")
                tmp_list.append(" | end | ")
                txt_full_list.append(''.join(tmp_list))

            x = torch.cat(img_ori_list, dim=1).view(-1,3,self.img_size,self.img_size) 
            x_target = torch.cat(img_tgt_list, dim=1).view(-1,3,self.img_size,self.img_size)
            x_rec = torch.cat(img_rec_list, dim=1).view(-1,3,self.img_size,self.img_size) 
            x_gen = torch.cat(img_gen_list, dim=1).view(-1,3,self.img_size,self.img_size)

            x_grid = torchvision.utils.make_grid(
                tensor=x,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )           
            xrec_grid = torchvision.utils.make_grid(
                tensor=x_rec,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )  
            xtarget_grid = torchvision.utils.make_grid(
                tensor=x_target,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )                                                         
            xgen_grid = torchvision.utils.make_grid(
                tensor=x_gen,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )                
            text_title = "val/text"
            txt_ori_list = ''.join(txt_ori_list)
            trainer.logger.experiment.add_text(text_title, txt_ori_list, global_step=trainer.global_step)

            text_gen_title = 'val/gen_text'
            txt_full_list = ''.join(txt_full_list)
            trainer.logger.experiment.add_text(text_gen_title, txt_full_list, global_step=trainer.global_step)

            x_title = "val/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            xtar_title = "val/target"
            trainer.logger.experiment.add_image(xtar_title, xtarget_grid, global_step=trainer.global_step)                 
            xrec_title = "val/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)      
            xgen_title = "val/generation"
            trainer.logger.experiment.add_image(xgen_title, xgen_grid, global_step=trainer.global_step)