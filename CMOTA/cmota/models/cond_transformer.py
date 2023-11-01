import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from cmota.modules.transformer.mingpt import RecurrentGPT
#from cmota.modules.dalle.tokenizer import tokenizer
from cmota.modules.dalle.tokenizer_mindalle import build_tokenizer
from cmota.utils.util import CLSProvider

from torchvision.utils import save_image

import copy

class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                args, batch_size, learning_rate,
                vae, cond_vae = None,
                ignore_keys=[],
                first_stage_key='image',
                cond_stage_key='text',
                keep_prob=1.0,
                sos_token=0,
                special_case_img_seq_len=True,
                epoch=150,
                infer_name='pororo_128_tmp',
                top_p=0.9
                 ):
        super().__init__()
        self.save_hyperparameters('args','batch_size','learning_rate')
        self.args = args
        self.be_unconditional = (cond_stage_key == 'unconditional')
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.first_stage_model = vae

        self.infer_name = infer_name
        self.top_p = top_p

        self.epoch = epoch
        self.char_idx = [16384,16385,16386,16387,16388,16389,16390,16391,16392] # For Pororo character index 

        self.tokenizer = build_tokenizer('tokenizer', context_length=80, lowercase=True, dropout=None)
        self.tokenizer.add_tokens(['pororo', 'loopy', 'eddy', 'harry', 'poby', 'tongtong', 'crong', 'rody', 'petty'])
        # [16384,16385,16386,16387,16388,16389,16390,16391,16392] pororo, loopy, eddy, harry, poby, tongtong, crong, rody, petty

        self.num_first_stage_tokens = self.first_stage_model.args.num_tokens # 1024
        self.first_stage_seq_len = 256 # token size 16 x 16
        image_seq_len = 256

            
        self.loss_img_weight = args.loss_img_weight
        self.output_imgs_dir = args.output_imgs_dir
        self.memory = args.memory

        self.global_val = 0
        self.batch_size = args.batch_size

        # if self.cond_stage_key == 'text':
        # [SOC]: self.args.num_text_tokens
        # [SOI]: self.num_first_stage_tokens 
        # [EOI]: self.num_first_stage_tokens + 1
        self.num_cond_stage_tokens = self.args.num_text_tokens + 1 # [SOC] # 16384 + 1
        self.cond_stage_seq_len = args.text_seq_len + 1 # [SOC]  # 80 + 1

        self.num_first_stage_tokens = self.num_first_stage_tokens + 2 # [SOI] + [EOI] # 1024 + 2
        self.first_stage_seq_len = self.first_stage_seq_len + 2 # [SOI] + [EOI] # 256 + 2
        
        self.cond_stage_vocab_size = self.num_cond_stage_tokens    # 16385
        self.first_stage_vocab_size = self.num_first_stage_tokens  # 1026
        self.vocab_size = self.cond_stage_vocab_size + self.first_stage_vocab_size # 49408 + 1026 = 50434 ==> 50435 // 16385 + 1026 ==> 17411
        self.block_size = self.first_stage_seq_len + self.cond_stage_seq_len # 256 + 80 + 3
        self.hidden_dim = args.hidden_dim
        self.depth = args.depth
        self.heads = args.heads
        self.soc_token = self.cond_stage_vocab_size - 2        
        self.soi_token = self.first_stage_vocab_size - 2
        self.eoi_token = self.first_stage_vocab_size - 1

        self.transformer = RecurrentGPT(vocab_size = self.vocab_size,                             
                                        block_size = self.block_size, 
                                        n_layer=self.depth, n_head=self.heads, 
                                        n_embd = args.hidden_dim, batch_size=self.batch_size,
                                        hybrid_mask=self.args.hybrid_mask,
                                        text_seq_length=args.text_seq_len, img_tok_seq_length=image_seq_len,
                                        n_memory_cells=args.n_memory_cells)          


        self.keep_prob = keep_prob
        
        self.train_total_loss = 0.0
        self.val_total_loss = 0.0
        self.train_cond2img = 0.0
        self.train_img2cond = 0.0
        self.val_cond2img = 0.0
        self.val_img2cond = 0.0


    def forward_step(self, prev_ms, text_features, img_features, img2cond, cached_memory=None):
        c_indices = self.encode_to_c(text_features)
        z_indices = self.encode_to_z(img_features) # [B, 258] soi + 256 + eoi

        if self.training and self.keep_prob < 1.0:
            mask = torch.bernoulli(self.keep_prob*torch.ones(z_indices.shape, device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)

            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size-1)
            z_indices = mask*z_indices+(1-mask)*r_indices

        if img2cond:
            # Image2Text Generation
            zc_indices = torch.cat((z_indices[:,:-1], c_indices), dim=1)
            z_seg = torch.zeros_like(z_indices[:,:-1])
            c_seg = torch.ones_like(c_indices)
            zc_seg = torch.cat((z_seg, c_seg), dim=1)

            #prev_ms, logits = self.transformer(zc_indices[:,:-1], prev_ms=prev_ms, seg=zc_seg[:,:-1], gen_type='i2t')
            prev_ms, logits = self.transformer(zc_indices, prev_ms=prev_ms, seg=zc_seg, cached_memory=cached_memory, gen_type='i2t')
            logits = rearrange(logits, 'b n c -> b c n')

            # Loss calculation of image and text modality
            image_logits = logits[:,:,:self.first_stage_seq_len-1].contiguous()
            image_target = z_indices[:,1:].contiguous()
            image_loss = F.cross_entropy(image_logits, image_target)

            cond_logits = logits[:,:,self.first_stage_seq_len-1:-1].contiguous()
            cond_target = c_indices[:,1:].contiguous()
            cond_loss = F.cross_entropy(cond_logits, cond_target)

        else:
            # Text2Image Generation
            cz_indices = torch.cat((c_indices, z_indices), dim=1)
            c_seg = torch.zeros_like(c_indices)
            z_seg = torch.ones_like(z_indices)
            cz_seg = torch.cat((c_seg, z_seg), dim=1)

            prev_ms, logits = self.transformer(cz_indices[:,:-1], prev_ms=prev_ms, seg=cz_seg[:,:-1], cached_memory=cached_memory, gen_type='t2i') # cz_indices[:,:-1] -> [B, 81+257]  --> logits [B, 50435, 81+257]
            logits = rearrange(logits, 'b n c -> b c n')

            # Loss calculation of image and text modality
                
            # Bidirectional Training
            image_logits = logits[:,:,self.cond_stage_seq_len:].contiguous() # [B, 257]
            image_target = z_indices[:,1:] # [B, 257]
            image_loss = F.cross_entropy(image_logits, image_target)

            cond_logits = logits[:,:,:self.cond_stage_seq_len-1].contiguous()
            cond_target = c_indices[:,1:self.cond_stage_seq_len].contiguous()
            cond_loss = F.cross_entropy(cond_logits, cond_target)
            

        loss = (cond_loss + self.loss_img_weight*image_loss) / (self.loss_img_weight+1)


        if img2cond:
            image_logits = image_logits[:,:,1:]
            image_logits = torch.argmax(image_logits, dim=1)
            image_target = image_target[:,1:]
        else:
            image_logits = image_logits[:,:,:-1]
            image_logits = torch.argmax(image_logits, dim=1)
            image_target = image_target[:,:-1]

        return prev_ms, loss, cond_loss, image_loss, image_logits, image_target


    def forward(self, story_batch, pseudo_sent_batch=False, img2cond=False):
        prev_ms = [None] * self.depth
        memory_list = []
        image_logit_list = []
        image_target_list = []
        total_loss = 0
        cond_loss = 0
        image_loss = 0

        for idx, paired_data in enumerate(story_batch):

            if pseudo_sent_batch is not False:
                text_batch = pseudo_sent_batch[idx]
            else:
                text_batch = paired_data['tokenized_text']

            if idx >= 2:
                prev_ms, tmp_loss, tmp_cond_loss, tmp_img_loss, img_logits, img_target = self.forward_step(prev_ms, text_batch, paired_data['image'], img2cond, memory_list)
            else:
                prev_ms, tmp_loss, tmp_cond_loss, tmp_img_loss, img_logits, img_target = self.forward_step(prev_ms, text_batch, paired_data['image'], img2cond)

            image_logit_list.append(img_logits)
            image_target_list.append(img_target)
            memory_list.append([x.detach() if x != None else None for x in prev_ms])
            total_loss += tmp_loss
            cond_loss  += tmp_cond_loss
            image_loss += tmp_img_loss


        return total_loss/5, cond_loss/5, image_loss/5, image_logit_list, image_target_list

        
    def top_k_logits(self, logits, thres = 0.5):
        num_logits = logits.shape[-1]
        k = max(int((1 - thres) * num_logits), 1)
        val, ind = torch.topk(logits, k)
        probs = torch.full_like(logits, float('-inf'))
        probs.scatter_(1, ind, val)
        return probs

    # Borrowed from hugging face TopPLogitsWarper
    def top_p_logits(self, logits, top_p=0.9, min_tokens_to_keep=1):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits


    @torch.no_grad()
    def sample(self, c, x, prev_ms, steps, temperature=1.0, sample=False, threshold=None, top_p=None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training

        for k in range(steps):
            assert x.size(1) <= block_size # make sure model can see conditioning
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed

            c_seg = torch.zeros_like(c)
            z_seg = torch.ones_like(x_cond[:, c.shape[1]:])
            cz_seg = torch.cat((c_seg,z_seg), dim=1)

            tmp_prev_ms, logits = self.transformer(x_cond, prev_ms, cz_seg) # x_cond shape [B, 130], cz_seg shape [B, 130]
              
            logits = logits[:, -1, :] # logits [B, 130, 50435] -> [B, 50435]
            # optionally crop probabilities to only the top k options
            if threshold is not None:
               logits = self.top_k_logits(logits, threshold)
            if top_p is not None:
               logits = self.top_p_logits(logits, top_p=top_p)
            
            # apply softmax to convert to probabilities
            probs = F.softmax(logits / temperature, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)            
            ix = torch.clamp(ix, min = 0.0, max=self.first_stage_vocab_size-1)                             
            # append to the sequence and continue                
            x = torch.cat((x, ix), dim=1)

        # cut off conditioning 
        x = x[:, c.shape[1]-1:]
        return x, tmp_prev_ms


    @torch.no_grad()
    def sample_cond(self, c, x, prev_ms, steps, temperature=1.0, sample=False, threshold=None, top_p=None):
        #make sure to place image first(c) and text last (x)
        c = c[:,:-1]
        x = torch.cat((c, x),dim=1) # 258, 259, ...
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training

        for k in range(steps):
            assert x.size(1) <= block_size # make sure model can see conditioning
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed

            c_seg = torch.zeros_like(c) # 257, 
            z_seg = torch.ones_like(x_cond[:, c.shape[1]:])
            cz_seg = torch.cat((c_seg,z_seg), dim=1)    # should be 258 (257 + 1), 259 (257 + 2), ... 

            tmp_prev_ms, logits = self.transformer(x_cond, prev_ms, cz_seg, gen_type='i2t')  
            
            logits = logits[:, -1, :] 
            # optionally crop probabilities to only the top k options
            if threshold is not None:
                logits = self.top_k_logits(logits, threshold)
            if top_p is not None:
                logits = self.top_p_logits(logits, top_p)

            # apply softmax to convert to probabilities
            probs = F.softmax(logits / temperature, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)                          
            ix = torch.clamp(ix, min = self.num_first_stage_tokens, max=self.vocab_size-2)                             
            # append to the sequence and continue                
            x = torch.cat((x, ix), dim=1)
        # cut off conditioning
        x = x[:, c.shape[1]+1:]
        # shift idxs to cond range
        x = x - self.num_first_stage_tokens  
        return x, tmp_prev_ms


    @torch.no_grad()
    def encode_to_z(self, x):
        indices = self.first_stage_model.get_codebook_indices(x)    
        
        indices = F.pad(indices, (1, 0), value = self.soi_token)
        indices = F.pad(indices, (0, 1), value = self.eoi_token)        
        return indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.cond_stage_key == 'text':  
            indices = c
        else:
            indices = self.cond_stage_model.get_codebook_indices(c)
        #add [SOC] 
        indices = F.pad(indices, (1, 0), value = self.soc_token)
        indices = indices + self.first_stage_vocab_size
        return indices

    @torch.no_grad()
    def decode_to_img(self, indices):
        #clamp ix into image token's range            
        indices = torch.clamp(indices, min = 0.0, max=self.first_stage_vocab_size-3)     
        # indices shape: [B, T]  token indexes     
        x = self.first_stage_model.decode(indices, feed_seq=True)
        
        return x


    @torch.no_grad()
    def storyfile_gen(self, batch, logits_list, target_list):

        tokenizer = build_tokenizer('tokenizer', context_length=80, lowercase=True, dropout=None)
        tokenizer.add_tokens(['pororo', 'loopy', 'eddy', 'harry', 'poby', 'tongtong', 'crong', 'rody', 'petty'])

        txt_ori_list = []
        img_ori_list = []
        img_tgt_list = []
        img_gen_list = []

        prev_ms = [None] * self.depth

        text_list = []

        for i, story in tqdm(enumerate(batch)):
            text, x = story['tokenized_text'], story['image']
            data_id = story['idx'].cpu().numpy()

            
            tmp_text_list = []
            for j in range(len(text)):
                sent = text[j]
                #tmp_token_list = sent.masked_select(sent != 0).tolist()
                tmp_decoded_text = tokenizer.decode(sent.tolist(), skip_special_tokens=True)
                tmp_decoded_text = self.character_name_change(tmp_decoded_text)
                tmp_text_list.append(tmp_decoded_text)
            tmp = copy.deepcopy(tmp_text_list)
            text_list.append(tmp)

            text = text.to(self.device)
            x = x.to(self.device)  

            #logits = logits_list[i]
            target = target_list [i] 

            self.eval()
            
            z_indices = self.encode_to_z(x)
            c_indices = self.encode_to_c(text)

            #target sample
            x_target = self.decode_to_img(target)

            #generate sample
            z_start_indices = z_indices[:,:1]
            full_steps = z_indices.shape[1] - 2    
            x_gen_idx, prev_ms = self.sample(c = c_indices, x=z_start_indices, prev_ms=prev_ms,
                                                    steps=full_steps, 
                                                    sample=True,
                                                    top_p=self.top_p,
                                                    threshold=0.9)
                                                            
            x_gen = self.decode_to_img(x_gen_idx[:,1:-1])

            self.train()

            img_ori_list.append(x.unsqueeze(1))
            img_tgt_list.append(x_target.unsqueeze(1))
            img_gen_list.append(x_gen.unsqueeze(1))

        for i in range(x.size()[0]):
            tmp_list = []
            for j in range(5):
                tmp_list.append(text_list[j][i] + " | sep | ")
            tmp_list.append("| end |  ")
            txt_ori_list.append(''.join(tmp_list))

        x = torch.cat(img_ori_list, dim=1).view(-1,3,self.args.img_size,self.args.img_size) 
        x_target = torch.cat(img_tgt_list, dim=1).view(-1,3,self.args.img_size,self.args.img_size)
        x_gen = torch.cat(img_gen_list, dim=1).view(-1,3,self.args.img_size,self.args.img_size)

        print("DataID: ", data_id)

        saved_path_prefix_ori = 'output_imgs/' + self.infer_name + '/'
        saved_path_prefix_gen = 'output_imgs/' + self.infer_name + '/'

        if os.path.isdir(saved_path_prefix_ori) == False:
            os.makedirs(saved_path_prefix_ori, exist_ok=True)

        for i, (x_1, x_gen_1) in enumerate(zip(x, x_gen)):
            tmp_saved_path_prefix_ori = saved_path_prefix_ori + 'x_ori_' + str(data_id[int(i//5)]) + '_' + str(int(i%5)) + '.png'
            tmp_saved_path_prefix_gen = saved_path_prefix_gen + 'x_gen_' + str(data_id[int(i//5)]) + '_' + str(int(i%5)) + '.png'
            save_image(tensor=x_1, fp=tmp_saved_path_prefix_ori, normalize=True)
            save_image(tensor=x_gen_1, fp=tmp_saved_path_prefix_gen, normalize=True)
            

    def character_name_change(self, sent):
        for char in ['pororo', 'loopy', 'eddy', 'harry', 'poby', 'tongtong', 'crong', 'rody', 'petty']:
            if char in sent:
                replace_char = char + ' '
                sent = sent.replace(char, replace_char)
        
        return sent

    @torch.no_grad()
    def seq_img_to_para(self, batch):

        i2t_full_list = []

        prev_ms_sent = [None] * self.depth

        ori_text_list = []

        for i, story in enumerate(batch):
            text, x = story['tokenized_text'], story['image']
            #data_id = story['idx'].cpu().numpy()

            tmp_text_list = []
            for j in range(len(text)):
                sent = text[j]
                tmp_decoded_text = self.tokenizer.decode(sent.tolist(), skip_special_tokens=True)
                tmp_decoded_text = self.character_name_change(tmp_decoded_text)
                tmp_text_list.append(tmp_decoded_text)
            tmp = copy.deepcopy(tmp_text_list)
            ori_text_list.append(tmp)

            text = text.to(self.device)
            x = x.to(self.device) 

            self.eval()
            
            z_indices = self.encode_to_z(x)
            c_indices = self.encode_to_c(text)

            #generate caption sample
            c_start_indices = c_indices[:,:1]
            full_steps = c_indices.shape[1] - 1
            c_gen_idx, prev_ms_sent = self.sample_cond(c = z_indices, x=c_start_indices, prev_ms=prev_ms_sent,
                                                    steps=full_steps,
                                                    sample=True,
                                                    threshold=0.9,
                                                    top_p=0.9)

            c_gen_tensor = []
            for c_idx in c_gen_idx:
                c_idx = F.pad(c_idx.masked_select(c_idx != 0), (0, 80-len(c_idx.masked_select(c_idx != 0))), "constant", 0)
                c_gen_tensor.append(c_idx)
            c_gen_tensor = torch.stack(c_gen_tensor)

            self.train()

            i2t_full_list.append(c_gen_tensor)

        return torch.stack(i2t_full_list)

    # Pseudo-setence filtering
    def determine_to_use_pseudo_sent(self, pseudo_sent, gt_sent):
        # one sentence 
        if gt_sent.tolist() not in self.char_idx:
            return gt_sent.tolist()
        
        chars_in_gt = [x for x in self.char_idx if x in gt_sent]
        count = 0
        for char_gt in chars_in_gt:
            for char_pseudo in pseudo_sent:
                if char_gt == char_pseudo:
                    count += 1

        if float(count/len(chars_in_gt)) >= 0.5:
            return pseudo_sent.tolist()
        else:
            return gt_sent.tolist()
        
        
    # Training, Validation, Test steps
    def training_step(self, story_batch, batch_idx):

        # Bidirectional training
        img2cond = bool(batch_idx % 2)
        story_sent_batch = False
        if not img2cond:
            p_pt = torch.rand(1)
            thresh = 0.3
            
            # Pseudo-sentence generation
            if self.args.pseudo_i2t and thresh > p_pt:
                with torch.no_grad():
                    story_sent_batch = self.seq_img_to_para(story_batch)

                    if self.args.pseudo_filtering:
                        ####### Pseudo-sentence filtering #######
                        filtered_sent_batch = []
                        for pseudo_sents, story in zip(story_sent_batch, story_batch):
                            gt_sents = story['tokenized_text']
                        
                            tmp_sents = []
                            for pseudo_sent, gt_sent in zip(pseudo_sents, gt_sents):
                                tmp_sents.append(self.determine_to_use_pseudo_sent(pseudo_sent, gt_sent))

                            filtered_sent_batch.append(copy.deepcopy(tmp_sents))

                        story_sent_batch = torch.Tensor(filtered_sent_batch).type(torch.int64).to(story_sent_batch.device)

        loss, cond_loss, image_loss, logits_list, target_list = self(story_batch, pseudo_sent_batch=story_sent_batch, img2cond=img2cond)
           

        if img2cond:
            self.log("train/img2cond/loss", loss, prog_bar=True, logger=True) 
            self.log("train/img2cond/cond_loss", cond_loss, prog_bar=True, logger=True) 
            self.log("train/img2cond/image_loss", image_loss, prog_bar=True, logger=True) 
            self.train_img2cond = loss

        else:
            self.log("train/cond2img/loss", loss, prog_bar=True, logger=True) 
            self.log("train/cond2img/cond_loss", cond_loss, prog_bar=True, logger=True) 
            self.log("train/cond2img/image_loss", image_loss, prog_bar=True, logger=True) 
            self.train_cond2img = loss

        self.train_total_loss = self.train_img2cond + self.train_cond2img
        self.log("train/total_loss", self.train_total_loss, prog_bar=False, logger=True)

        return {'loss':loss, 'logits':[logits.detach() for logits in logits_list], 'target':[target.detach() for target in target_list]} 



    def validation_step(self, story_batch, batch_idx):

        img2cond = bool(batch_idx % 2)

        loss, cond_loss, image_loss, logits_list, target_list = self(story_batch, img2cond=img2cond) 


        if img2cond:
            self.log("val/img2cond/loss", loss, prog_bar=True, logger=True) 
            self.log("val/img2cond/cond_loss", cond_loss, prog_bar=True, logger=True) 
            self.log("val/img2cond/image_loss", image_loss, prog_bar=True, logger=True)
            self.val_img2cond = loss

        else:
            self.log("val/cond2img/loss", loss, prog_bar=True, logger=True) 
            self.log("val/cond2img/cond_loss", cond_loss, prog_bar=True, logger=True) 
            self.log("val/cond2img/image_loss", image_loss, prog_bar=True, logger=True)   
            self.val_cond2img = loss

        self.val_total_loss = self.val_img2cond + self.val_cond2img
        self.log("val/total_loss", self.val_total_loss, prog_bar=False, logger=True)

        return {'loss':loss, 'logits':[logits.detach() for logits in logits_list], 'target':[target.detach() for target in target_list]}    
    

    def test_step(self, story_batch, batch_idx):
        _, _, _, logits_list, target_list = self(story_batch)
        self.storyfile_gen(story_batch, logits_list, target_list)


    # Configuration on optimizers
    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # last_layer_memory and not first_layer_memory:
        decay_tmp = 'blocks.' + str(int(self.depth)-1) + '.memory_initializer.init_memory_fc.1.weight'
        decay.add(decay_tmp)
    
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.learning_rate, betas=(0.9, 0.95))
        if self.args.lr_decay:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                cooldown=10,
                min_lr=1e-6,
                verbose=True,
                )    
            sched = {'scheduler':scheduler, 'monitor':'train/total_loss'}            
            return [optimizer], [sched]
        else:
            return [optimizer], [] 