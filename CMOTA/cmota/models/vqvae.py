import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from cmota.modules.vqvae.vae import Encoder, Decoder
from cmota.modules.vqvae.quantize import VectorQuantizer, EMAVectorQuantizer, GumbelQuantizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import rearrange
from cmota.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from cmota.modules.losses.ldist import LogitDistributionLoss, clamp_with_grad
from cmota.modules.losses.lpips import RecPerceptualLoss

class VQVAE(pl.LightningModule):
    def __init__(self,
                 args, batch_size, learning_rate,
                 ignore_keys=[], finetuned=False, 
                 ft_attn_resolutions=None,
                 ft_loss_type=None, ft_args=None,
                 ):
        super().__init__()   
        self.image_size = args.resolution
        self.num_tokens = args.num_tokens

        self.encoders = nn.ModuleList()
        self.pre_quants = nn.ModuleList()
        self.quantizers = nn.ModuleList()
        self.post_quants = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        self.enc_attn_resolutions = args.attn_resolutions
        self.dec_attn_resolutions = args.attn_resolutions
        self.enc_resolution = args.resolution
        self.dec_resolution = args.resolution
        self.quant_idxs = []
        encoder = Encoder(hidden_dim=args.hidden_dim, in_channels=args.in_channels, ch_mult= args.ch_mult,
                                num_res_blocks=args.num_res_blocks, 
                                dropout=args.dropout, attn_resolutions = [args.attn_resolutions[0],],
                                resolution=args.resolution, z_channels=args.z_channels,
                                double_z=args.double_z, use_attn=args.use_attn)
        self.encoders.append(encoder)

        pre_quant = torch.nn.Conv2d(args.z_channels, args.codebook_dim, 1)
        self.pre_quants.append(pre_quant) 

        quantizer = VectorQuantizer(args.num_tokens, args.codebook_dim, beta=args.quant_beta)
        self.quantizers.append(quantizer)

        post_quant = torch.nn.Conv2d(args.codebook_dim, args.z_channels, 1)  
        self.post_quants.append(post_quant)

        self.quant_idxs.append(0)

        dec_in_channels = args.z_channels
        for i, res in enumerate(args.attn_resolutions[1:]):
            encoder = Encoder(hidden_dim=args.hidden_dim, in_channels=args.z_channels, ch_mult= [2, 4],
                                num_res_blocks=args.num_res_blocks, 
                                dropout=args.dropout, attn_resolutions = [res,],
                                resolution=args.attn_resolutions[i], z_channels=args.z_channels,
                                double_z=args.double_z, use_attn=args.use_attn)
            self.encoders.append(encoder)
        
            pre_quant = torch.nn.Conv2d(args.z_channels, args.codebook_dim, 1)
            self.pre_quants.append(pre_quant) 

            quantizer = VectorQuantizer(args.num_tokens, args.codebook_dim, beta=args.quant_beta)
            self.quantizers.append(quantizer)

            post_quant = torch.nn.Conv2d(args.codebook_dim, args.z_channels, 1)  
            self.post_quants.append(post_quant)
            
            self.quant_idxs.append(i+1)

            decoder = Decoder(hidden_dim=args.hidden_dim, out_channels = args.z_channels, ch_mult= [2, 4],
                                num_res_blocks=args.num_res_blocks, 
                                dropout=args.dropout, in_channels = args.z_channels, attn_resolutions = [args.attn_resolutions[-(i+1)],],
                                resolution=args.attn_resolutions[-(i+2)], z_channels=dec_in_channels, use_attn=args.use_attn)
            self.decoders.append(decoder) 

            #double decoder in_channels after first decoder init          
            if i == 0:
                dec_in_channels = args.z_channels * 2   

        if args.loss_type in ['dist_l1', 'dist_l2']:
            #assure out_channel for decoder is 6 when using ldist loss
            args.out_channels = 6
        else:        
            args.out_channels = 3

        decoder = Decoder(hidden_dim=args.hidden_dim, out_channels=args.out_channels, ch_mult= args.ch_mult,
                                num_res_blocks=args.num_res_blocks, 
                                dropout=args.dropout, in_channels=args.in_channels, attn_resolutions = [args.attn_resolutions[0],],
                                resolution=args.resolution, z_channels=dec_in_channels, use_attn=args.use_attn)
        self.decoders.append(decoder)

        self.setup_loss(args)

        self.save_hyperparameters("args", "batch_size", "learning_rate")
        self.args = args  
        self.image_seq_len = 0
        for i in self.quant_idxs:
            self.image_seq_len =self.image_seq_len + self.enc_attn_resolutions[i] ** 2
        if finetuned:
            self.setup_finetune(ft_attn_resolutions, ft_loss_type, ft_args)
        if args.model == 'vqvae':
            self.quant_weight_share()

    def setup_loss(self, args):
        if args.loss_type == 'gan':
            self.loss = VQLPIPSWithDiscriminator(disc_start=args.disc_start, codebook_weight=args.codebook_weight,
                                            disc_in_channels=args.disc_in_channels,disc_weight=args.disc_weight)
        elif args.loss_type == 'smooth_l1':
            self.loss = nn.SmoothL1Loss()

        elif args.loss_type == 'l1':
            self.loss = nn.L1Loss()
        
        elif args.loss_type == 'mse':
            self.loss = nn.MSELoss()
        
        elif args.loss_type == 'dist_l1':
            self.loss = LogitDistributionLoss('laplace')

        elif args.loss_type == 'dist_l2':
            self.loss = LogitDistributionLoss('normal')

        elif args.loss_type in ['lpips_l1', 'lpips_l2']:
            self.loss = RecPerceptualLoss(loss_type = args.loss_type, perceptual_weight=args.p_loss_weight)
        else:
            print(f"Loss type {args.loss_type} is not currently supported. Using default MSELoss.")
            self.loss = nn.MSELoss()            

    def quant_weight_share(self):
        for i in range(len(self.pre_quants[1:])):
            self.pre_quants[i+1].weight = self.pre_quants[0].weight
        for i in range(len(self.quantizers[1:])):
            self.quantizers[i+1].embedding.weight = self.quantizers[0].embedding.weight              
        for i in range(len(self.post_quants[1:])):
            self.post_quants[i+1].weight = self.post_quants[0].weight  

    def setup_eval(self):
        self.freeze()
        for quantizer in self.quantizers:
            quantizer.embedding.update = False
        del self.loss

    def setup_finetune(self, attn_resolutions, loss_type, args):
        self.args.finetune = True
        self.hparams.learning_rate = args.learning_rate
        self.hparams.batch_size = args.batch_size

        self.quant_idxs = []
        
        for i, attn in enumerate(self.enc_attn_resolutions):
            if attn in attn_resolutions:
                self.quant_idxs.append(i)

        self.image_seq_len = 0
        for i in self.quant_idxs:
            self.image_seq_len = self.image_seq_len + self.enc_attn_resolutions[i] ** 2
        
        self.enc_attn_resolutions = attn_resolutions
        self.dec_attn_resolutions = attn_resolutions 

        del self.decoders[:len(self.decoders) - self.quant_idxs[-1] - 1]
        del self.encoders[self.quant_idxs[-1]+1:]

        self.args.loss_type = loss_type
        self.setup_loss(args)

        self.connectors = nn.ModuleList()
        for i in range(len(self.decoders)):
            conn = torch.nn.Conv2d(self.args.z_channels, self.args.z_channels * 2, 1)
            self.connectors.append(conn)

    def on_post_move_to_device(self):
        # Weights shared after the model has been moved to TPU Device
        self.quant_weight_share()   

    def encode(self, input):
        quants = []
        encoding_indices = []
        emb_loss = None
        enc_idxs = None
        enc = input

        for i, encoder in enumerate(self.encoders):
            enc = encoder(enc)
            if self.args.finetune and i == len(self.encoders):
                h = self.pre_quants[i](enc)               
                quant, loss, info = self.quantizers[i](h)
                quants.append(quant)
                encoding_indices.append(info[2])
                emb_loss = loss
                enc_idxs = info[2]
            else:           
                h = self.pre_quants[i](enc)               
                quant, loss, info = self.quantizers[i](h)
                quants.append(quant)
                encoding_indices.append(info[2])
                if emb_loss == None:
                    emb_loss = loss
                    enc_idxs = info[2]
                else:
                    emb_loss = emb_loss + loss
                    enc_idxs = torch.cat((enc_idxs, info[2]))

        encodings = F.one_hot(enc_idxs, self.args.num_tokens).type(h.dtype)     
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))           
        info = (perplexity, encodings, encoding_indices)
                
        return quants, emb_loss, info

    def decode(self, input, feed_seq=False, return_var=False):
        quants = []
        if feed_seq:
            image_seq = input
            split_idxes = []
            for i, res in enumerate(self.enc_attn_resolutions):
                split_idxes.append(res ** 2)
                if image_seq.shape[-1] == 256:
                    split_idxes = 256
                quant_seqs = torch.split(image_seq,split_idxes, dim=1)

            for i, seq in enumerate(quant_seqs):
                z = self.quantizers[self.quant_idxs[i]].embedding(seq)               
                b, n, c = z.shape
                h = w = int(math.sqrt(n))            
                z = rearrange(z, 'b (h w) c -> b c h w', h = h, w = w)
                quants.append(z)  

        else:
            quants = input

        dec = None
        for i, decoder in enumerate(self.decoders):
            if self.args.finetune:
                if i == 0:
                    dec = self.post_quants[i](quants[-(i+1)])  

                quant = self.connectors[i](dec)
            else:
                quant = self.post_quants[i](quants[-(i+1)])
                if i != 0:
                    quant = torch.cat((quant, dec), dim=1)
            dec = decoder(quant)

        if self.args.loss_type in ['dist_l1', 'dist_l2']:
            mean, logvar = dec.chunk(2, dim=1)
            mean = clamp_with_grad(mean, -1.01, 1.01)
            logvar = clamp_with_grad(logvar, math.log(1e-5), 1.0)            
            if return_var:
                return (mean, logvar)
            else:
                return mean
        else:
            return dec

    # def decode(self, input, feed_seq=False, return_var=False):
    #     quants = []

    #     if feed_seq:
    #         image_seq = input
    #         split_idxes = []
    #         for i, res in enumerate(self.enc_attn_resolutions):
    #             split_idxes.append(res ** 2)
    #         quant_seqs = torch.split(image_seq,split_idxes, dim=1)

    #         for i, seq in enumerate(quant_seqs):
    #             z = self.quantizers[self.quant_idxs[i]].embedding(seq)               
    #             b, n, c = z.shape
    #             h = w = int(math.sqrt(n))            
    #             z = rearrange(z, 'b (h w) c -> b c h w', h = h, w = w)
    #             quants.append(z)  

    #     else:
    #         quants = input

    #     dec = None

    #     for i, decoder in enumerate(self.decoders):
    #         if self.args.finetune:
    #             if i == 0:
    #                 dec = self.post_quants[i](quants[-(i+1)])  
    #             quant = self.connectors[i](dec)
    #         else:
    #             quant = self.post_quants[i](quants[-(i+1)])
    #             if i != 0:
    #                 quant = torch.cat((quant, dec), dim=1)
    #         dec = decoder(quant)

    #     return dec

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        concat_indices = None
        _, _, [_, _, indices] = self.encode(img)

        for i in self.quant_idxs:
            idxs = indices[i]
            n = idxs.shape[0] // b
            idxs = idxs.view(b,n)     
            if concat_indices == None:
                   concat_indices = idxs
            else:
                concat_indices = torch.cat((concat_indices,idxs), dim=1) 
        return concat_indices

    def forward(self, input, return_var=False):
        quant, diff, info = self.encode(input)
 
        dec = self.decode(quant, return_var=return_var)

        return dec, diff, info[0]

    def get_trainable_params(self):
        return [params for params in self.parameters() if params.requires_grad]


    def get_last_layer(self):
        return self.decoders[-1].conv_out.weight

    def training_step(self, batch, batch_idx, optimizer_idx=0):     
        x = batch[0]
        if self.args.loss_type in ['dist_l1', 'dist_l2']:
            out, qloss, perplexity = self(x, return_var=True)
            xrec = out[0]
            logvar = out[1]
            aeloss = self.loss(xrec, logvar, x)
            loss = aeloss + qloss
            if self.args.loss_type == 'dist_l1':
                recloss = F.l1_loss(xrec, x)
            else:
                recloss = F.mse_loss(xrec, x)
            self.log("train/dist_loss", aeloss, prog_bar=True, logger=True)
            self.log("train/rec_loss", recloss, prog_bar=True, logger=True)            
            self.log("train/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("train/total_loss", loss, prog_bar=False, logger=True)            
            self.log("train/log_perplexity", perplexity, prog_bar=True, logger=True)        
            self.log("train/variance", logvar.mean().exp(), prog_bar=False, logger=True) 

        elif self.args.loss_type == 'gan':
            xrec, qloss, perplexity = self(x)
            if optimizer_idx == 0:
                aeloss = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
                loss = aeloss + qloss
                self.log("train/rec_loss", aeloss, prog_bar=True, logger=True)
                self.log("train/embed_loss", qloss, prog_bar=True, logger=True)    
                self.log("train/total_loss", loss, prog_bar=False, logger=True)                 
                self.log("train/log_perplexity", perplexity, prog_bar=True, logger=True)                       
 
            elif optimizer_idx == 1:  
                discloss = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
                loss = discloss                                             
                self.log("train/disc_loss", discloss, prog_bar=True,logger=True)

        elif self.args.loss_type in ['lpips_l1', 'lpips_l2']:                   
            xrec, qloss, perplexity = self(x)            
            aeloss = self.loss(xrec, x)
            loss = aeloss + qloss      
            if self.args.loss_type == 'lpips_l1':
                recloss = F.l1_loss(xrec, x)
            else:
                recloss = F.mse_loss(xrec, x) 
            self.log("train/lpips_loss", aeloss - recloss, prog_bar=True, logger=True)
            self.log("train/rec_loss", recloss, prog_bar=True, logger=True)
            self.log("train/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("train/total_loss", loss, prog_bar=False, logger=True)            
            self.log("train/log_perplexity", perplexity, prog_bar=True, logger=True)        
        
        else:
            xrec, qloss, perplexity = self(x)            
            aeloss = self.loss(xrec, x)
            loss = aeloss + qloss                 
            self.log("train/rec_loss", aeloss, prog_bar=True, logger=True)
            self.log("train/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("train/total_loss", loss, prog_bar=False, logger=True)            
            self.log("train/log_perplexity", perplexity, prog_bar=True, logger=True)        

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}

        return loss

    def validation_step(self, batch, batch_idx):     
        x = batch[0]
        if self.args.loss_type in ['dist_l1', 'dist_l2']:
            out, qloss, perplexity = self(x, return_var=True)
            xrec = out[0]
            logvar = out[1]
            aeloss = self.loss(xrec, logvar, x)
            loss = aeloss + qloss
            if self.args.loss_type == 'dist_l1':
                recloss = F.l1_loss(xrec, x)
            else:
                recloss = F.mse_loss(xrec, x)
            self.log("val/dist_loss", aeloss, prog_bar=True, logger=True)
            self.log("val/rec_loss", recloss, prog_bar=True, logger=True) 
            self.log("val/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("val/total_loss", loss, prog_bar=False, logger=True)            
            self.log("val/log_perplexity", perplexity, prog_bar=True, logger=True)        
            self.log("val/variance", logvar.mean().exp(), prog_bar=False, logger=True) 

        elif self.args.loss_type == 'gan':
            xrec, qloss, perplexity = self(x)
            aeloss = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
            discloss = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")                                            
            loss = aeloss + qloss
            self.log("val/rec_loss", aeloss, prog_bar=True, logger=True)
            self.log("val/embed_loss", qloss, prog_bar=True, logger=True) 
            self.log("val/disc_loss", discloss, prog_bar=True,logger=True)               
            self.log("val/total_loss", loss, prog_bar=False, logger=True)                 
            self.log("val/log_perplexity", perplexity, prog_bar=True, logger=True)                       
        
        elif self.args.loss_type in ['lpips_l1', 'lpips_l2']:                   
            xrec, qloss, perplexity = self(x)            
            aeloss = self.loss(xrec, x)
            loss = aeloss + qloss      
            if self.args.loss_type == 'lpips_l1':
                recloss = F.l1_loss(xrec, x)
            else:
                recloss = F.mse_loss(xrec, x) 
            self.log("val/lpips_loss", aeloss - recloss, prog_bar=True, logger=True)
            self.log("val/rec_loss", recloss, prog_bar=True, logger=True)
            self.log("val/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("val/total_loss", loss, prog_bar=False, logger=True)            
            self.log("val/log_perplexity", perplexity, prog_bar=True, logger=True)        
                                    
        else:
            xrec, qloss, perplexity = self(x)            
            aeloss = self.loss(xrec, x)
            loss = aeloss + qloss                 
            self.log("val/rec_loss", aeloss, prog_bar=True, logger=True)
            self.log("val/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("val/total_loss", loss, prog_bar=False, logger=True)            
            self.log("val/log_perplexity", perplexity, prog_bar=True, logger=True)        

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}

        return loss

    def configure_optimizers(self):
        if self.args.loss_type == 'gan':
            lr = self.hparams.learning_rate
            params = []
            for encoder in self.encoders:
                params += list(encoder.parameters())
            for pre_quant in self.pre_quants:
                params += list(pre_quant.parameters())
            for quantizer in self.quantizers:
                params += list(quantizer.parameters())
            for post_quant in self.post_quants:
                params += list(post_quant.parameters())
            opt_ae = torch.optim.Adam(set(params),
                                  lr=lr, betas=(0.9, 0.999))
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.9, 0.999))
            if self.args.lr_decay:
                scheduler_ae = ReduceLROnPlateau(
                opt_ae,
                mode="min",
                factor=0.5,
                patience=10,
                cooldown=10,
                min_lr=1e-6,
                verbose=True,
                )  
                scheduler_disc = ReduceLROnPlateau(
                opt_disc,
                mode="min",
                factor=0.5,
                patience=10,
                cooldown=10,
                min_lr=1e-6,
                verbose=True,
                )    
                sched_ae = {'scheduler':scheduler_ae, 'monitor':'train/total_loss'}
                sched_disc = {'scheduler':scheduler_disc, 'monitor':'train/disc_loss'}
                return [opt_ae, opt_disc], [sched_ae, sched_disc]
            else:
                return [opt_ae, opt_disc], []              
        else:
            lr = self.hparams.learning_rate    
            opt = torch.optim.AdamW(self.get_trainable_params(), lr=lr, betas=(0.9, 0.999),weight_decay=1e-4)
            if self.args.lr_decay:
                scheduler = ReduceLROnPlateau(
                opt,
                mode="min",
                factor=0.5,
                patience=10,
                cooldown=10,
                min_lr=1e-6,
                verbose=True,
                )    
                sched = {'scheduler':scheduler, 'monitor':'train/total_loss'}            
                return [opt], [sched]
            else:
                return [opt], []   

    def get_last_layer(self):
        return self.decoders[-1].conv_out.weight

class EMAVQVAE(VQVAE):
    def __init__(self,
                 args, batch_size, learning_rate, 
                 ignore_keys=[], finetuned=False, 
                 ft_attn_resolutions=None,
                 ft_loss_type=None, ft_args=None,
                 ):  
        super().__init__(args, batch_size, learning_rate,
                         ignore_keys=ignore_keys, finetuned=finetuned,
                         ft_attn_resolutions=ft_attn_resolutions,
                         ft_loss_type=ft_loss_type, ft_args=ft_args
                         )
        for quantizer in self.quantizers:
            del quantizer
        del self.quantizers

        self.quantizers = nn.ModuleList()

        quantizer = EMAVectorQuantizer(num_tokens=args.num_tokens,
                                       codebook_dim=args.codebook_dim,
                                       beta=args.quant_beta, decay=args.quant_ema_decay, eps=args.quant_ema_eps) 
        self.quantizers.append(quantizer)
        for i, res in enumerate(args.attn_resolutions[1:]):
            quantizer = EMAVectorQuantizer(num_tokens=args.num_tokens,
                                       codebook_dim=args.codebook_dim,
                                       beta=args.quant_beta, decay=args.quant_ema_decay, eps=args.quant_ema_eps) 
            self.quantizers.append(quantizer)

        self.quant_weight_share()

    def quant_weight_share(self):
        for i in range(len(self.pre_quants[1:])):
            self.pre_quants[i+1].weight = self.pre_quants[0].weight
        for i in range(len(self.quantizers[1:])):
            self.quantizers[i+1].embedding.weight = self.quantizers[0].embedding.weight  
            self.quantizers[i+1].embedding.cluster_size = self.quantizers[0].embedding.cluster_size     
            self.quantizers[i+1].embedding.embed_avg = self.quantizers[0].embedding.embed_avg                              
        for i in range(len(self.post_quants[1:])):
            self.post_quants[i+1].weight = self.post_quants[0].weight 

class GumbelVQVAE(VQVAE):
    def __init__(self,
                 args, batch_size, learning_rate,
                 ignore_keys=[],finetuned=False, 
                 ft_attn_resolutions=None,
                 ft_loss_type=None, ft_args=None,
                 ):  
        super().__init__(args, batch_size, learning_rate, 
                         ignore_keys=ignore_keys, finetuned=finetuned,
                         ft_attn_resolutions=ft_attn_resolutions,
                         ft_loss_type=ft_loss_type, ft_args=ft_args
                         )
        self.temperature = args.starting_temp
        self.anneal_rate = args.anneal_rate
        self.temp_min = args.temp_min
        #quant conv channel should be different for gumbel   
                
        for pre_quant in self.pre_quants:
            del pre_quant
        del self.pre_quants
        self.pre_quants = nn.ModuleList()

        for quantizer in self.quantizers:
            del quantizer
        del self.quantizers
        self.quantizers = nn.ModuleList()
                                    
        pre_quant = torch.nn.Conv2d(args.codebook_dim, args.z_channels,  1)
        self.pre_quants.append(pre_quant) 

        quantizer = GumbelQuantizer(num_tokens=args.num_tokens,
                                       codebook_dim=args.codebook_dim,
                                       kl_weight=args.kl_loss_weight, temp_init=args.starting_temp)
        self.quantizers.append(quantizer)

        for i, res in enumerate(args.attn_resolutions[1:]):
            pre_quant = torch.nn.Conv2d(args.codebook_dim, args.z_channels,  1)
            self.pre_quants.append(pre_quant) 

            quantizer = GumbelQuantizer(num_tokens=args.num_tokens,
                                       codebook_dim=args.codebook_dim,
                                       kl_weight=args.kl_loss_weight, temp_init=args.starting_temp)
            self.quantizers.append(quantizer)

        self.quant_weight_share()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x = batch[0]
        #temperature annealing
        self.temperature = max(self.temperature * torch.exp(-self.anneal_rate * self.global_step), self.temp_min)

        for quantizer in self.quantizers:
            if self.global_step == self.args.kl_anneal:
                quantizer.kl_weight = 1e-8
            quantizer.temperature = self.temperature               

        if self.args.loss_type in ['dist_l1', 'dist_l2']:
            out, qloss, perplexity = self(x, return_var=True)
            xrec = out[0]
            logvar = out[1]
            aeloss = self.loss(xrec, logvar, x)
            loss = aeloss + qloss
            self.log("train/rec_loss", aeloss, prog_bar=True, logger=True)
            self.log("train/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("train/total_loss", loss, prog_bar=False, logger=True)            
            self.log("train/log_perplexity", perplexity, prog_bar=True, logger=True)        
            self.log("train/variance", logvar.mean().exp(), prog_bar=False, logger=True) 
            self.log("train/temperature", self.temperature, prog_bar=False, logger=True)

        elif self.args.loss_type == 'gan':
            xrec, qloss, perplexity = self(x)
            if optimizer_idx == 0:
                aeloss = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
                loss = aeloss + qloss
                self.log("train/rec_loss", aeloss, prog_bar=True, logger=True)
                self.log("train/embed_loss", qloss, prog_bar=True, logger=True)    
                self.log("train/total_loss", loss, prog_bar=False, logger=True)                 
                self.log("train/log_perplexity", perplexity, prog_bar=True, logger=True)                       
                self.log("train/temperature", self.temperature, prog_bar=False, logger=True)
            elif optimizer_idx == 1:  
                discloss = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
                loss = discloss                                             
                self.log("train/disc_loss", discloss, prog_bar=True,logger=True)

        elif self.args.loss_type in ['lpips_l1', 'lpips_l2']:                   
            xrec, qloss, perplexity = self(x)            
            aeloss = self.loss(xrec, x)
            loss = aeloss + qloss      
            if self.args.loss_type == 'lpips_l1':
                recloss = F.l1_loss(xrec, x)
            else:
                recloss = F.mse_loss(xrec, x) 
            self.log("train/lpips_loss", aeloss - recloss, prog_bar=True, logger=True)
            self.log("train/rec_loss", recloss, prog_bar=True, logger=True)
            self.log("train/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("train/total_loss", loss, prog_bar=False, logger=True)            
            self.log("train/log_perplexity", perplexity, prog_bar=True, logger=True)        
                                    
        else:
            xrec, qloss, perplexity = self(x)            
            aeloss = self.loss(xrec, x)
            loss = aeloss + qloss                 
            self.log("train/rec_loss", aeloss, prog_bar=True, logger=True)
            self.log("train/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("train/total_loss", loss, prog_bar=False, logger=True)            
            self.log("train/log_perplexity", perplexity, prog_bar=True, logger=True)        
            self.log("train/temperature", self.temperature, prog_bar=False, logger=True)

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        for quantizer in self.quantizers:
            quantizer.temperature = self.temperature     

        if self.args.loss_type in ['dist_l1', 'dist_l2']:
            out, qloss, perplexity = self(x, return_var=True)
            xrec = out[0]
            logvar = out[1]
            aeloss = self.loss(xrec, logvar, x)
            loss = aeloss + qloss
            self.log("val/rec_loss", aeloss, prog_bar=True, logger=True)
            self.log("val/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("val/total_loss", loss, prog_bar=False, logger=True)            
            self.log("val/log_perplexity", perplexity, prog_bar=True, logger=True)        
            self.log("val/variance", logvar.mean().exp(), prog_bar=False, logger=True) 
            self.log("val/temperature", self.temperature, prog_bar=False, logger=True)
        elif self.args.loss_type == 'gan':
            xrec, qloss, perplexity = self(x)
            aeloss = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
            discloss = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")                                            
            loss = aeloss + qloss
            self.log("val/rec_loss", aeloss, prog_bar=True, logger=True)
            self.log("val/embed_loss", qloss, prog_bar=True, logger=True) 
            self.log("val/disc_loss", discloss, prog_bar=True,logger=True)               
            self.log("val/total_loss", loss, prog_bar=False, logger=True)                 
            self.log("val/log_perplexity", perplexity, prog_bar=True, logger=True)                       
            self.log("val/temperature", self.temperature, prog_bar=False, logger=True)

        elif self.args.loss_type in ['lpips_l1', 'lpips_l2']:                   
            xrec, qloss, perplexity = self(x)            
            aeloss = self.loss(xrec, x)
            loss = aeloss + qloss      
            if self.args.loss_type == 'lpips_l1':
                recloss = F.l1_loss(xrec, x)
            else:
                recloss = F.mse_loss(xrec, x) 
            self.log("val/lpips_loss", aeloss - recloss, prog_bar=True, logger=True)
            self.log("val/rec_loss", recloss, prog_bar=True, logger=True)
            self.log("val/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("val/total_loss", loss, prog_bar=False, logger=True)            
            self.log("val/log_perplexity", perplexity, prog_bar=True, logger=True)        
        
        else:
            xrec, qloss, perplexity = self(x)            
            aeloss = self.loss(xrec, x)
            loss = aeloss + qloss                 
            self.log("val/rec_loss", aeloss, prog_bar=True, logger=True)
            self.log("val/embed_loss", qloss, prog_bar=True, logger=True)
            self.log("val/total_loss", loss, prog_bar=False, logger=True)            
            self.log("val/log_perplexity", perplexity, prog_bar=True, logger=True)        
            self.log("val/temperature", self.temperature, prog_bar=False, logger=True)

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}

        return loss