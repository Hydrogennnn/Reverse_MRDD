from .specificity_models import ViewSpecificAE
import torch.nn as nn
import torch

class IVAE(nn.Module):
    def __init__(self, args, device = 'cpu'):
        super(IVAE, self).__init__()
        self.args = args
        self.device = device
        self.views = args.views
        # m unimodal vae
        for i in range(self.args.views):
            self.__setattr__(f"venc_{i + 1}", ViewSpecificAE(c_enable=False,
                                                             v_dim=self.args.vspecific.v_dim,
                                                             latent_ch=self.args.vspecific.latent_ch,
                                                             num_res_blocks=self.args.vspecific.num_res_blocks,
                                                             block_size=self.args.vspecific.block_size,
                                                             channels=self.args.vspecific.in_channel,
                                                             basic_hidden_dim=self.args.vspecific.basic_hidden_dim,
                                                             ch_mult=self.args.vspecific.ch_mult,
                                                             init_method=self.args.backbone.init_method,
                                                             kld_weight=self.args.vspecific.kld_weight,
                                                             device=self.device))



    def forward(self, Xs):
        outs = []
        for i in range(self.views): #each view
            venc = self.__getattr__(f"venc_{i+1}")
            out, _, _ = venc(Xs[i]) #decoder(z), mu, log(v)
            outs.append(out)

        return outs

    def get_loss(self, Xs):
        return_details = {}
        losses = []
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            recon_loss, kld_loss = venc.get_loss(Xs[i])

            return_details[f"v{i+1}_recon-loss"] = recon_loss.item()
            return_details[f"v{i+1}_kld-loss"] = kld_loss.item()

            loss = kld_loss + recon_loss
            return_details[f"v{i+1}_total-loss"] = loss.item()
            losses.append(loss)
        return losses, return_details
    












