import torch
import torch.nn as nn
from .independent_VAE import IVAE
from .consistency_models import ConsistencyAE
from .mi_estimators import CLUBSample

class RMRDD(nn.Module):
    def __init__(self, config, specific_encoder_path = None, device = 'cpu'):
        super(RMRDD, self).__init__()
        self.config = config
        self.views = self.config.views
        self.device = device
        self.c_dim = config.consistency.c_dim
        self.v_dim = config.vspecific.v_dim

        # Specific model
        self.spe_enc = IVAE(args=self.config, device=device)

        # Freeze specific encoders
        assert specific_encoder_path
        self.spe_enc.load_state_dict(torch.load(
            specific_encoder_path, map_location= 'cpu'
        ))
        for param in self.spe_enc.parameters():
            param.requires_grad = False

        # Consistency model
        self.cons_enc = ConsistencyAE(
            basic_hidden_dim=config.consistency.basic_hidden_dim,
            c_dim=config.consistency.c_dim,
            v_dim=config.vspecific.v_dim,
            continous=config.consistency.continous,
            in_channel=config.consistency.in_channel,
            num_res_blocks=config.consistency.num_res_blocks,
            ch_mult=config.consistency.ch_mult,
            block_size=config.consistency.block_size,
            temperature=config.consistency.temperature,
            latent_ch=config.consistency.latent_ch,
            kld_weight=config.consistency.kld_weight,
            views=config.views,
            categorical_dim=config.dataset.class_num
        )

        # mutual information estimation
        for i in range(self.views):
            self.__setattr__(f"mi_est_{i+1}", CLUBSample(self.c_dim, self.v_dim, hidden_size=self.config.disent.hidden_size))
    def get_loss(self, Xs):
        # extract specific-views
        assert len(Xs) == self.views
        spe_repr = self.vspecific_features(Xs)
        con_repr = self.consistency_features(Xs)
        # kld loss & reconstruction loss of each view
        loss = self.cons_enc.get_loss(Xs=Xs,
                                      Ys=spe_repr,
                                      mask_ratio=self.config.train.masked_ratio,
                                      mask_patch_size=self.config.train.mask_patch_size
                                      )

        # MI loss
        for i in range(self.views):
            mi_est = self.__getattr__(f"mi_est_{i+1}")
            disent_loss = mi_est.learning_loss(spe_repr[i], con_repr)
            loss += disent_loss

        return loss

    def forward(self, Xs):
        con_repr = self.cons_enc(Xs)

    def all_features(self, Xs):
        C = self.consistency_features(Xs)
        spe_repr = self.vspecific_features(Xs)
        V = spe_repr[self.config.vspecific.best_view]
        return C, V, torch.cat([C, V], dim=-1)


    @torch.no_grad()
    def consistency_features(self, Xs):
        consist_feature = self.cons_enc.consistency_features(Xs)
        return consist_feature


    @torch.no_grad()
    def vspecific_features(self, Xs, best_view=False):
        vspecific_features = []
        for i in range(self.views):
            venc = self.spe_enc.__getattr__(f"venc_{i+1}")
            feature = venc.latent(Xs[i])
            vspecific_features.append(feature)
        if best_view:
            return vspecific_features[self.config.best_view]
        else:
            return vspecific_features











