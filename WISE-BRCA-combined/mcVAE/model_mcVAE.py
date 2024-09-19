import torch

from torch import nn
from utils import Normal, compute_kl, compute_ll, compute_logvar
from model_WISE_BRCA import WISE_BRCA


class mcVAE(nn.Module):
    def __init__(self, patho_dim=1536, pheno_dim=96, WISE_BRCA=WISE_BRCA):
        super(mcVAE, self).__init__()

        self.encoder_patho = nn.Sequential(nn.Linear(patho_dim, 2*patho_dim), nn.LeakyReLU(), nn.Linear(2*patho_dim, patho_dim), nn.LeakyReLU(), nn.Linear(patho_dim, patho_dim//2))
        self.encoder_pheno = nn.Sequential(nn.Linear(21, 2*pheno_dim), nn.LeakyReLU(), nn.Linear(2*pheno_dim, 4*pheno_dim), nn.LeakyReLU(), nn.Linear(4*pheno_dim, 8*pheno_dim))
        self.decoder_patho = nn.Sequential(nn.Linear(patho_dim//2, patho_dim//2), nn.LeakyReLU(), nn.Linear(patho_dim//2, patho_dim), nn.LeakyReLU(), nn.Linear(patho_dim, patho_dim))
        self.decoder_pheno = nn.Sequential(nn.Linear(8*pheno_dim, 4*pheno_dim), nn.LeakyReLU(), nn.Linear(4*pheno_dim, 2*pheno_dim), nn.LeakyReLU(), nn.Linear(2*pheno_dim, 21))
        self.log_alpha_patho = torch.nn.Parameter(torch.Tensor(1, patho_dim//2))
        self.log_alpha_pheno = torch.nn.Parameter(torch.Tensor(1, 8*pheno_dim))
        tmp_noise_par_patho = torch.FloatTensor(1, 1536).fill_(-3)
        tmp_noise_par_pheno = torch.FloatTensor(1, 21).fill_(-3)
        self.W_out_logvar_patho = torch.nn.Parameter(data=tmp_noise_par_patho, requires_grad=True)
        self.W_out_logvar_pheno = torch.nn.Parameter(data=tmp_noise_par_pheno, requires_grad=True)
        torch.nn.init.normal_(self.log_alpha_patho, 0.0, 0.01)
        torch.nn.init.normal_(self.log_alpha_pheno, 0.0, 0.01)
        self.WISE_BRCA = WISE_BRCA()
        self.sparse = True
        self.n_channels = 2
        self.beta = 1.0
        self.enc_channels = list(range(self.n_channels))
        self.dec_channels = list(range(self.n_channels))

        for layer in self.encoder_patho:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)
        for layer in self.encoder_pheno:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)
        for layer in self.decoder_patho:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)
        for layer in self.decoder_pheno:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)

    def load_state_dict_from_checkpoint(self):
        print('Load pathology pretrained parameters!')
        self.WISE_BRCA = self.WISE_BRCA.cuda()
        pretrained_dict = torch.load(
            rf'../../checkpoints/WISE_BRCA.pth',
            map_location='cuda:0')['model']
        model_dict = self.WISE_BRCA.state_dict()
        state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.WISE_BRCA.load_state_dict(model_dict)
        for k, v in self.WISE_BRCA.named_parameters():
            v.requires_grad = False

    def encode_patho(self, x):
        mu = self.encoder_patho(x)
        logvar = compute_logvar(mu, self.log_alpha_patho)
        return Normal(loc=mu, scale=logvar.exp().pow(0.5))

    def decode_patho(self, z):
        pi = Normal(
            loc=self.decoder_patho(z),
            scale=self.W_out_logvar_patho.exp().pow(0.5)
        )
        return pi

    def encode_pheno(self, x):
        mu = self.encoder_pheno(x)
        logvar = compute_logvar(mu, self.log_alpha_pheno)
        return Normal(loc=mu, scale=logvar.exp().pow(0.5))

    def compute_kl(self, q):
        kl = 0
        for i, qi in enumerate(q):
            if i in self.enc_channels:
                # "compute_kl" ignores p2 if sparse=True.
                kl += compute_kl(p1=qi, p2=Normal(0, 1), sparse=self.sparse)

        return kl

    def compute_ll(self, p, x):
        # p[x][z]: p(x|z)
        ll = 0
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                # xi = reconstructed; zj = encoding
                if i in self.dec_channels and j in self.enc_channels:
                    ll += compute_ll(p=p[i][j], x=x[i])

        return ll

    def decode_pheno(self, z):
        pi = Normal(
            loc=self.decoder_pheno(z),
            scale=self.W_out_logvar_pheno.exp().pow(0.5)
        )
        return pi

    def loss_function(self, fwd_ret):
        x = fwd_ret['x']
        q = fwd_ret['q']
        p = fwd_ret['p']

        kl = self.compute_kl(q)
        kl *= self.beta
        ll = self.compute_ll(p=p, x=x)

        total = kl - ll
        return total

    def forward(self, x1, x2, pheno_feats):

        patho_feats = self.WISE_BRCA(x1, x2)
        q_patho = self.encode_patho(patho_feats)
        q_pheno = self.encode_pheno(pheno_feats)
        if self.training:
            z_patho = q_patho.rsample()
            z_pheno = q_pheno.rsample()
        else:
            z_patho = q_patho.loc
            z_pheno = q_pheno.loc

        p_patho = self.decode_patho(z_patho)
        p_pheno = self.decode_pheno(z_pheno)
        p_patho_pheno = self.decode_patho(z_pheno)
        p_pheno_patho = self.decode_pheno(z_patho)
        q = [q_patho, q_pheno]
        x = [patho_feats, pheno_feats]
        p = [[p_patho, p_patho_pheno], [p_pheno, p_pheno_patho]]
        fwd_ret = {
            'q':q,
            'x':x,
            'p':p
        }
        l_mcVAE = self.loss_function(fwd_ret)
        return l_mcVAE




