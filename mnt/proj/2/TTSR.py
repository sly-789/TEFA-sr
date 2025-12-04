# TTSR.py
import torch
import torch.nn as nn
from model import LTE, MainNet, SearchTransfer
from utils import SpatialAdaptationModule, AttentionSpatialAdaptationModule


class TTSR(nn.Module):
    def __init__(self, args):
        super(TTSR, self).__init__()
        self.args = args
        self.LTE = LTE.LTE(requires_grad=True)
        self.num_res_blocks = args.num_res_blocks
        self.MainNet = MainNet.MainNet(in_channels=3,num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, res_scale=args.res_scale)
        self.LTE_copy = LTE.LTE(requires_grad=False)
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.sam_module = AttentionSpatialAdaptationModule(256)

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if sr is not None:
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        _, _, lrsr_lv3 = self.LTE((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)
        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
        # if isinstance(lr,tuple):
        #     lr_feature = lr[0]
        # else:
        #     lr_feature = lr
        lr_feature = self.LTE(lr)[0]
        #lr_feature = self.LTE(lr)
        ref_feature  = ref_lv3
        adapted_features = self.sam_module(ref_feature, lr_feature)

        sr_output = self.MainNet(adapted_features)
        #sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)
        return sr_output, S, T_lv3, T_lv2, T_lv1