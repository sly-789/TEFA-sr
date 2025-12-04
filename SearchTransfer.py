import torch
import torch.nn as nn
import torch.nn.functional as F

class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        ### search
        lrsr_lv3_unfold = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2)
        lrsr_lv3_unfold = F.normalize(lrsr_lv3_unfold, dim=1)

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold)
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)

        #jiancha R_lv3 R_lv3_star de value

        # print("R_lv3 shape:",R_lv3.shape)
        # print("R_lv3 min:", R_lv3.min())
        # print("R_lv3 max:", R_lv3.max())
        # print("R_lv3_star shape:", R_lv3_star.shape)
        # print("R_lv3_star min:", R_lv3_star.min())
        # print("R_lv3_star max:", R_lv3_star.max())

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3, 3), padding=1) / (3 * 3)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2) * 2, lrsr_lv3.size(3) * 2), kernel_size=(6, 6), padding=2, stride=2) / (3 * 3)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2) * 4, lrsr_lv3.size(3) * 4), kernel_size=(12, 12), padding=4, stride=4) / (3 * 3)

        # Ensure S has the same spatial dimensions as lrsr_lv3
        R_lv3_star = R_lv3_star.view(R_lv3_star.size(0), 1, *lrsr_lv3.size()[-2:])
        S = torch.sigmoid(R_lv3_star)
        # print("S shape:",S.shape)
        # print("S min:", S.min())
        # print("S max:", S.max())
        if refsr_lv3 is None:
            T_lv3 = torch.zeros_like(lrsr_lv3)
        else:
            T_lv3 = refsr_lv3

        if ref_lv1 is None:
            T_lv1 = torch.zeros_like(lrsr_lv3)
        else:
            T_lv1 = ref_lv1

        if ref_lv2 is None:
            T_lv2 = torch.zeros_like(lrsr_lv3)
        else:
            T_lv2 = ref_lv2

        return S, T_lv3, T_lv2, T_lv1