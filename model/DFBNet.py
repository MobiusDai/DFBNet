from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from model.base.transformer import Transformer, PositionalEncoding


class DFBNet(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize, shot=1):
        super(DFBNet, self).__init__()
        self.shot = shot 
        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        self.model = FEC_model(in_channels=self.feat_channels, stack_ids=self.stack_ids, shot=self.shot)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, support_img, support_mask):
        if self.shot == 1:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                support_feats = self.extract_feats(support_img)
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                for k in range(self.shot):
                    # print(support_img.shape)
                    support_feats = self.extract_feats(support_img[:, k])
                    n_support_feats.append(support_feats)
                support_feats = n_support_feats

        logit_mask = self.model(query_feats, support_feats, support_mask.clone())

        return logit_mask

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []
        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']

        if nshot == 1:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                support_feats = self.extract_feats(support_imgs[0])
                logit_mask = self.model(query_feats, support_feats, support_masks[0].clone())
        else:
            # for 5-shot, max-vote is also a good choice, but we choose to retrain the model.
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                for k in range(nshot):
                    support_feats = self.extract_feats(support_imgs[:, k])
                    n_support_feats.append(support_feats)
                logit_mask = self.model(query_feats, n_support_feats, support_masks.clone())

        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(logit_mask, support_imgs[0].size()[2:], mode='bilinear', align_corners=True)

        return logit_mask.argmax(dim=1)

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()

class FEC_model(nn.Module):
    def __init__(self, in_channels, stack_ids, shot):
        super(FEC_model, self).__init__()
        self.shot = shot
        self.stack_ids = stack_ids
        self.in_channels = in_channels
        outch1, outch2, outch3 = 512, 256, 128

        self.atten1 = Transformer(dim = 2*self.in_channels[3], depth = 2, heads = 8, dim_head = 64, mlp_dim = 1024, out_dim=outch1, dropout = 0.5)
        self.PE1 = PositionalEncoding(d_model = 2*self.in_channels[3], dropout=0.5)
        self.atten2 = Transformer(dim = 2*self.in_channels[2], depth = 2, heads = 8, dim_head = 64, mlp_dim = 1024, out_dim=outch1, dropout = 0.5)
        self.PE2 = PositionalEncoding(d_model = 2*self.in_channels[2], dropout=0.5)
        self.atten3 = Transformer(dim = 2*self.in_channels[1], depth = 2, heads = 8, dim_head = 64, mlp_dim = 1024, out_dim=outch1, dropout = 0.5)
        self.PE3 = PositionalEncoding(d_model = 2*self.in_channels[1], dropout=0.5)

        self.mix1 = self.build_conv_block(outch1, [outch1, outch1], [3, 3], [1, 1])
        self.mix2 = self.build_conv_block(outch1, [outch1, outch1], [3, 3], [1, 1])

        self.mix3 = nn.Sequential(
            nn.Conv2d(outch1+self.in_channels[1]+self.in_channels[0], outch1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(outch1, outch2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.mix4 = nn.Sequential(
            nn.Conv2d(outch2, outch3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(outch3, outch3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        self.cls = nn.Sequential(nn.Conv2d(outch3, 64, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 2, (3, 3), padding=(1, 1), bias=True))
        
    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))
        building_block_layers.append(nn.Dropout2d(0.2))

        return nn.Sequential(*building_block_layers)
        
    def SSP_func(self, feature_q, out, ch):
        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7 #0.9 #0.6
            bg_thres = 0.6 #0.6
            cur_feat = feature_q[epi].view(ch, -1)
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices] #.mean(-1)
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices] #.mean(-1)
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(ch, f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(ch, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

    def forward(self, query_feats, support_feats, support_mask):
        results = {
            'query_feats':{'s8':[], 's16':[], 's32':[]},
            'prior mask 0':{'s8':[], 's16':[], 's32':[]},
            'prior mask 1':{'s8':[], 's16':[], 's32':[]},
            'enhanced feats':{'s8':None, 's16':None, 's32':None},
            'align feats':{'s8':None, 's16':None, 's32':None},
            }
        origin_size = query_feats[0].shape[-2:]
        origin_size = (origin_size[0]*4, origin_size[1]*4)
        masked_queryfeats = []
        bmasked_queryfeats = []
        for idx, query_feat in enumerate(query_feats):
            if idx < self.stack_ids[0]: continue 
            if idx < self.stack_ids[1]:
                results['query_feats']['s8'].append(query_feat)
            elif idx < self.stack_ids[2]:
                results['query_feats']['s16'].append(query_feat)
            else:
                results['query_feats']['s32'].append(query_feat)
            if self.shot == 1:
                feature_fg_list = [self.masked_average_pooling(support_feats[idx],(support_mask == 1).float())[None, :]]
                feature_bg_list = [self.masked_average_pooling(support_feats[idx],(support_mask == 0).float())[None, :]]
            else:
                feature_fg_list = []
                feature_bg_list = []
                for k in range(self.shot):
                    fg = self.masked_average_pooling(support_feats[k][idx],(support_mask[:,k] == 1).float())[None, :]
                    bg = self.masked_average_pooling(support_feats[k][idx],(support_mask[:,k] == 0).float())[None, :]
                    feature_fg_list.append(fg)
                    feature_bg_list.append(bg)
            FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
            BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

            out_0 = self.similarity_func(query_feat, FP, BP)
            if idx < self.stack_ids[1]:
                results['prior mask 0']['s8'].append(out_0/10)
            elif idx < self.stack_ids[2]:
                results['prior mask 0']['s16'].append(out_0/10)
            else:
                results['prior mask 0']['s32'].append(out_0/10)

            SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(query_feat, out_0, query_feat.shape[1])
            FP_1 = FP * 0.5 + SSFP_1 * 0.5
            BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
            
            out_1 = self.similarity_func(query_feat, FP_1, BP_1)
            if idx < self.stack_ids[1]:
                results['prior mask 1']['s8'].append(out_1/10)
            elif idx < self.stack_ids[2]:
                results['prior mask 1']['s16'].append(out_1/10)
            else:
                results['prior mask 1']['s32'].append(out_1/10)

            out_1 = out_1.softmax(dim=1)
            out_1f = (out_1[:, 1, ...].unsqueeze(1)-0.5)*4
            out_1b = (out_1[:, 0, ...].unsqueeze(1)-0.5)*4
            masked_queryfeats.append(query_feat * out_1f)
            bmasked_queryfeats.append(query_feat * out_1b)
        
        mask_feat_1f = sum(masked_queryfeats[self.stack_ids[2]-self.stack_ids[0]:self.stack_ids[3]-self.stack_ids[0]]) # 1/32
        mask_feat_2f = sum(masked_queryfeats[self.stack_ids[1]-self.stack_ids[0]:self.stack_ids[2]-self.stack_ids[0]]) # 1/16
        mask_feat_3f = sum(masked_queryfeats[0: self.stack_ids[1]-self.stack_ids[0]]) # 1/8
        
        mask_feat_1b = sum(bmasked_queryfeats[self.stack_ids[2]-self.stack_ids[0]:self.stack_ids[3]-self.stack_ids[0]]) # 1/32
        mask_feat_2b = sum(bmasked_queryfeats[self.stack_ids[1]-self.stack_ids[0]:self.stack_ids[2]-self.stack_ids[0]]) # 1/16
        mask_feat_3b = sum(bmasked_queryfeats[0: self.stack_ids[1]-self.stack_ids[0]]) # 1/8

        mask_feat_1 = torch.cat([mask_feat_1b, mask_feat_1f], dim=1)
        results['enhanced feats']['s8'] = mask_feat_1
        bsz, ch, h, w = mask_feat_1.shape
        mask_feat_1 = mask_feat_1.view(bsz, ch, h*w).permute(0, 2, 1)
        mask_feat_1 = self.atten1(self.PE1(mask_feat_1))
        mask_feat_1 = mask_feat_1.permute(0, 2, 1).view(bsz, 512, h, w)
        results['align feats']['s8'] = mask_feat_1

        mask_feat_2 = torch.cat([mask_feat_2b, mask_feat_2f], dim=1)
        results['enhanced feats']['s16'] = mask_feat_2
        bsz, ch, h, w = mask_feat_2.shape
        mask_feat_2 = mask_feat_2.view(bsz, ch, h*w).permute(0, 2, 1)
        mask_feat_2 = self.atten2(self.PE2(mask_feat_2))
        mask_feat_2 = mask_feat_2.permute(0, 2, 1).view(bsz, 512, h, w)
        results['align feats']['s16'] = mask_feat_2

        mask_feat_3 = torch.cat([mask_feat_3b, mask_feat_3f], dim=1)
        bsz, ch, h, w = mask_feat_3.shape
        results['enhanced feats']['s32'] = mask_feat_3
        mask_feat_3 = mask_feat_3.view(bsz, ch, h*w).permute(0, 2, 1)
        mask_feat_3 = self.atten3(self.PE3(mask_feat_3))
        mask_feat_3 = mask_feat_3.permute(0, 2, 1).view(bsz, 512, h, w)
        results['align feats']['s32'] = mask_feat_3
        query_feat = query_feats[self.stack_ids[0]-1]

        mask_feat_1 = F.interpolate(mask_feat_1, size=mask_feat_2.shape[-2:], mode='bilinear', align_corners=True)
        mix_feat = mask_feat_1 + mask_feat_2
        mix_feat = self.mix1(mix_feat)
        mix_feat = F.interpolate(mix_feat, size=mask_feat_3.shape[-2:], mode='bilinear', align_corners=True)
        mix_feat = mask_feat_3 + mix_feat
        mix_feat = self.mix2(mix_feat)
        mix_feat = torch.cat([mix_feat, query_feats[self.stack_ids[1]-1]], dim=1)
        mix_feat = F.interpolate(mix_feat, size=query_feat.shape[-2:], mode='bilinear', align_corners=True)
        mix_feat = torch.cat([mix_feat, query_feat], dim=1)
        mix_feat = self.mix3(mix_feat)
        mix_feat = F.interpolate(mix_feat, size=(origin_size[0]//2, origin_size[1]//2), mode='bilinear', align_corners=True)
        mix_feat = self.mix4(mix_feat)
        mix_feat = F.interpolate(mix_feat, size=origin_size, mode='bilinear', align_corners=True)
        results['mix_feat'] = mix_feat
        logit_mask = self.cls(mix_feat)

        return logit_mask, results