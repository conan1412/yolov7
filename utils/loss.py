# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou, bbox_alpha_iou, box_iou, box_giou, box_diou, box_ciou, xywh2xyxy
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class SigmoidBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale = 2.0, use_loss_regression=True, use_fw_regression=True, BCE_weight=1.0, smooth_eps=0.0):
        super(SigmoidBin, self).__init__()
        
        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight

        start = min + (self.scale/2.0) / self.bin_count
        end = max - (self.scale/2.0) / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        #print(f" start = {start}, end = {end}, step = {step} ")

        bins = torch.range(start, end + 0.0001, step).float() 
        self.register_buffer('bins', bins) 
               

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)

        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]

        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)

        return result


    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (pred.shape[0], target.shape[0])
        device = pred.device

        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)
    
        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias

        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # targets
        n = pred.shape[0] 
        target_bins[range(n), bin_idx] = self.cp

        loss_bin = self.BCEbins(pred_bin, target_bins) # BCE

        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)  # MSE        
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin

        out_result = result.clamp(min=self.min, max=self.max)

        return loss, out_result


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class RankSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta_RS=0.50, eps=1e-10): 

        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets > 0.)
        fg_logits = logits[fg_labels]
        fg_targets = targets[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta_RS
        relevant_bg_labels=((targets==0) & (logits>=threshold_logit))
        
        relevant_bg_logits = logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        sorting_error=torch.zeros(fg_num).cuda()
        ranking_error=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            # Difference Transforms (x_ij)
            fg_relations=fg_logits-fg_logits[ii] 
            bg_relations=relevant_bg_logits-fg_logits[ii]

            if delta_RS > 0:
                fg_relations=torch.clamp(fg_relations/(2*delta_RS)+0.5,min=0,max=1)
                bg_relations=torch.clamp(bg_relations/(2*delta_RS)+0.5,min=0,max=1)
            else:
                fg_relations = (fg_relations >= 0).float()
                bg_relations = (bg_relations >= 0).float()

            # Rank of ii among pos and false positive number (bg with larger scores)
            rank_pos=torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)

            # Rank of ii among all examples
            rank=rank_pos+FP_num
                            
            # Ranking error of example ii. target_ranking_error is always 0. (Eq. 7)
            ranking_error[ii]=FP_num/rank      

            # Current sorting error of example ii. (Eq. 7)
            current_sorting_error = torch.sum(fg_relations*(1-fg_targets))/rank_pos

            #Find examples in the target sorted order for example ii         
            iou_relations = (fg_targets >= fg_targets[ii])
            target_sorted_order = iou_relations * fg_relations

            #The rank of ii among positives in sorted order
            rank_pos_target = torch.sum(target_sorted_order)

            #Compute target sorting error. (Eq. 8)
            #Since target ranking error is 0, this is also total target error 
            target_sorting_error= torch.sum(target_sorted_order*(1-fg_targets))/rank_pos_target

            #Compute sorting error on example ii
            sorting_error[ii] = current_sorting_error - target_sorting_error
  
            #Identity Update for Ranking Error 
            if FP_num > eps:
                #For ii the update is the ranking error
                fg_grad[ii] -= ranking_error[ii]
                #For negatives, distribute error via ranking pmf (i.e. bg_relations/FP_num)
                relevant_bg_grad += (bg_relations*(ranking_error[ii]/FP_num))

            #Find the positives that are misranked (the cause of the error)
            #These are the ones with smaller IoU but larger logits
            missorted_examples = (~ iou_relations) * fg_relations

            #Denominotor of sorting pmf 
            sorting_pmf_denom = torch.sum(missorted_examples)

            #Identity Update for Sorting Error 
            if sorting_pmf_denom > eps:
                #For ii the update is the sorting error
                fg_grad[ii] -= sorting_error[ii]
                #For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
                fg_grad += (missorted_examples*(sorting_error[ii]/sorting_pmf_denom))

        #Normalize gradients by number of positives 
        classification_grads[fg_labels]= (fg_grad/fg_num)
        classification_grads[relevant_bg_labels]= (relevant_bg_grad/fg_num)

        ctx.save_for_backward(classification_grads)

        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None

class aLRPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta=1., eps=1e-5): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example to compute classification loss 
            prec[ii]=rank_pos/rank[ii]                
            #For stability, set eps to a infinitesmall value (e.g. 1e-6), then compute grads
            if FP_num > eps:   
                fg_grad[ii] = -(torch.sum(fg_relations*regression_losses)+FP_num)/rank[ii]
                relevant_bg_grad += (bg_relations*(-fg_grad[ii]/FP_num))   
                    
        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= (fg_num)
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss, rank, order

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None, None
    
    
class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta=1.): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example 
            current_prec=rank_pos/rank[ii]
            
            #Compute interpolated AP and store gradients for relevant bg examples
            if (max_prec<=current_prec):
                max_prec=current_prec
                relevant_bg_grad += (bg_relations/rank[ii])
            else:
                relevant_bg_grad += (bg_relations/rank[ii])*(((1-max_prec)/(1-current_prec)))
            
            #Store fg gradients
            fg_grad[ii]=-(1-max_prec)
            prec[ii]=max_prec 

        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= fg_num
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss

    @staticmethod
    def backward(ctx, out_grad1):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.1, .05])  # P3-P7
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.5, 0.4, .1])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    #t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ComputeLossOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                #pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
    

class ComputeLossBinOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossBinOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        #MSEangle = nn.MSELoss().to(device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride', 'bin_count':
            setattr(self, k, getattr(det, k))

        #xy_bin_sigmoid = SigmoidBin(bin_count=11, min=-0.5, max=1.5, use_loss_regression=False).to(device)
        wh_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0, use_loss_regression=False).to(device)
        #angle_bin_sigmoid = SigmoidBin(bin_count=31, min=-1.1, max=1.1, use_loss_regression=False).to(device)
        self.wh_bin_sigmoid = wh_bin_sigmoid

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            obj_idx = self.wh_bin_sigmoid.get_length()*2 + 2     # x,y, w-bce, h-bce     # xy_bin_sigmoid.get_length()*2

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                
                #pxy = ps[:, :2].sigmoid() * 2. - 0.5
                ##pxy = ps[:, :2].sigmoid() * 3. - 1.
                #pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                #pbox = torch.cat((pxy, pwh), 1)  # predicted box

                #x_loss, px = xy_bin_sigmoid.training_loss(ps[..., 0:12], tbox[i][..., 0])
                #y_loss, py = xy_bin_sigmoid.training_loss(ps[..., 12:24], tbox[i][..., 1])
                w_loss, pw = self.wh_bin_sigmoid.training_loss(ps[..., 2:(3+self.bin_count)], selected_tbox[..., 2] / anchors[i][..., 0])
                h_loss, ph = self.wh_bin_sigmoid.training_loss(ps[..., (3+self.bin_count):obj_idx], selected_tbox[..., 3] / anchors[i][..., 1])

                pw *= anchors[i][..., 0]
                ph *= anchors[i][..., 1]

                px = ps[:, 0].sigmoid() * 2. - 0.5
                py = ps[:, 1].sigmoid() * 2. - 0.5

                lbox += w_loss + h_loss # + x_loss + y_loss

                #print(f"\n px = {px.shape}, py = {py.shape}, pw = {pw.shape}, ph = {ph.shape} \n")

                pbox = torch.cat((px.unsqueeze(1), py.unsqueeze(1), pw.unsqueeze(1), ph.unsqueeze(1)), 1).to(device)  # predicted box

                
                
                
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, (1+obj_idx):], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, (1+obj_idx):], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., obj_idx], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                obj_idx = self.wh_bin_sigmoid.get_length()*2 + 2
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, obj_idx:(obj_idx+1)])
                p_cls.append(fg_pred[:, (obj_idx+1):])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pw = self.wh_bin_sigmoid.forward(fg_pred[..., 2:(3+self.bin_count)].sigmoid()) * anch[i][idx][:, 0] * self.stride[i]
                ph = self.wh_bin_sigmoid.forward(fg_pred[..., (3+self.bin_count):obj_idx].sigmoid()) * anch[i][idx][:, 1] * self.stride[i]
                
                pxywh = torch.cat([pxy, pw.unsqueeze(1), ph.unsqueeze(1)], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]            
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs       

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


class ComputeLossAuxOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossAuxOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':  # det是个类，把det.k赋值给self.k
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device  # targets: [[0.00000, 0.00000, 0.58512, 0.27778, 0.59368, 0.33896]]
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets2(p[:self.nl], targets, imgs)  # self.nl: 4,
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets, imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]]  # pre_gen_gains_aux的值: [[160, 160, 160, 160],[80, 80, 80, 80],[40, 40, 40, 40],[20, 20, 20, 20]]
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]]  # pre_gen_gains的值: [[160, 160, 160, 160],[80, 80, 80, 80],[40, 40, 40, 40],[20, 20, 20, 20]]
    

        # Losses
        for i in range(self.nl):  # layer index, layer predictions, i: 2
            pi = p[i]  # pi.shape: [1, 3, 40, 40, 10]
            pi_aux = p[i+self.nl]  # pi_aux.shape: [1, 3, 40, 40, 10]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            # b: [0, 0, 0, 0, 0, 0]
            # a: [2, 2, 2, 2, 2, 2]
            # gj: [20, 20, 20, 20, 21, 21]
            # gi: [21, 29, 20, 28, 21, 29]
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]  # image, anchor, gridy, gridx
            # b_aux: [0, 0, 0, 0, 0, 0, 0]
            # a_aux: [2, 2, 2, 2, 2, 2, 2]
            # gj_aux: [20, 20, 20, 19, 19, 21, 21]
            # gi_aux: [21, 29, 20, 21, 29, 21, 29]
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj, tobj.shape: [1, 3, 40, 40]
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj, tobj_aux.shape: [1, 3, 40, 40]

            n = b.shape[0]  # number of targets, n: 6
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets, ps.shape: [6, 10]

                # Regression
                grid = torch.stack([gi, gj], dim=1)  # grid: [[21, 20],[29, 20],[20, 20],[28, 20],[21, 21],[29, 21]]
                pxy = ps[:, :2].sigmoid() * 2. - 0.5  # pxy.shape: [6, 2]
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # pwh.shape: [6, 2]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box, pbox.shape: [6, 4]
                # pbox: [[0.50276, 0.49482, 7.25245, 16.93105],
                #      [0.50263, 0.49700, 7.37642, 16.93352],
                #      [0.49943, 0.49512, 7.41274, 16.93250],
                #      [0.50340, 0.49598, 7.36602, 16.93436],
                #      [0.50062, 0.49776, 7.33114, 16.92381],
                #      [0.50213, 0.49341, 7.34410, 16.93394]]
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]  # selected_tbox.shape: [6, 4], 还原当前feature_map上的真实坐标
                # selected_tbox: [[21.27874, 20.68615, 8.01240, 32.36983],
                #              [29.44584, 20.56247, 8.44552, 32.30798],
                #              [21.27874, 20.68615, 8.01240, 32.36983],
                #              [29.44584, 20.56247, 8.44552, 32.30798],
                #              [21.27874, 20.68615, 8.01240, 32.36983],
                #              [29.44584, 20.56247, 8.44552, 32.30798]]
                selected_tbox[:, :2] -= grid
                # selected_tbox: [[0.27874, 0.68615, 8.01240, 32.36983],
                #              [0.44584, 0.56247, 8.44552, 32.30798],
                #              [1.27874, 0.68615, 8.01240, 32.36983],
                #              [1.44584, 0.56247, 8.44552, 32.30798],
                #              [0.27874, -0.31385, 8.01240, 32.36983],
                #              [0.44584, -0.43753, 8.44552, 32.30798]]
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target), 使用CIOU计算方式
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio, self.gr: 1, tobj[b, a, gj, gi]: [0.47315, 0.45760, 0.43809, 0.42067, 0.47750, 0.45484]

                # Classification
                selected_tcls = targets[i][:, 1].long()  # 类别id, selected_tcls: [0, 0, 0, 0, 0, 0]
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets, self.cn标签平滑
                    # t: [[0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.]]
                    t[range(n), selected_tcls] = self.cp
                    # t: [[1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.]]
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE, 无fl_gamma为BCEWithLogitsLoss(),有fl_gamma为FocalLoss

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            
            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets, ps_aux.shape: [7, 10]
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)  # grig_aux.shape: [7, 2]
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5  # pxy_aux.shape: [7, 2]
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]  # pwh_aux.shape: [7, 2]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box, pbox_aux.shape: [7, 4]
                # pbox_aux: [[0.16602, 0.46875, 9.10515, 21.25085],
                #  [0.41699, 0.70410, 12.86952, 13.27349],
                #  [0.10742, 0.44824, 9.85950, 23.92382],
                #  [-0.04688, 0.66016, 6.08861, 24.04192],
                #  [0.44971, 0.67578, 11.78509, 19.26170],
                #  [0.58105, 0.27051, 3.80577, 23.88452],
                #  [0.47705, 0.63965, 12.37749, 9.12236]]
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]  # selected_tbox_aux.shape: [7, 4], 还原当前feature_map上的真实坐标
                # selected_tbox_aux: [[21.27874, 20.68615,  8.01240, 32.36983],
                #                 [29.44584, 20.56247,  8.44552, 32.30798],
                #                 [21.27874, 20.68615,  8.01240, 32.36983],
                #                 [21.27874, 20.68615,  8.01240, 32.36983],
                #                 [29.44584, 20.56247,  8.44552, 32.30798],
                #                 [21.27874, 20.68615,  8.01240, 32.36983],
                #                 [29.44584, 20.56247,  8.44552, 32.30798]]
                selected_tbox_aux[:, :2] -= grid_aux
                # selected_tbox_aux: [[0.27874, 0.68615, 8.01240, 32.36983],
                #  [0.44584, 0.56247, 8.44552, 32.30798],
                #  [1.27874, 0.68615, 8.01240, 32.36983],
                #  [0.27874, 1.68615, 8.01240, 32.36983],
                #  [0.44584, 1.56247, 8.44552, 32.30798],
                #  [0.27874, -0.31385, 8.01240, 32.36983],
                #  [0.44584, -0.43753, 8.44552, 32.30798]]
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target), 使用CIOU计算方式
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux.dtype)  # iou ratio, tobj_aux[b_aux, a_aux, gj_aux, gi_aux]: [0.60205, 0.32324, 0.59863, 0.56348, 0.47949, 0.35010, 0.21118]

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()  # 类别id, selected_tcls: [0, 0, 0, 0, 0, 0, 0]
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets, self.cn标签平滑
                    # t_aux: [[0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.],
                    #  [0., 0., 0., 0., 0.]]
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    # t_aux: [[1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.],
                    #  [1., 0., 0., 0., 0.]]
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE, 无fl_gamma为BCEWithLogitsLoss(),有fl_gamma为FocalLoss

            obji = self.BCEobj(pi[..., 4], tobj)  # tobj.shape: [1, 3, 160, 160], tobj[b, a, gj, gi]的值在上面被赋过值了, 无fl_gamma为BCEWithLogitsLoss(),有fl_gamma为FocalLoss
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)  # tobj_aux.shape: [1, 3, 160, 160], tobj_aux[b_aux, a_aux, gj_aux, gi_aux]的值在上面被赋过值了, 无fl_gamma为BCEWithLogitsLoss(),有fl_gamma为FocalLoss
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i] # obj loss
            if self.autobalance:  # False
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:  # False
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size, bs: 1

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        indices, anch = self.find_3_positive(p, targets)  # 一个目标匹配3个临近正样本，更精准的匹配

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]  # this_target: [[0.00000, 0.00000, 0.53197, 0.51715, 0.20031, 0.80925],[0.00000, 0.00000, 0.73615, 0.51406, 0.21114, 0.80770]]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]  # txywh: [[ 680.91974,  661.95691,  256.39685, 1035.83472],[ 942.26672,  657.99915,  270.25653, 1033.85522]]
            txyxy = xywh2xyxy(txywh)  #txyxy: [[ 552.72131,  144.03955,  809.11816, 1179.87427],[ 807.13843,  141.07153, 1077.39502, 1174.92676]]

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                # b: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # a: [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
                # gj: [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21]
                # gi: [21, 29, 21, 29, 21, 29, 20, 28, 20, 28, 20, 28, 21, 29, 21, 29, 21, 29]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                
                fg_pred = pi[b, a, gj, gi]  # fg_pred.shape: [18, 10], [18个正样本anchor，10个值xywh+5cls]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]  # (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid):还原当前feature_map上的坐标，* self.stride[i]:还原原图上的真实坐标
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]  # pxy.shape:[18, 2]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]  # pwh.shape:[18, 2]
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
            # txyxy真实目标坐标值，pxyxys预测坐标值
            pair_wise_iou = box_iou(txyxy, pxyxys)  #txyxy.shape:[2, 4],pxyxys.shape:[36, 4],pair_wise_iou.shape:[2, 36]

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)  # iou loss

            top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1)  # top_k.shape:[2, 20]， 选出前20个iou值最大的
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)  # top_k.sum(1): [5.97737, 6.06848], .int(): [5, 6], dynamic_ks: [5, 6]

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )  # gt_cls_per_image.shape: [2, 36, 5], gt_cls_per_image: [[[1., 0., 0., 0., 0.]x36], [[1., 0., 0., 0., 0.]x36]]

            num_gt = this_target.shape[0]  # num_gt: 2
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )  # p_cls.shape: [36, 5], p_obj.shape: [36, 1], cls_preds_.shape: [2, 36, 5], 2是指2个label目标，36个anchor, 5是类别数量, cls_preds_=cls*obj.repeat

            y = cls_preds_.sqrt_()  # 开根号
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)  # 计算cls_loss, pair_wise_cls_loss.shape: [2, 36]
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )  # 计算cost，cls:reg=1:3, cost.shape: [2, 60]

            matching_matrix = torch.zeros_like(cost)  # matching_matrix.shape: [2, 36]

            for gt_idx in range(num_gt):  # 选出前k个
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )  # cost[gt_idx].shape: [36], dynamic_ks[0].item(): 5, dynamic_ks[1].item(): 6
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)  # 计算每个anchor匹配上的gt数量, anchor_matching_gt.shape: [36]
            if (anchor_matching_gt > 1).sum() > 0:  # 每个anchor只能匹配一个gt，取cost值最小的为当前anchor匹配到的gt
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0  # fg_mask_inboxes.shape: [36]
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # anchor匹配上的gt index, matched_gt_inds.shape: 11
        
            from_which_layer = from_which_layer[fg_mask_inboxes]  # from_which_layer: [2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3.]
            all_b = all_b[fg_mask_inboxes]  # all_b.shape: [11]
            all_a = all_a[fg_mask_inboxes]  # all_a.shape: [11]
            all_gj = all_gj[fg_mask_inboxes]  # all_gj.shape: [11]
            all_gi = all_gi[fg_mask_inboxes]  # all_gi.shape: [11]
            all_anch = all_anch[fg_mask_inboxes]  # all_anch.shape: [11, 2]
        
            this_target = this_target[matched_gt_inds]  # this_target.shape: [11, 6]
        
            for i in range(nl):  # nl: 4
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
        # matching_bs: [[], [], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        # matching_as: [[], [], [2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0]]
        # matching_gjs: [[], [], [20, 20, 20, 20, 21, 21], [10, 10,  9, 10, 10]]
        # matching_gis: [[], [], [21, 29, 20, 28, 21, 29], [10, 14, 14, 11, 15]]
        # matching_targets: [[],
        #                    [],
        #                    [[0.00000, 0.00000, 0.53197, 0.51715, 0.20031, 0.80925],
        #                     [0.00000, 0.00000, 0.73615, 0.51406, 0.21114, 0.80770],
        #                     [0.00000, 0.00000, 0.53197, 0.51715, 0.20031, 0.80925],
        #                     [0.00000, 0.00000, 0.73615, 0.51406, 0.21114, 0.80770],
        #                     [0.00000, 0.00000, 0.53197, 0.51715, 0.20031, 0.80925],
        #                     [0.00000, 0.00000, 0.73615, 0.51406, 0.21114, 0.80770]],
        #                    [[0.00000, 0.00000, 0.53197, 0.51715, 0.20031, 0.80925],
        #                     [0.00000, 0.00000, 0.73615, 0.51406, 0.21114, 0.80770],
        #                     [0.00000, 0.00000, 0.73615, 0.51406, 0.21114, 0.80770],
        #                     [0.00000, 0.00000, 0.53197, 0.51715, 0.20031, 0.80925],
        #                     [0.00000, 0.00000, 0.73615, 0.51406, 0.21114, 0.80770]]]
        # matching_anchs: [[],
        #                  [],
        #                  [[7.4375, 16.938],
        #                   [7.4375, 16.938],
        #                   [7.4375, 16.938],
        #                   [7.4375, 16.938],
        #                   [7.4375, 16.938],
        #                   [7.4375, 16.938]],
        #                  [[6.8125, 9.6094],
        #                   [6.8125, 9.6094],
        #                   [6.8125, 9.6094],
        #                   [6.8125, 9.6094],
        #                   [6.8125, 9.6094]]]
        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def build_targets2(self, p, targets, imgs):
        
        indices, anch = self.find_5_positive(p, targets)  # 一个目标匹配5个临近正样本，粗略匹配

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    # nl: 4
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]  # this_target: [[0.00000, 0.00000, 0.58512, 0.27778, 0.59368, 0.33896]]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]  # txywh: [[748.95386, 355.56366, 759.90851, 433.87509]]
            txyxy = xywh2xyxy(txywh)  # txyxy: [[ 368.99960,  138.62611, 1128.90808,  572.50122]]

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p): # i:2, pi.shape:[1, 3, 40, 40, 10]
                
                b, a, gj, gi = indices[i]
                # b: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # a: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
                # gj: [11, 11, 11, 11, 10, 10, 11, 11, 12, 12]
                # gi: [23, 23, 22, 22, 23, 23, 24, 24, 23, 23]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                
                fg_pred = pi[b, a, gj, gi]  # fg_pred.shape: [10, 10], [10个正样本anchor，10个值xywh+5cls]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]  # (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid):还原当前feature_map上的坐标，* self.stride[i]:还原原图上的真实坐标
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]  # pxy.shape:[10, 2]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]  # pwh.shape:[10, 2]
                pxywh = torch.cat([pxy, pwh], dim=-1)  #
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            ###========================从这里的debug开始变成2个目标========================###
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
            # txyxy真实目标坐标值，pxyxys预测坐标值
            pair_wise_iou = box_iou(txyxy, pxyxys)  #txyxy.shape:[2, 4],pxyxys.shape:[60, 4],pair_wise_iou.shape:[2, 60]

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)  # iou loss

            top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1)  # top_k.shape:[2, 20]， 选出前20个iou值最大的
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)  # top_k.sum(1): [6.89685, 7.04802], .int(): [6, 7], dynamic_ks: [6, 7]

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )  # gt_cls_per_image.shape: [2, 60, 5], gt_cls_per_image: [[[1., 0., 0., 0., 0.]x60],[[1., 0., 0., 0., 0.]x60]]

            num_gt = this_target.shape[0]  # num_gt: 2
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )  # p_cls.shape: [60, 5], p_obj.shape: [60, 1], cls_preds_.shape: [2, 60, 5], 2是指2个label目标，60个anchor, 5是类别数量, cls_preds_=cls*obj.repeat

            y = cls_preds_.sqrt_()  # 开根号
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)  # 计算cls_loss, pair_wise_cls_loss.shape: [2, 60]
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )  # 计算cost，cls:reg=1:3, cost.shape: [2, 60]

            matching_matrix = torch.zeros_like(cost)  # matching_matrix.shape: [2, 60]

            for gt_idx in range(num_gt):  # 选出前k个
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )  # cost[gt_idx].shape: [60], dynamic_ks[0].item(): 6, dynamic_ks[1].item(): 7
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)  # 计算每个anchor匹配上的gt数量, anchor_matching_gt.shape: [60]
            if (anchor_matching_gt > 1).sum() > 0:  # 每个anchor只能匹配一个gt，取cost值最小的为当前anchor匹配到的gt
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0  # fg_mask_inboxes.shape: [60]
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # anchor匹配上的gt index, matched_gt_inds.shape: 13
        
            from_which_layer = from_which_layer[fg_mask_inboxes]  # from_which_layer: [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
            all_b = all_b[fg_mask_inboxes]  # all_b.shape: [13]
            all_a = all_a[fg_mask_inboxes]  # all_a.shape: [13]
            all_gj = all_gj[fg_mask_inboxes]  # all_gj.shape: [13]
            all_gi = all_gi[fg_mask_inboxes]  # all_gi.shape: [13]
            all_anch = all_anch[fg_mask_inboxes]  # all_anch.shape: [13, 2]
        
            this_target = this_target[matched_gt_inds]  # this_target.shape: [13, 6]
        
            for i in range(nl):  # nl: 4
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
        # matching_bs: [[], [], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        # matching_as: [[], [], [2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0]]
        # matching_gjs: [[], [], [20, 20, 20, 19, 19, 21, 21], [10, 10, 10, 9, 10, 10]]
        # matching_gis: [[], [], [21, 29, 20, 21, 29, 21, 29], [10, 14, 13, 14, 11, 15]]
        # matching_targets: [[],
        #                    [],
        #                    [[          0,           0,     0.53174,     0.51709,     0.20032,     0.80908],
        #                     [          0,           0,     0.73633,     0.51416,     0.21118,     0.80762],
        #                     [          0,           0,     0.53174,     0.51709,     0.20032,     0.80908],
        #                     [          0,           0,     0.53174,     0.51709,     0.20032,     0.80908],
        #                     [          0,           0,     0.73633,     0.51416,     0.21118,     0.80762],
        #                     [          0,           0,     0.53174,     0.51709,     0.20032,     0.80908],
        #                     [          0,           0,     0.73633,     0.51416,     0.21118,     0.80762]],
        #                    [[          0,           0,     0.53174,     0.51709,     0.20032,     0.80908],
        #                     [          0,           0,     0.73633,     0.51416,     0.21118,     0.80762],
        #                     [          0,           0,     0.73633,     0.51416,     0.21118,     0.80762],
        #                     [          0,           0,     0.73633,     0.51416,     0.21118,     0.80762],
        #                     [          0,           0,     0.53174,     0.51709,     0.20032,     0.80908],
        #                     [          0,           0,     0.73633,     0.51416,     0.21118,     0.80762]]]
        # matching_anchs: [[],
        #                  [],
        #                  [[7.4375, 16.938],
        #                   [7.4375, 16.938],
        #                  [7.4375, 16.938],
        #                  [7.4375, 16.938],
        #                  [7.4375, 16.938],
        #                  [7.4375, 16.938],
        #                  [7.4375, 16.938]],
        #                  [[6.8125, 9.6094],
        #                   [6.8125, 9.6094],
        #                   [6.8125, 9.6094],
        #                   [6.8125, 9.6094],
        #                   [6.8125, 9.6094],
        #                   [6.8125, 9.6094]]]
        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs              

    def find_5_positive(self, p, targets):
        # 输出结果：
        # - indices: [(b,a,gj,gi) * 4]，yolov7-w6有4个head输出，所以indices里面有4个列表,
        #             b里面对应的是哪个batch_size，a里面对应都是哪个head层的anchor，
        #             gj对应的是匹配到的anchor的y坐标值(int)，gi对应的是匹配到的anchor的x坐标值(int)
        # - anch：[(anchorwh) * 4]，yolov7-w6有4个head输出，所以anch里面有4个列表,
        #           anchorhw对应的是匹配到的anchor[anchorh, anchow]

        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets, na: 3, nt: 1
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain, gain: [1, 1, 1, 1, 1, 1, 1]
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt), ai: [[0.0], [1.0], [2.0]]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, targets: [[[0.0, 0.0, 0.585, 0.2778, 0.5938, 0.3389, 0.0]], [[0.0, 0.0, 0.585, 0.2778, 0.5938, 0.3389, 1.0]], [[0.0, 0.0, 0.585, 0.2778, 0.5938, 0.3389, 2.0]]]

        g = 1.0  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets , 一个目标同时匹配5个正样本

        for i in range(self.nl): # self.anchors: [[[2.375, 3.375], [5.5, 5.0], [4.75, 11.75]], [[6.0, 4.25], [5.375, 9.5], [11.25, 8.5625]], [[4.375, 9.40625], [9.46875, 8.25], [7.4375, 16.9375]], [[6.8125, 9.609375], [11.546875, 5.9375], [14.453125, 12.375]]]
            anchors = self.anchors[i]  # anchors: [[ 2.37500,  3.37500],[ 5.50000,  5.00000],[ 4.75000, 11.75000]]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain, gain: [  1,   1, 160, 160, 160, 160,   1]

            # Match targets to anchors, 还原当前feature_map下的真实坐标值
            t = targets * gain # t:[[[0.0, 0.0, 93.61923, 44.445457, 94.98856, 54.234386, 0.0]], [[0.0, 0.0, 93.61923, 44.445457, 94.98856, 54.234386, 1.0]], [[0.0, 0.0, 93.61923, 44.445457, 94.98856, 54.234386, 2.0]]]
            if nt:
                # Matches，计算wh与anchors_wh的比值，如果都小于4，则判定当前anchor尺寸存在正样本
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare, self.hyp['anchor_t']:4
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets, g: 1.0
                gxy = t[:, 2:4]  # grid xy， gxy: [[23.40481, 11.11136],[23.40481, 11.11136]]
                gxi = gain[[2, 3]] - gxy  # inverse , gxi: [[16.59519, 28.88864],[16.59519, 28.88864]]
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T  # j, k: [True, True], [True, True]
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T  # l, m: [True, True], [True, True]
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # j: [[True, True],[True, True],[True, True],[True, True],[True, True]]
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # offsets: [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [0.0, -1.0]]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class , b: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], c: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            gxy = t[:, 2:4]  # grid xy , gxy: [[23.40481, 11.11136] x 10]
            gwh = t[:, 4:6]  # grid wh, gwh: [[23.74714, 13.55860] x 10]
            gij = (gxy - offsets).long() # gij: [[23, 11], [23, 11], [22, 11], [22, 11], [23, 10], [23, 10], [24, 11], [24, 11], [23, 12], [23, 12]]
            gi, gj = gij.T  # grid xy indices, gi: [23, 23, 22, 22, 23, 23, 24, 24, 23, 23], gj: [11, 11, 11, 11, 10, 10, 11, 11, 12, 12]

            # Append
            a = t[:, 6].long()  # anchor indices, a: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices, # .clamp_(min,max)压缩到(min,max)区间
            anch.append(anchors[a])  # anchors
        # indices: [ ( [], [], [], [] ),  # 一个列表四个值(b,a,gj,gi), list0.shape:[[0],[0],[0],[0]]
        #          ( [], [], [], [] ),  # list1.shape:[[0],[0],[0],[0]]
        #          ( [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        #          [11, 11, 11, 11, 10, 10, 11, 11, 12, 12], [23, 23, 22, 22, 23, 23, 24, 24, 23, 23] ),  # list2.shape:[[10],[10],[10],[10]]
        #          ( [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        #          [5, 5, 5, 5, 5, 5, 4, 4, 4, 5, 5, 5, 6, 6, 6], [11, 11, 11, 10, 10, 10, 11, 11, 11, 12, 12, 12, 11, 11, 11] ) ]  # list3.shape:[[15],[15],[15],[15]]
        # anch: [ ( [] ),
        #       ( [] ),
        #       ( [[ 9.46875,  8.25000],[ 7.43750, 16.93750],[ 9.46875,  8.25000],[ 7.43750, 16.93750],[ 9.46875,  8.25000],
        #         [ 7.43750, 16.93750],[ 9.46875,  8.25000],[ 7.43750, 16.93750],[ 9.46875,  8.25000],[ 7.43750, 16.93750]] ),
        #       ( [[ 6.81250,  9.60938],[11.54688,  5.93750],[14.45312, 12.37500],[ 6.81250,  9.60938],[11.54688,  5.93750],
        #         [14.45312, 12.37500],[ 6.81250,  9.60938],[11.54688,  5.93750],[14.45312, 12.37500],[ 6.81250,  9.60938],
        #         [11.54688,  5.93750],[14.45312, 12.37500],[ 6.81250,  9.60938],[11.54688,  5.93750],[14.45312, 12.37500]] ) ]
        return indices, anch                 

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        # 与find_5_positive唯一的区别在于这个g，影响到下面的取值范围off
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors
        # indices: [ (tensor([], device='cuda:0', dtype=torch.int64), tensor([], device='cuda:0', dtype=torch.int64),
        #  tensor([], device='cuda:0', dtype=torch.int64), tensor([], device='cuda:0', dtype=torch.int64)
        # ),
        # (tensor([], device='cuda:0', dtype=torch.int64), tensor([], device='cuda:0', dtype=torch.int64),
        #  tensor([], device='cuda:0', dtype=torch.int64), tensor([], device='cuda:0', dtype=torch.int64)
        # ),
        # (tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0'),
        #  tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2], device='cuda:0'),
        #  tensor([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21], device='cuda:0'),
        #  tensor([21, 29, 21, 29, 21, 29, 20, 28, 20, 28, 20, 28, 21, 29, 21, 29, 21, 29], device='cuda:0')
        # ),
        # (tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0'),
        #  tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2], device='cuda:0'),
        #  tensor([10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10], device='cuda:0'),
        #  tensor([10, 14, 10, 14, 10, 14, 10, 14, 10, 14, 10, 14, 11, 15, 11, 15, 11, 15], device='cuda:0')
        # ) ]
        # anch: [tensor([], device='cuda:0', size=(0, 2)),
        #        tensor([], device='cuda:0', size=(0, 2)),
        #        tensor([[ 4.37500,  9.40625],[ 4.37500,  9.40625],[ 9.46875,  8.25000],
        #                [ 9.46875,  8.25000],[ 7.43750, 16.93750],[ 7.43750, 16.93750],
        #                [ 4.37500,  9.40625],[ 4.37500,  9.40625],[ 9.46875,  8.25000],
        #                [ 9.46875,  8.25000],[ 7.43750, 16.93750],[ 7.43750, 16.93750],
        #                [ 4.37500,  9.40625],[ 4.37500,  9.40625],[ 9.46875,  8.25000],
        #                [ 9.46875,  8.25000],[ 7.43750, 16.93750],[ 7.43750, 16.93750]], device='cuda:0'),
        #        tensor([[ 6.81250,  9.60938],[ 6.81250,  9.60938],[11.54688,  5.93750],
        #             [11.54688,  5.93750],[14.45312, 12.37500],[14.45312, 12.37500],
        #             [ 6.81250,  9.60938],[ 6.81250,  9.60938],[11.54688,  5.93750],
        #             [11.54688,  5.93750],[14.45312, 12.37500],[14.45312, 12.37500],
        #             [ 6.81250,  9.60938],[ 6.81250,  9.60938],[11.54688,  5.93750],
        #             [11.54688,  5.93750],[14.45312, 12.37500],[14.45312, 12.37500]], device='cuda:0')]
        return indices, anch
