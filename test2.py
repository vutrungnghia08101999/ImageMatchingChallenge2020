import torch.nn as nn
import torch

class ContrastiveLoss(nn.Module):
    """
    an implementation of Constrastive loss from the paper:
    https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temp=1., reduction="mean"):
        """
        args:
            --temp(float): logit temperature
            --reduction: either None, "mean", "sum"
        """
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        if reduction is not None:
            assert reduction in ["mean", "sum"], \
                f"expect 'mean' or 'sum' for reduction, got {reduction}"
        self.reduction = reduction    
        
    def forward(self, logits, targets):
        """
        args: 
            --logits: L2-normalized logits of shape (N,D)
            --targets: gt of shape (N,)
        """
        # normalize:
        norm = torch.norm(logits, p=2, dim=1, keepdim=True)#.detach()
        logits = logits/(norm + 1e-5)
        # print("logits", logits)
        gt_matrix = self._convert_target(logits, targets) # (N,N)
        # print(gt_matrix)
        pair_wise_dis = (logits*logits[:,None,:]).sum(dim=2) / self.temp # (N,N)
        pair_wise_dis.fill_diagonal_(0.)
        print(pair_wise_dis)
        n_log_softmax = - torch.log_softmax(pair_wise_dis, dim=1)
        print(n_log_softmax)
        loss_per_sample = (n_log_softmax*gt_matrix).sum(dim=1) / (gt_matrix.sum(dim=1) + 1) 
        print(loss_per_sample)
        if self.reduction is None:
            return loss_per_sample
        elif self.reduction == "mean":
            return loss_per_sample.mean()
        else:
            return loss_per_sample.sum()
    
    def _convert_target(self, logits, targets):
        """
        convert targets of shape (N,) to matrix A of shape (N,N)
        where A[i,j] = 1 if targets[i] == targets[j] for all i != j
        """
        
        diff = (targets - targets[:,None]) == 0
        diff = diff.type_as(logits)
        diff.fill_diagonal_(0.)
        return diff
        
if __name__ == "__main__":
    contrastive_loss = ContrastiveLoss()
    logits = torch.tensor([
        [1., 0.],
        [0.6,0.8],
        [5./13, 12./13],
        [0., 1.],
        [.8, .6]
    ])
    logits.requires_grad = True
    gt = torch.tensor([1,2,2,1,1])
    gt.requires_grad = False
    loss = contrastive_loss(logits, gt)
    for i in range(2000):
        loss = contrastive_loss(logits, gt)
        loss.backward()
        with torch.no_grad():
            logits = (logits - 0.01*logits.grad)
            # logits_ = torch.zeros_like(logits)
            
            # logits_.copy_(logits)
            # logits_.requires_grad = True
            # logits = logits_
            logits.requires_grad = True
            # logits.grad.data.zero_()
            # print(logits.requires_grad)
            # print(logits.grad)
        # break
    print(logits)
    # print(logits.grad)