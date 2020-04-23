def pairwise_dist(kps1, kps2):
    """
    compute pairwise square of l2 distance
    
    args:
        kps1: tensor of shape (N,2)
        kps2: tensor of shape (M,2)
        
    return:
        tensor of shape (N,M)
    """
    diff = kps1[:,None,:] - kps2
    return (diff*diff).sum(dim=2)


def match(gt_kps, pred_kps):
    """
    match nearest predicted keypoint for each groundtruth keypoint
    
    return index of matched pred kp for each gt kp
    """
    dist = pairwise_dist(gt_kps, pred_kps)
    min_dist, idxs = dist.min(dim=1)
    
    # print(len(gt_kps), len(pred_kps))
    # diff = gt_kps[:,None,:] - pred_kps
    # diff = (diff*diff).sum(dim=2)
    # min_dist, idxs = diff.min(dim=1)
    
    return idxs.cpu().numpy(), min_dist.cpu().numpy()
    # return dist.argmin(dim=1) 
    
    
if __name__ == "__main__":
    import torch
    a = torch.tensor([[1,2],[3,4],[5,6]])
    b = torch.tensor([[1,1],[2,2]])
    print(pairwise_dist(a,b))
    print(match(a,b))