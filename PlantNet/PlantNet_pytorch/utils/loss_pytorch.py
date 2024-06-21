import torch



def discriminative_loss_single(prediction, correct_label, feature_dim,
                   delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Discriminative loss for a single prediction/label pair.
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''

    ### Reshape so pixels are aligned along a vector
    reshaped_pred = prediction.reshape([-1, feature_dim])
    ### Count instances,计算各元素出现idx和次数,a=[1,2,2,1]调用return_inverse则返回[0,1,1,0]
    unique_labels, unique_id, counts = torch.unique(correct_label, return_inverse=True, return_counts=True)
    counts = counts.float()
    num_instances = unique_labels.shape[0]
    #将不同索引值对应元素求和
    segmented_sum = torch.zeros(num_instances, feature_dim).to(prediction.device)
    for i in range(num_instances):
        mask = (correct_label == unique_labels[i])
        masked_pred = reshaped_pred[mask]
        sum_pred = torch.sum(masked_pred, dim=0)
        segmented_sum[i] = sum_pred

    mu = segmented_sum / counts.view(-1, 1)
    mu_expand = mu[unique_id]#（4096,feather_dim）

    ### Calculate l_var
    tmp_distance = reshaped_pred - mu_expand
    distance = torch.norm(tmp_distance, p=1, dim=1)
    distance = distance - delta_v
    distance = torch.clamp(distance, min=0.)
    distance = distance ** 2
    l_var = torch.zeros(num_instances, device=prediction.device)
    for i in range(num_instances):
        mask = (unique_id == i)
        l_var[i] = torch.sum(distance[mask]) / counts[i]
    l_var = torch.sum(l_var) / num_instances

    ### Calculate l_dist
    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3
    mu_interleaved_rep = mu.repeat(num_instances, 1)
    mu_band_rep = mu.repeat(1, num_instances).view(num_instances * num_instances, feature_dim)
    mu_diff = mu_band_rep - mu_interleaved_rep
    # Filter out zeros from same cluster subtraction
    diff_cluster_mask = torch.eye(num_instances).eq(0).view(-1)
    mu_diff_bool = mu_diff[diff_cluster_mask]
    mu_norm = torch.norm(mu_diff_bool, p=1, dim=1)
    mu_norm = 2. * delta_d - mu_norm
    mu_norm = torch.clamp(mu_norm, min=0.)
    mu_norm = mu_norm ** 2
    l_dist = mu_norm.mean()
    def rt_0():
        return torch.tensor(0.)

    def rt_l_dist():
        return l_dist

    l_dist = torch.where(torch.eq(torch.tensor(1.), num_instances), rt_0(), rt_l_dist())


    # Calculate l_reg
    l_reg = torch.norm(mu, p=1, dim=1).mean()

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss, l_var, l_dist, l_reg

def discriminative_loss(prediction, correct_label, feature_dim,
            delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Iterate over a batch of prediction/label and cumulate loss
    :return: discriminative loss and its three components
    '''

    batchsize=prediction.shape[0]
    output_loss = []
    output_var = []
    output_dist = []
    output_reg = []
    for i in range(batchsize):
       disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction[i], correct_label[i], feature_dim,delta_v, delta_d, param_var, param_dist, param_reg)
       output_loss.append(disc_loss)
       output_var.append(l_var)
       output_dist.append(l_dist)
       output_reg.append(l_reg)

    disc_loss = torch.mean(torch.stack(output_loss))
    l_var = torch.mean(torch.stack(output_var))
    l_dist = torch.mean(torch.stack(output_dist))
    l_reg = torch.mean(torch.stack(output_reg))

    return disc_loss, l_var, l_dist, l_reg