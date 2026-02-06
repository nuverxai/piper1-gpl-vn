import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l_dg = torch.mean((1 - dg) ** 2)
        gen_losses.append(l_dg)
        loss += l_dg

    return loss, gen_losses

def flow_matching_loss(v_pred, x_1, x_0, mask):
    """
    Conditional Flow Matching Loss (Eq. 1 in Supertonic/Flow Matching papers).
    
    Replaces KL Divergence. instead of maximizing likelihood via VAE, 
    we minimize the regression error of the vector field.
    
    Assumption: Optimal Transport (OT) path is used.
    Target Vector Field u_t(x|x_1) = x_1 - x_0
    
    Args:
        v_pred: Predicted velocity/vector field from the model [b, channels, time]
        x_1: Target data (Posterior Latent / Spectrogram) [b, channels, time]
        x_0: Source noise (Standard Gaussian) [b, channels, time]
        mask: Sequence mask to handle padding [b, 1, time]
    """
    v_pred = v_pred.float()
    x_1 = x_1.float()
    x_0 = x_0.float()
    mask = mask.float()

    # Target velocity for Optimal Transport path is simply (Target - Source)
    target_v = x_1 - x_0
    
    # Calculate MSE between predicted velocity and target velocity
    # loss = || v_pred - (x_1 - x_0) ||^2
    loss = (v_pred - target_v) ** 2
    
    # Apply mask and normalize by the number of valid elements
    loss = torch.sum(loss * mask) / torch.sum(mask)
    
    return loss

# not use it in new version
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l_kl = kl / torch.sum(z_mask)
    return l_kl
