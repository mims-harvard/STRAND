import torch, math
import numpy as np

def cosine_beta_schedule(T, s=0.008):
    steps = np.arange(T + 1)
    f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f(steps) / f(0)
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = 1 - betas
    return torch.tensor(betas, dtype=torch.float32), torch.tensor(alphas,dtype=torch.float32), torch.tensor(alphas_cumprod, dtype=torch.float32),


def make_beta_schedule(n_timestep=1000, linear_start=1e-3, linear_end=1e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep) ** 2
    )
    
    # # print(f"betas: {betas.shape}")
    # betas = np.concatenate([betas[:n_timestep//2], np.flip(betas[:n_timestep//2])])
    
    alphas = 1 - betas
    alphas_cumprod   = np.cumprod(alphas, axis=0)
    # alphas_cumprod = np.concatenate([[1.0], alphas_cumprod]) 
    return torch.tensor(betas, dtype=torch.float32), torch.tensor(alphas,dtype=torch.float32), torch.tensor(alphas_cumprod, dtype=torch.float32),



def symmetric_beta(n_timestep=1000, linear_start=1e-4, beta_max=3.):
    
    assert n_timestep % 2 == 0, "n_timestep must be even"
    linear_end = beta_max / n_timestep
    # mid-point rule (centres of cells) gives lower discretisation error
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        np.linspace((linear_start) ** 0.5, linear_end ** 0.5, n_timestep//2, dtype=np.float64) ** 2
    ) ## control status from slight noisier status
    
    flip_betas = np.flip((
        np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep//2, dtype=np.float64) ** 2
    ))
    
    # # print(f"betas: {betas.shape}")
    betas = np.concatenate([betas, flip_betas])
    
    alphas = 1 - betas
    alphas_cumprod   = np.cumprod(alphas, axis=0)
    # alphas_cumprod = np.concatenate([[1.0], alphas_cumprod]) 
    return torch.tensor(betas, dtype=torch.float32), torch.tensor(alphas,dtype=torch.float32), torch.tensor(alphas_cumprod, dtype=torch.float32),



def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    """
    Generates the initial full beta schedule.
    """
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def get_official_betas(n_timestep=1000, linear_start=5e-5, beta_max=10.):
    """
    Implements the logic from the Runner class to get the final symmetric schedule.
    """
    # Calculate the linear end value
    linear_end = beta_max / n_timestep
    
    # 1. Create a FULL schedule with n_timestep points
    betas = make_beta_schedule(n_timestep=n_timestep, linear_start=linear_start, linear_end=linear_end)
    
    # 2. Take the first half and concatenate it with its flipped version
    final_betas = np.concatenate([betas[:n_timestep//2], np.flip(betas[:n_timestep//2])])
    final_betas = torch.tensor(final_betas, dtype=torch.float32)
    alphas = 1 - final_betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return final_betas, alphas, alphas_cumprod

def symmetric_sin_beta(n_timestep=1000, linear_start=1e-4, beta_max=50.):
    linear_end=beta_max/n_timestep
    # Use sine to smoothly increase from start to end over half the steps
    t_half = np.linspace(0, np.pi / 2, n_timestep // 2)
    beta_half = linear_start + (linear_end - linear_start) * np.sin(t_half)
    
    # Symmetrically mirror the first half
    betas = np.concatenate([beta_half, beta_half[::-1]])
    
    # Calculate alphas and cumulative product
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    
    return (torch.tensor(betas, dtype=torch.float32), 
            torch.tensor(alphas, dtype=torch.float32), 
            torch.tensor(alphas_cumprod, dtype=torch.float32))
    
    
    
def _safe_norm_pos(betas, eps=1e-12):
    """Normalize betas along time to sum to 1, clamp tiny negatives."""
    betas = betas.clamp_min(eps)
    return betas / (betas.sum(dim=-1, keepdim=True) + eps)

def _mirror_half(half, T):
    """
    Given a (K, L) half-curve of betas, mirror to length T.
    If T is even: [half, flip(half)]
    If T is odd:  [half, center, flip(half)], with center = last(half)
    """
    K, L = half.shape
    if T % 2 == 0:
        return torch.cat([half, torch.flip(half, dims=[1])], dim=1)  # (K, 2L)
    else:
        center = half[:, -1:].clone()
        return torch.cat([half, center, torch.flip(half, dims=[1])], dim=1)  # (K, 2L+1)

def make_symmetric_beta_bases(T, K=4, device=None, dtype=torch.float32):
    """
    Build K symmetric beta bases. Each base is defined on the first half, then mirrored.
    The shapes differ on the first half (linear, concave, convex, cosine-like, etc.).
    """
    half_len = T // 2 if T % 2 == 0 else (T // 2)  # we'll add a center step if odd
    # use a fine grid for the half (avoid degenerate 0 length)
    L = max(1, half_len)
    th = torch.linspace(0.0, 1.0, L, device=device, dtype=dtype)  # [0,1] on first half

    # half-side prototypes (positive on [0,1], different curvature)
    # 1) linear rise
    h_lin   = th
    # 2) concave-up (slow start, fast end in the half)
    h_sq    = th**2
    # 3) concave-down (fast start, slow end)
    h_sqrt  = th.clamp_min(1e-8).sqrt()
    # 4) cosine-bump concentrated near the mid (on the half)
    h_cos   = torch.sin(0.5*torch.pi*(th + 0.5/L)).pow(2)

    H = torch.stack([h_lin, h_sq, h_sqrt, h_cos], dim=0)[:K]               # (K, L)
    # convert half *slopes* to half *betas* by finite difference of a monotone cum (or just use the curve itself)
    half_betas = _safe_norm_pos(H + 1e-8)                                   # (K, L)

    # mirror to T
    betas_full = _mirror_half(half_betas, T)                                # (K, T)
    # re-normalize to ensure sum_t beta = 1 precisely
    betas_full = _safe_norm_pos(betas_full)
    return betas_full  # (K, T)
    
    
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_epoch   = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model, epoch=-1):
        if self.best_loss is None or self.best_loss - val_loss > self.min_delta:
            self.best_loss   = val_loss
            self.best_epoch  = epoch
            self.counter     = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = {k: v.detach().clone()
                             for k, v in model.state_dict().items()}