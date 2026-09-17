import torch
import math

class SHDiffuseCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.degree = 2
        
        self.cosine_lobe_factors = torch.tensor([
            1.0,                  # l=0
            2.0 / 3.0,            # l=1 (3 terms)
            2.0 / 3.0, 
            2.0 / 3.0,
            1.0 / 4.0,            # l=2 (5 terms)
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0
        ], device=self.device, dtype=torch.float32)

    def get_sh_basis(self, dirs):
        x = dirs[..., 0]
        y = dirs[..., 1]
        z = dirs[..., 2]
        
        basis = torch.zeros((dirs.shape[0], 9), device=dirs.device, dtype=dirs.dtype)
        
        # l=0
        basis[:, 0] = 0.282095
        
        # l=1
        basis[:, 1] = 0.488603 * y
        basis[:, 2] = 0.488603 * z
        basis[:, 3] = 0.488603 * x
        
        # l=2
        basis[:, 4] = 1.092548 * x * y
        basis[:, 5] = 1.092548 * y * z
        basis[:, 6] = 0.315392 * (3 * z * z - 1)
        basis[:, 7] = 1.092548 * x * z
        basis[:, 8] = 0.546274 * (x * x - y * y)
        
        return basis

    def project_environment_to_sh(self, envmap, num_samples=100000):
        """
        Envmap: [H, W, 3] linear RGB
        """
        H, W, C = envmap.shape
        
        # Fibonacci Sphere Sampling
        idx = torch.arange(0, num_samples, dtype=torch.float32, device=self.device)
        offset = 2.0 / num_samples
        increment = math.pi * (3.0 - math.sqrt(5.0))
        
        # Coordinate System: Z-up for SH formula
        # z goes from 1 (top) to -1 (bottom)
        z = ((idx * offset) - 1) + (offset / 2) 
        r = torch.sqrt(1 - z**2)
        phi = (idx * increment) % (2 * math.pi)
        
        x = torch.cos(phi) * r
        y = torch.sin(phi) * r
        
        dirs = torch.stack([x, y, z], dim=1) # [N, 3]
        
        theta = torch.acos(z) # [0, pi]
        
        # phi from atan2 is [-pi, pi], map to [0, 1]
        raw_phi = torch.atan2(y, x) 
        u = (raw_phi / (2 * math.pi)) + 0.5
        v = theta / math.pi
        
        # Bilinear sampling or Nearest for speed (High sample count makes nearest acceptable)
        px_u = (u * W).long().clamp(0, W-1)
        px_v = (v * H).long().clamp(0, H-1)
        
        L = envmap[px_v, px_u] # [N, 3]
        
        # Calculate Basis
        basis = self.get_sh_basis(dirs) # [N, 9]
        
        # Integral: sum(L * Y) * 4pi/N
        coeffs = torch.einsum('n k, n c -> k c', basis, L)
        coeffs *= (4.0 * math.pi / num_samples)
        
        # Apply Convolution (Diffuse Factors)
        coeffs *= self.cosine_lobe_factors.unsqueeze(1)
        
        return coeffs

    def reconstruct_diffuse_map(self, sh_coeffs, H, W):
        # Generate Output Grid
        v_grid, u_grid = torch.meshgrid(
            torch.linspace(0, 1, H, device=self.device),
            torch.linspace(0, 1, W, device=self.device),
            indexing='ij'
        )
        
        # UV to Direction (Z-up standard for SH)
        theta = v_grid * math.pi
        phi = (u_grid - 0.5) * 2 * math.pi
        
        # Note: Coordinate mapping must match projection
        # x = cos(phi)sin(theta)
        # y = sin(phi)sin(theta)
        # z = cos(theta)
        sin_theta = torch.sin(theta)
        x = torch.cos(phi) * sin_theta
        y = torch.sin(phi) * sin_theta
        z = torch.cos(theta)
        
        dirs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        
        # Reconstruct
        basis = self.get_sh_basis(dirs)
        diffuse = torch.mm(basis, sh_coeffs)
        
        return diffuse.view(H, W, 3)

def compute_diffuse_panorama(envmap_tensor):
    """
    Input: [H, W, 3] Tensor, Range [0, whatever], Linear Space
    Output: [H, W, 3] Tensor
    """
    calc = SHDiffuseCalculator(device=envmap_tensor.device)
    
    # 1. Project
    sh_coeffs = calc.project_environment_to_sh(envmap_tensor)
    
    # 2. Reconstruct
    H, W = envmap_tensor.shape[:2]
    diffuse = calc.reconstruct_diffuse_map(sh_coeffs, H, W)
    
    # Clamp negative ringing artifacts (SH property)
    diffuse = torch.max(diffuse, torch.zeros_like(diffuse))
    
    return diffuse
