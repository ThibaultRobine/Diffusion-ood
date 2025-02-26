import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from typing import Any, List, Tuple

from openood.postprocessors.base_postprocessor import BasePostprocessor
from improved_diffusion import logger
from improved_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LatentDiffIntpostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = config.postprocessor.postprocessor_args
        
        # Feature extraction parameters
        self.feature_type = self.args.feature_type
        
        # Diffusion parameters
        self.diffusion_config = self._get_diffusion_config()
        self.latent_dim = None  # Determined during setup
        
        # Reference point parameters
        self.kmeans_k = self.args.kmeans_k
        self.optim_steps = self.args.optim_steps
        self.optim_lr = self.args.optim_lr
        
        # Quadrature parameters
        self.n_quadrature_points = self.args.n_quadrature_points
        
        # Models and data
        self.model, self.diffusion = None, None
        self.reference_points = None
        self.scaler = None
        self.kde = None

    def _get_diffusion_config(self):
        """Create diffusion configuration with dynamic dimensions"""
        defaults = model_and_diffusion_defaults()
        return dict(
            image_size=32,  # Placeholder, updated in setup
            num_channels=self.args.num_channels,
            num_res_blocks=self.args.num_res_blocks,
            learn_sigma=self.args.learn_sigma,
            attention_resolutions=self.args.attention_resolutions,
            num_heads=self.args.num_heads,
            num_heads_upsample=self.args.num_heads_upsample,
            use_scale_shift_norm=self.args.use_scale_shift_norm,
            dropout=self.args.dropout,
            diffusion_steps=self.args.diffusion_steps,
            noise_schedule=self.args.noise_schedule,
            timestep_respacing=self.args.timestep_respacing,
            use_kl=self.args.use_kl,
            predict_xstart=self.args.predict_xstart,
            rescale_timesteps=self.args.rescale_timesteps,
            rescale_learned_sigmas=self.args.rescale_learned_sigmas,
        )

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # Extract features from classifier
        features, _ = self._extract_features(net, id_loader_dict['train'])
        
        
        self.latent_dim = features.shape[1]
        self._update_diffusion_config()
        
        # Convert to tensor
        self.train_data = torch.tensor(features, dtype=torch.float32, device=device)
        
        # Train diffusion model on latent features
        self._init_diffusion_model()
        self._train_diffusion_model()
        
        # Get optimized reference points
        self.reference_points = self._get_optimized_reference_points()
        
        # Compute integral scores for KDE training
        id_scores = self._compute_integrals(self.train_data)
        
        # Train KDE
        self.scaler, self.kde = self._train_kde(id_scores)

    def _extract_features(self, net: nn.Module, loader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from classifier network"""
        net.eval()
        features, labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting features"):
                data = batch['data'].to(device)
                label = batch['label'].cpu().numpy()
                
                # Forward pass with feature return
                if hasattr(net, 'forward_features'):
                    feat = net.forward_features(data)
                else:
                    _, feat = net(data, return_feature=True)
                
                features.append(feat.cpu().numpy())
                labels.append(label)
        
        return np.concatenate(features), np.concatenate(labels)


    def _update_diffusion_config(self):
        """Update diffusion config based on latent dimension"""
        # Treat latent dimension as 1D "image"
        self.diffusion_config.update({
            'image_size': self.latent_dim,
            'in_channels': 1,  # Treat as 1-channel 1D "image"
            'num_channels': self.args.num_channels,
        })

    def _init_diffusion_model(self):
        """Initialize diffusion model for 1D latent features"""
        logger.configure()
        self.model, self.diffusion = create_model_and_diffusion(
            **self.diffusion_config
        )
        self.model.to(device)
        self.model.train()

    def _train_diffusion_model(self):
        """Train diffusion model on latent features"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # Convert features to "1D images" [B, 1, D]
        data = self.train_data.unsqueeze(1)  # Add channel dimension
        
        for epoch in range(self.args.epochs):
            permutation = torch.randperm(len(data))
            for i in tqdm(range(0, len(data), self.args.batch_size),
                         desc=f"Training Epoch {epoch+1}"):
                batch_idx = permutation[i:i+self.args.batch_size]
                batch = data[batch_idx]
                
                # Sample timesteps
                t = torch.randint(0, self.diffusion.num_timesteps, 
                                (len(batch),), device=device)
                
                # Compute loss
                loss = self.diffusion.training_losses(
                    model=self.model,
                    x_start=batch,
                    t=t,
                    model_kwargs={}
                )['loss'].mean()
                
                # Optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        self.model.eval()

    def _get_optimized_reference_points(self):
        """Find and optimize reference points in latent space"""
        # K-means initialization
        kmeans = KMeans(n_clusters=self.kmeans_k, n_init=10)
        kmeans.fit(self.train_data.cpu().numpy())
        x_init = torch.tensor(kmeans.cluster_centers_, 
                            dtype=torch.float32, device=device)
        
        return self._optimize_reference_points(x_init)

    def _optimize_reference_points(self, x_init: torch.Tensor):
        """Gradient-based optimization in latent space"""
        x = nn.Parameter(x_init.clone())
        optimizer = torch.optim.Adam([x], lr=self.optim_lr)
        
        for _ in tqdm(range(self.optim_steps), desc="Reference Optimization"):
            optimizer.zero_grad()
            
            # Format as 1D "images"
            x_ = x.unsqueeze(1)  # [K, 1, D]
            
            # Compute score
            t = torch.zeros(len(x_), device=device, dtype=torch.long)
            noise = torch.randn_like(x_)
            x_t = self.diffusion.q_sample(x_, t, noise=noise)
            model_output = self.model(x_t, self.diffusion._scale_timesteps(t))
            
            score = (noise - model_output) / self.diffusion.sqrt_one_minus_alphas_cumprod[t]
            
            # Manual gradient assignment
            x.grad = -score.squeeze(1).sum(dim=0)
            optimizer.step()
            
        return x.detach()

    def _compute_integrals(self, data: torch.Tensor):
        """Compute Gaussian quadrature integrals"""
        scores = []
        for x_ref in tqdm(self.reference_points, desc="Reference Points"):
            scores.append(self._gaussian_quadrature(data, x_ref))
        return torch.stack(scores, dim=1).cpu().numpy()

    def _gaussian_quadrature(self, x: torch.Tensor, x_ref: torch.Tensor):
        """Compute line integral in latent space"""
        points, weights = self._get_gauss_legendre_params(self.n_quadrature_points)
        integral = torch.zeros(len(x), device=device)
        
        for t, w in zip(points, weights):
            t_scaled = 0.5 * (t + 1)  # Map to [0,1]
            z = x * (1 - t_scaled) + x_ref.unsqueeze(0) * t_scaled
            
            # Compute score at z
            with torch.no_grad():
                t_batch = torch.zeros(len(z), device=device, dtype=torch.long)
                noise = torch.randn_like(z.unsqueeze(1))
                z_t = self.diffusion.q_sample(z.unsqueeze(1), t_batch, noise=noise)
                model_output = self.model(z_t, self.diffusion._scale_timesteps(t_batch))
                
                score = (noise - model_output) / self.diffusion.sqrt_one_minus_alphas_cumprod[t_batch]
                score = score.squeeze(1)
            
            directional = torch.sum(score * (x_ref - x), dim=1)
            integral += w * directional
        
        return -0.5 * integral

    # _train_kde, _get_gauss_legendre_params same as previous implementation

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """Compute OOD scores through latent space"""
        # Extract features
        if hasattr(net, 'forward_features'):
            features = net.forward_features(data)
        else:
            _, features = net(data, return_feature=True)
        
        
        # Compute integral scores
        features = features.to(device)
        scores = []
        for x_ref in self.reference_points:
            scores.append(self._gaussian_quadrature(features, x_ref).cpu().numpy())
        scores = np.stack(scores, axis=1)
        
        # Score with KDE
        scaled_scores = self.scaler.transform(scores)
        log_probs = self.kde.score_samples(scaled_scores)
        
        return (
            torch.zeros(len(data), dtype=torch.long, device=device),
            torch.tensor(-log_probs, device=device)
        )