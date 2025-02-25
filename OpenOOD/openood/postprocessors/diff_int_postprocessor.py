import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from typing import Any, List

from openood.postprocessors.base_postprocessor import BasePostprocessor
from improved_diffusion import logger
from improved_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDiffusionPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = config.postprocessor.postprocessor_args
        
        # Diffusion parameters
        self.diffusion_config = self._get_diffusion_config()
        self.model, self.diffusion = None, None
        
        # Reference point parameters
        self.kmeans_k = self.args.kmeans_k
        self.optim_steps = self.args.optim_steps
        self.optim_lr = self.args.optim_lr
        
        # Quadrature parameters
        self.n_quadrature_points = self.args.n_quadrature_points
        
        # Models and data
        self.reference_points = None
        self.scaler = None
        self.kde = None

    def _get_diffusion_config(self):
        """Create diffusion configuration from arguments"""
        defaults = model_and_diffusion_defaults()
        return dict(
            image_size=self.args.image_size,
            num_channels=self.args.num_channels,
            num_res_blocks=self.args.num_res_blocks,
            learn_sigma=self.args.learn_sigma,
            class_cond=False,
            use_checkpoint=False,
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
        # Collect ID training data
        id_train_loader = id_loader_dict['train']
        all_data = []
        for batch in tqdm(id_train_loader, desc='Collecting ID data'):
            data = batch['data'].to(device)
            all_data.append(data)
        self.train_data = torch.cat(all_data)
        
        # Initialize diffusion model
        self._init_diffusion_model()
        
        # Train diffusion model
        self._train_diffusion_model(id_train_loader)
        
        # Get optimized reference points
        self.reference_points = self._get_optimized_reference_points()
        
        # Compute integral scores for KDE training
        id_scores = self._compute_integrals(self.train_data)
        
        # Train KDE
        self.scaler, self.kde = self._train_kde(id_scores)

    def _init_diffusion_model(self):
        """Initialize Improved Diffusion model and diffusion process"""
        logger.configure()
        self.model, self.diffusion = create_model_and_diffusion(
            **self.diffusion_config
        )
        self.model.to(device)
        self.model.train()

    def _train_diffusion_model(self, train_loader):
        """Train diffusion model using Improved Diffusion framework"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        for epoch in range(self.args.epochs):
            with tqdm(train_loader, desc=f"Diffusion Training Epoch {epoch+1}") as pbar:
                for batch in pbar:
                    data = batch['data'].to(device)
                    
                    # Sample timesteps
                    t = torch.randint(
                        0, self.diffusion.num_timesteps, data.shape[0], device=device
                    )
                    
                    # Compute loss
                    loss = self.diffusion.training_losses(
                        model=self.model,
                        x_start=data,
                        t=t,
                        model_kwargs={}
                    )['loss'].mean()
                    
                    # Optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    pbar.set_postfix({'loss': loss.item()})
                    
            # Save checkpoints if needed
            # torch.save(self.model.state_dict(), f"diffusion_epoch{epoch}.pt")

        self.model.eval()

    def _get_optimized_reference_points(self):
        """Find and optimize reference points using K-means + gradient ascent"""
        # Flatten data for clustering
        flattened_data = self.train_data.view(len(self.train_data), -1).cpu().numpy()
        
        # K-means initialization
        kmeans = KMeans(n_clusters=self.kmeans_k, n_init=10)
        kmeans.fit(flattened_data)
        x_init = torch.tensor(kmeans.cluster_centers_, 
                            dtype=torch.float32, 
                            device=device)
        
        # Gradient-based optimization
        return self._optimize_reference_points(x_init)

    def _optimize_reference_points(self, x_init):
        """Batch gradient ascent on model score"""
        x = nn.Parameter(x_init.clone())
        optimizer = torch.optim.Adam([x], lr=self.optim_lr)
        
        for _ in tqdm(range(self.optim_steps), desc="Reference Optimization"):
            optimizer.zero_grad()
            
            # Reshape to original data dimensions
            x_reshaped = x.view(-1, *self.train_data.shape[1:])
            
            # Compute score using Improved Diffusion API
            t = torch.zeros(x.shape[0], device=device, dtype=torch.long)
            noise = torch.randn_like(x_reshaped)
            x_t = self.diffusion.q_sample(x_reshaped, t, noise=noise)
            model_output = self.model(x_t, self.diffusion._scale_timesteps(t))
            
            if self.diffusion.model_var_type in ['learned', 'learned_range']:
                model_output, _ = torch.split(model_output, x_reshaped.shape[1], dim=1)
            
            score = (noise - model_output) / self.diffusion.sqrt_one_minus_alphas_cumprod[t]
            score = score.view(x.shape)  # Reshape back to flattened
            
            # Manual gradient assignment
            x.grad = -score.mean(dim=0)
            optimizer.step()
            
        return x.detach()

    def _compute_integrals(self, data):
        """Compute Gaussian quadrature integrals for all samples"""
        scores = []
        for x_ref in tqdm(self.reference_points, desc="Reference Points"):
            scores.append(self._gaussian_quadrature(data, x_ref))
        return torch.stack(scores, dim=1).cpu().numpy()

    def _gaussian_quadrature(self, x, x_ref):
        """Compute line integral using Gauss-Legendre quadrature"""
        points, weights = self._get_gauss_legendre_params(self.n_quadrature_points)
        integral = torch.zeros(x.shape[0], device=device)
        
        # Reshape references to match data dimensions
        x_ref_reshaped = x_ref.view(1, *x.shape[1:]).repeat(x.shape[0], 1, 1, 1)
        
        for t, w in zip(points, weights):
            t_scaled = 0.5 * (t + 1)  # Map to [0,1]
            z = x * (1 - t_scaled) + x_ref_reshaped * t_scaled
            
            # Compute score at current z
            with torch.no_grad():
                t_batch = torch.zeros(z.shape[0], device=device, dtype=torch.long)
                noise = torch.randn_like(z)
                z_t = self.diffusion.q_sample(z, t_batch, noise=noise)
                model_output = self.model(z_t, self.diffusion._scale_timesteps(t_batch))
                
                if self.diffusion.model_var_type in ['learned', 'learned_range']:
                    model_output, _ = torch.split(model_output, z.shape[1], dim=1)
                
                score = (noise - model_output) / self.diffusion.sqrt_one_minus_alphas_cumprod[t_batch]
            
            # Compute directional derivative
            directional = torch.sum(score.view(x.shape) * (x_ref - x.view(x_ref.shape)), dim=1)
            integral += w * directional
        
        return -0.5 * integral

    def _get_gauss_legendre_params(self, n):
        """Return Gauss-Legendre points and weights"""
        points, weights = np.polynomial.legendre.leggauss(n)
        return (
            torch.tensor(points, device=device), 
            torch.tensor(weights, device=device)
        )

    def _train_kde(self, scores):
        """Train KDE on normalized integral scores"""
        self.scaler = StandardScaler()
        scaled_scores = self.scaler.fit_transform(scores)
        
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': np.logspace(-2, 1, 20)},
            cv=5,
            n_jobs=-1
        )
        grid.fit(scaled_scores)
        
        self.kde = grid.best_estimator_
        self.kde.fit(scaled_scores)
        return self.scaler, self.kde

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """Compute OOD confidence scores"""
        # Compute integral scores
        scores = []
        for x_ref in self.reference_points:
            scores.append(self._gaussian_quadrature(data, x_ref).cpu().numpy())
        scores = np.stack(scores, axis=1)
        
        # Transform and score with KDE
        scaled_scores = self.scaler.transform(scores)
        log_probs = self.kde.score_samples(scaled_scores)
        
        # Return dummy predictions and confidence scores
        return (
            torch.zeros(len(data), dtype=torch.long, device=device),
            torch.tensor(-log_probs, device=device)
        )