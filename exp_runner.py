import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SDFNetworkHigh, SingleVarianceNetwork, NeRF
from semantic.network import SemanticNetwork
from semantic.guidance import SemanticGuidance
from models.renderer import NeuSRenderer
import json
import lpips as lpips_lib
from third_party import pytorch_ssim
import matplotlib
import datetime


matplotlib.use('Agg')

def mse2psnr(mse):
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, ckpt_name=None, base_exp_dir=None, end_iter=None):
        self.device = torch.device('cuda')
        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        if base_exp_dir is not None:
            self.base_exp_dir = base_exp_dir
            self.conf.put('general.base_exp_dir', base_exp_dir)
        else:
            self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.train_images = self.dataset.images.cuda()
        self.iter_step = 0
        # Training parameters
        if end_iter is not None:
            self.end_iter = end_iter
            self.conf.put('train.end_iter', end_iter)
        else:
            self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.ckpt_name = ckpt_name
        self.mode = mode
        self.model_list = []
        self.writer = None
        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.sdf_network_high = SDFNetworkHigh(**self.conf['model.sdf_network_high']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        self.semantic_network = SemanticNetwork(**self.conf['model.semantic_network']).to(self.device)
        self.semantic_guidance = SemanticGuidance().to(self.device) # Init Guidance
        
        self.lpips_vgg_fn = lpips_lib.LPIPS(net='vgg').to(self.device)  # net="alex"
        
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.sdf_network_high.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.semantic_network.parameters())
        params_to_train += list(self.semantic_guidance.parameters()) # Train linear projection
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        
        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.sdf_network_high,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'],
                                     semantic_network=self.semantic_network,
                                     semantic_guidance=self.semantic_guidance) # Pass guidance
        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if self.ckpt_name is not None:
                latest_model_name = self.ckpt_name
            else:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                num = -1
                for model_name in model_list_raw:
                    iter = int(model_name[5:-4])
                    if model_name[-3:] == 'pth' and iter <= self.end_iter:
                        if iter > num:
                            num = iter
                            model_list = model_name
                latest_model_name = model_list
        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        if self.mode[:5] == 'train' and not is_continue:
            self.file_backup()

    def train(self):
        lpips_vgg_fn = self.lpips_vgg_fn
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        image_perm = self.get_image_perm()
        res_step = self.end_iter - self.iter_step

        progress_data = 1.0
        self.nerf_outside.progress.data.fill_(progress_data)
        
        # Load Semantic Weight ---
        sem_weight = self.conf.get_float('train.sem_weight', default=0.0)

        # ETA Init
        training_start_time = time.time()
        
        for iter_i in tqdm(range(res_step)):
            idx_list = image_perm[self.iter_step % len(image_perm)]
            img_idx = idx_list.item() 

            data = self.dataset.gen_random_rays_at(idx_list, self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
            
            # Get features
            feature_map = self.dataset.dino_features[img_idx]
            intrinsics = self.dataset.intrinsics_all[img_idx]
            pose = self.dataset.pose_all[img_idx]
            img_size = (self.dataset.H, self.dataset.W)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3]).to(self.device)

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5

            self.optimizer.zero_grad()
            
            # --- AMP Context ---
            with torch.amp.autocast('cuda'):
                render_out = self.renderer.render(rays_o, rays_d, near, far,
                                                  background_rgb=background_rgb,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  feature_map=feature_map,
                                                  intrinsics=intrinsics,
                                                  pose=pose,
                                                  img_size=img_size)

                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_error = render_out['gradient_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                
                # --- PHASE 2: Retrieve Semantic Loss ---
                sem_loss_val = render_out['sem_loss']

                # Loss Calculation
                color_error = (color_fine - true_rgb) * mask
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                eikonal_loss = gradient_error
                
            
            with torch.amp.autocast('cuda', enabled=False):
                weight_sum_f32 = weight_sum.float().clip(1e-3, 1.0 - 1e-3)
                mask_f32 = mask.float()
                mask_loss = F.binary_cross_entropy(weight_sum_f32, mask_f32)
                
                # Add Semantic Loss ---
                loss = color_fine_loss + \
                       eikonal_loss * self.igr_weight + \
                       mask_loss * self.mask_weight + \
                       sem_loss_val * sem_weight

            # Scaled Backward
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            # Log Semantic Loss
            self.writer.add_scalar('Loss/sem_loss', sem_loss_val, self.iter_step)
            
            self.writer.add_scalar('Statistics/s_val', s_val.mean().item(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0 or self.iter_step == 1:
                elapsed_time = time.time() - training_start_time
                iterations_done = iter_i + 1 
                if iterations_done > 0:
                    avg_time_per_iter = elapsed_time / iterations_done
                    remaining_iters = res_step - iterations_done
                    eta_seconds = avg_time_per_iter * remaining_iters
                    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    eta_str = "Calculating..."
                
                print(self.base_exp_dir)
                outstr = 'iter:{:8>d} loss = {:.4f} sem_loss={:.4f} lr={:.6f} ETA={}\n'.format(
                    self.iter_step, loss, sem_loss_val, self.optimizer.param_groups[0]['lr'], eta_str)
                print(outstr)
                f = os.path.join(self.base_exp_dir, 'logs', 'loss.txt')
                with open(f, 'a') as file:
                    file.write(outstr)

            if self.iter_step % self.save_freq == 0 or self.iter_step == 1:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0 or self.iter_step == 1:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0 or self.iter_step == 1:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step > self.end_iter / 2:
                progress_data = 1.0
            else:
                progress_data = 0.5 + self.iter_step / (self.end_iter)
            self.sdf_network.progress.data.fill_(progress_data)
            self.sdf_network_high.progress.data.fill_(progress_data)
            self.color_network.progress.data.fill_(progress_data)

            if (iter_i+1) % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.sdf_network_high.load_state_dict(checkpoint['sdf_network_high_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.semantic_network.load_state_dict(checkpoint['semantic_network']) # Load semantic network
        self.semantic_guidance.load_state_dict(checkpoint['semantic_guidance']) # Load guidance projection
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'sdf_network_high_fine': self.sdf_network_high.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'semantic_network': self.semantic_network.state_dict(), # Save
            'semantic_guidance': self.semantic_guidance.state_dict(), # Save
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def compute_metrics_for_idx(self, idx, resolution_level=1):
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            
            # Autocast during inference
            with torch.amp.autocast('cuda'):
                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  near,
                                                  far,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  background_rgb=background_rgb,
                                                  feature_map=feature_map,
                                                  intrinsics=intrinsics,
                                                  pose=pose,
                                                  img_size=img_size)
            if 'color_fine' in render_out and render_out['color_fine'] is not None:
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            del render_out
            torch.cuda.empty_cache()
        if out_rgb_fine:
            color_fine = torch.from_numpy(np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])).to(self.device)
            true_rgb = torch.from_numpy(self.dataset.image_at(idx, resolution_level=resolution_level) / 255.0).float().to(self.device)
            mse = F.mse_loss(color_fine, true_rgb).item()
            psnr = mse2psnr(mse)
            ssim = pytorch_ssim.ssim(color_fine.permute(2, 0, 1).unsqueeze(0),
                                     true_rgb.permute(2, 0, 1).unsqueeze(0)).item()
            lpips = self.lpips_vgg_fn(color_fine.permute(2, 0, 1).unsqueeze(0).contiguous(),
                                       true_rgb.permute(2, 0, 1).unsqueeze(0).contiguous(), normalize=True).item()
            del color_fine, true_rgb
            torch.cuda.empty_cache()
            return psnr, ssim, lpips
        return None, None, None

    def compute_metrics_for_indices(self, indices=[50, 51, 52], resolution_level=1):
        psnr_list, ssim_list, lpips_list = [], [], []
        for idx in indices:
            psnr, ssim, lpips = self.compute_metrics_for_idx(idx, resolution_level)
            if psnr is not None:
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                lpips_list.append(lpips)
                metrics_file = os.path.join(self.base_exp_dir, f'metrics_idx{idx}.txt')
                with open(metrics_file, 'w') as f:
                    f.write(f"Metrics for Camera idx={idx} at iteration {self.iter_step}:\n")
                    f.write(f"PSNR: {psnr:.4f}\n")
                    f.write(f"SSIM: {ssim:.4f}\n")
                    f.write(f"LPIPS: {lpips:.4f}\n")
                print(f"Metrics saved to {metrics_file}")
        # Compute geometry metrics (once, view-independent)
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        vertices, triangles = self.renderer.extract_geometry(bound_min, bound_max, resolution=1024, threshold=0.0)
        pred_mesh = trimesh.Trimesh(vertices, triangles)
        gt_mesh_path = os.path.join(self.dataset.data_dir, 'ground_truth.ply')
        gt_mesh = trimesh.load(gt_mesh_path) if os.path.exists(gt_mesh_path) else None
        f_score = trimesh.metrics.f_score(pred_mesh, gt_mesh, threshold=0.001) if gt_mesh else None
        nc = trimesh.metrics.normal_consistency(pred_mesh, gt_mesh) if gt_mesh else None
        # Save average metrics and geometry metrics
        avg_metrics_file = os.path.join(self.base_exp_dir, 'metrics_avg.txt')
        with open(avg_metrics_file, 'w') as f:
            f.write(f"Average Metrics across Camera indices {indices} at iteration {self.iter_step}:\n")
            f.write(f"PSNR: {np.mean(psnr_list) if psnr_list else 'N/A':.4f}\n")
            f.write(f"SSIM: {np.mean(ssim_list) if ssim_list else 'N/A':.4f}\n")
            f.write(f"LPIPS: {np.mean(lpips_list) if lpips_list else 'N/A':.4f}\n")
            f.write(f"F-Score: {f_score if f_score is not None else 'N/A':.4f}\n")
            f.write(f"Normal Consistency: {nc if nc is not None else 'N/A':.4f}\n")
        print(f"Average metrics saved to {avg_metrics_file}")

    def validate_image(self, idx=-1, resolution_level=-1):
        lpips_vgg_fn = self.lpips_vgg_fn
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        outstr = 'Validate: iter: {}, camera: {}\n'.format(self.iter_step, idx)
        print(outstr)
        if resolution_level == 1:
            f = os.path.join(self.base_exp_dir, 'logs', 'image_metric.txt')
            with open(f, 'a') as file:
                file.write(outstr)
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        
        # Get DINO
        feature_map = self.dataset.dino_features[idx]
        intrinsics = self.dataset.intrinsics_all[idx]
        pose = self.dataset.pose_all[idx]
        img_size = (self.dataset.H, self.dataset.W)

        out_rgb_fine = []
        out_normal_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            
            # Autocast in validation
            with torch.amp.autocast('cuda'):
                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  near,
                                                  far,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  background_rgb=background_rgb,
                                                  feature_map=feature_map,
                                                  intrinsics=intrinsics,
                                                  pose=pose,
                                                  img_size=img_size)
            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)
            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out
            torch.cuda.empty_cache()
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            if resolution_level == 1:
                color_fine = torch.from_numpy(np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])).to(self.device)
                true_rgb = self.dataset.images[idx].to(self.device)
                color_error = color_fine - true_rgb
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='mean')
                outstr = '{0:4d} img: loss: {1:.2f}\n'.format(idx, color_fine_loss)
                print(outstr)
                f = os.path.join(self.base_exp_dir, 'logs', 'image_metric.txt')
                with open(f, 'a') as file:
                    file.write(outstr)
                mse = F.mse_loss(color_fine, true_rgb).item()
                psnr = mse2psnr(mse)
                ssim = pytorch_ssim.ssim(color_fine.permute(2, 0, 1).unsqueeze(0),
                                          true_rgb.permute(2, 0, 1).unsqueeze(0)).item()
                lpips_loss = lpips_vgg_fn(color_fine.permute(2, 0, 1).unsqueeze(0).contiguous(),
                                          true_rgb.permute(2, 0, 1).unsqueeze(0).contiguous(), normalize=True).item()
                outstr = '{0:4d} img: PSNR: {1:.2f}, SSIM: {2:.2f}, LPIPS {3:.2f}\n'.format(idx, psnr, ssim, lpips_loss)
                print(outstr)
                f = os.path.join(self.base_exp_dir, 'logs', 'image_metric.txt')
                with open(f, 'a') as file:
                    file.write(outstr)
        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)
            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            del render_out
        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        print('Validate: iter: {} mesh'.format(self.iter_step))
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        vertices, triangles = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh_path = os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step))
        mesh.export(mesh_path)
        logging.info(f'Mesh saved to {mesh_path}')
        if self.mode == 'validate_mesh':
            print("Computing metrics for multiple camera indices in validate_mesh mode...")
            self.compute_metrics_for_indices(resolution_level=self.validate_resolution_level)

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                                  resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(self.base_exp_dir,
                                             'render',
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))
        for image in images:
            writer.write(image)
        writer.release()

if __name__ == '__main__':
    print('Hello Wooden')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--image_idx', type=int, default=0)
    parser.add_argument('--image_resolution', type=int, default=4)
    parser.add_argument('--mesh_resolution', type=int, default=1024)
    parser.add_argument('--base_exp_dir', type=str, default=None)
    parser.add_argument('--end_iter', type=int, default=None)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.ckpt_name, args.base_exp_dir, args.end_iter)
    if args.mode == 'train':
        runner.train()
        runner.validate_mesh(world_space=True, resolution=args.mesh_resolution,
                             threshold=args.mcube_threshold)
        runner.validate_image(idx=args.image_idx, resolution_level=args.image_resolution)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=args.mesh_resolution, threshold=args.mcube_threshold)
    elif args.mode == 'validate_image':
        runner.validate_image(idx=args.image_idx, resolution_level=args.image_resolution)
    elif args.mode.startswith('interpolate'):
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)