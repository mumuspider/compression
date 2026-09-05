import copy
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
from tqdm import tqdm

from quantize import *
from utils import LogWriter, compute_sam, loss_fn


def reset_peak_gpu_memory_stats(device):
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())


def get_peak_gpu_memory_mib(device):
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(torch.cuda.current_device()) / (1024 ** 2))


class SimpleTrainerHSI:
    """Trains random 2d gaussians to fit an image."""

    def __init__(
        self,
        ground_truth: torch.tensor,
        endmember: np.array,
        num_points: int = 2000,
        model_name: str = 'GaussianImage_Cholesky_nd',
        iterations: int = 50000,
        model_path=None,
        data_name='HSI',
        image_name=None,
    ):
        self.device = torch.device('cuda:0')
        reset_peak_gpu_memory_stats(self.device)
        torch.cuda.synchronize()
        self.gt_image = ground_truth.to(self.device).half()
        self.endmember = endmember
        self.num_points = num_points
        block_h, block_w = 16, 16
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        self.H, self.W, self.rank, self.C = self.gt_image.shape[2], self.gt_image.shape[3], endmember.shape[0], self.gt_image.shape[1]

        self.iterations = iterations
        self.log_dir = Path(f'./checkpoints/{data_name}/{model_name}_{iterations}_{num_points}_{self.rank}/{image_name}')
        self.image_name = image_name

        torch.cuda.synchronize()
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f'GPU memory GT: {gpu_memory:.2f} MB')

        if model_name == 'GaussianImage_Cholesky_nd':
            from gaussianimage_cholesky_unknown import GaussianImage_Cholesky_EA
            self.gaussian_model = GaussianImage_Cholesky_EA(
                loss_type='L2',
                opt_type='adan',
                num_points=self.num_points,
                GT=self.gt_image,
                E=self.endmember,
                H=self.H,
                W=self.W,
                C=self.C,
                rank=self.rank,
                BLOCK_H=block_h,
                BLOCK_W=block_w,
                device=self.device,
                lr=5e-3,
                quantize=False,
            ).to(self.device)

        torch.cuda.synchronize()
        model_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f'GPU memory after model initialization: {model_gpu_memory:.2f} MB (Model size: {model_gpu_memory - gpu_memory:.2f} MB)')

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f'loading model path:{model_path}')
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self):
        psnr_list, loss_list = [], []
        use_tqdm = sys.stderr.isatty() or sys.stdout.isatty()
        progress_bar = tqdm(total=self.iterations, desc='Training progress', ncols=100, dynamic_ncols=False, leave=True) if use_tqdm else None
        self.gaussian_model.train()
        start_time = time.time()
        best_psnr = 0.0
        best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())

        for iteration in range(1, self.iterations + 1):
            loss, psnr = self.gaussian_model.train_iter_quantize()
            psnr_list.append(psnr)
            loss_list.append(float(loss.item()))
            if math.isfinite(psnr) and best_psnr < psnr:
                best_psnr = psnr
                best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())

            if progress_bar is not None:
                progress_bar.update(1)
                if iteration == 1 or iteration % 10 == 0 or iteration == self.iterations:
                    progress_bar.set_postfix(loss=f'{loss.item():.7f}', psnr=f'{psnr:.4f}', best_psnr=f'{best_psnr:.4f}')
            elif iteration == 1 or iteration % 100 == 0 or iteration == self.iterations:
                print(f'[{iteration:5d}/{self.iterations}] loss={loss.item():.7f}, psnr={psnr:.4f}, best_psnr={best_psnr:.4f}')

        if progress_bar is not None:
            progress_bar.close()

        end_time = time.time() - start_time
        psnr_value, ssim_value, ms_ssim_value, sam, bpppb, bit_components = self.test()
        torch.save(self.gaussian_model.state_dict(), self.log_dir / 'gaussian_model.pth.tar')
        self.gaussian_model.load_state_dict(best_model_dict)
        best_psnr_value, best_ssim_value, best_ms_ssim_value, best_sam, best_bpppb, best_bit_components = self.test(True)
        torch.save(best_model_dict, self.log_dir / 'gaussian_model.best.pth.tar')
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for _ in range(100):
                _ = self.gaussian_model.forward_quantize()
            test_end_time = (time.time() - test_start_time) / 100

        self.logwriter.write('Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}'.format(end_time, test_end_time, 1 / test_end_time))
        peak_gpu_mem_mib = get_peak_gpu_memory_mib(self.device)
        if peak_gpu_mem_mib is not None:
            self.logwriter.write('Peak GPU Mem:{:.2f} MiB'.format(peak_gpu_mem_mib))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / 'gaussian_model.pth.tar')

        training_logs = {
            'psnr': psnr_list,
            'loss': loss_list,
        }
        return (
            {
                'psnr': psnr_value,
                'ssim': ssim_value,
                'ms_ssim': ms_ssim_value,
                'sam': float(best_sam if math.isnan(sam) else sam),
                'bpppb': bpppb,
                'bit_components': bit_components,
                'best_psnr': best_psnr_value,
                'best_ssim': best_ssim_value,
                'best_ms_ssim': best_ms_ssim_value,
                'best_sam': float(best_sam),
                'best_bpppb': best_bpppb,
                'best_bit_components': best_bit_components,
                'training_time': end_time,
                'eval_time': test_end_time,
                'eval_fps': 1 / test_end_time,
                'peak_gpu_mem_mib': peak_gpu_mem_mib,
            },
            training_logs,
            {
                'latest': self.log_dir / 'gaussian_model.pth.tar',
                'best': self.log_dir / 'gaussian_model.best.pth.tar',
                'train_txt': self.log_dir / 'train.txt',
                'curve': self.log_dir / f'{self.image_name}_training_curves.png',
            },
        )

    def test(self, best=False):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize()
            A = out['render'].float()
            E = FakeQuantizationHalf.apply(self.gaussian_model.endmember.to(torch.float32))
            I = A @ E
            I = I.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
            mse_per_channel = F.mse_loss(I, self.gt_image, reduction='none')
            mse_per_channel_avg = mse_per_channel.mean(dim=(0, 2, 3)).clamp_min(1e-12)
            psnr_per_channel = 10 * torch.log10(1.0 / mse_per_channel_avg)
            psnr = psnr_per_channel.mean().item()

        ssim_value = ssim(I, self.gt_image.float(), data_range=1, size_average=True, win_size=7).item()
        ms_ssim_value = ms_ssim(I, self.gt_image.float(), data_range=1, size_average=True, win_size=7).item()
        mean_sam = compute_sam(self.gt_image.squeeze(0).permute(1, 2, 0).cpu().numpy(), I.squeeze(0).permute(1, 2, 0).cpu().numpy())

        m_bit, s_bit, r_bit, c_bit = out['unit_bit']
        e_bit = self.rank * self.C * 16
        total_bits = m_bit + s_bit + r_bit + c_bit + e_bit
        denom = self.H * self.W * self.C
        bpppb = total_bits / denom
        bit_components = {
            'm_bit': float(m_bit),
            's_bit': float(s_bit),
            'r_bit': float(r_bit),
            'c_bit': float(c_bit),
            'e_bit': float(e_bit),
            'total_bits': float(total_bits),
            'm_bpppb': float(m_bit / denom),
            's_bpppb': float(s_bit / denom),
            'r_bpppb': float(r_bit / denom),
            'c_bpppb': float(c_bit / denom),
            'e_bpppb': float(e_bit / denom),
        }

        prefix = 'Best Test' if best else 'Test'
        self.logwriter.write('{} PSNR:{:.4f}, SSIM:{:.6f}, MS_SSIM:{:.6f}, bpppb:{:.4f}'.format(prefix, psnr, ssim_value, ms_ssim_value, bpppb))
        return psnr, ssim_value, ms_ssim_value, mean_sam, bpppb, bit_components


def save_training_curve(logdir: Path, image_name: str, training_logs: dict):
    plt.figure(figsize=(10, 6))
    plt.plot(training_logs['psnr'], color='#2E86C1', linewidth=2)
    plt.title('PSNR Training Curve', fontsize=14, pad=15)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(logdir / f'{image_name}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def train_nd(gt, endmember, image_name, iterations, num_points, model_name='GaussianImage_Cholesky_nd', data_name='HSI'):
    trainer = SimpleTrainerHSI(
        ground_truth=gt,
        endmember=endmember,
        num_points=num_points,
        iterations=iterations,
        model_name=model_name,
        image_name=image_name,
        data_name=data_name,
    )
    metrics, training_logs, checkpoint_paths = trainer.train()
    trainer.logwriter.write('{}: {}x{}x{}, Rank: {}, bpppb: {:.4f}, PSNR:{:.4f}, SSIM:{:.4f}, MS-SSIM:{:.4f}, Best bpppb: {:.4f}, Best PSNR:{:.4f}, Best SSIM:{:.4f}, Best MS-SSIM:{:.4f}, Best SAM: {:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}, Peak GPU Mem:{:.2f} MiB'.format(
        image_name,
        trainer.H,
        trainer.W,
        trainer.C,
        trainer.rank,
        metrics['bpppb'],
        metrics['psnr'],
        metrics['ssim'],
        metrics['ms_ssim'],
        metrics['best_bpppb'],
        metrics['best_psnr'],
        metrics['best_ssim'],
        metrics['best_ms_ssim'],
        metrics['best_sam'],
        metrics['training_time'],
        metrics['eval_time'],
        metrics['eval_fps'],
        metrics.get('peak_gpu_mem_mib') or 0.0,
    ))
    save_training_curve(trainer.log_dir, image_name, training_logs)
    return metrics, training_logs, checkpoint_paths
