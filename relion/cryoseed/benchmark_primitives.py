import torch
import torch.utils.benchmark as benchmark
import gc

# MUST initialize CUDA runtime before importing Triton to avoid "0 active drivers" error
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark.")
# Force CUDA initialization
_ = torch.tensor([0.0], device='cuda')

from cryoseed.backends.torch.project import project as project_torch
from cryoseed.backends.triton.project import project as project_triton

from cryoseed.backends.torch.backproject import backproject as backproject_torch
from cryoseed.backends.triton.backproject import backproject as backproject_triton

from cryoseed.backends.torch.spectral_mse_loss import spectral_mse_loss as mse_torch
from cryoseed.backends.triton.spectral_mse_loss import spectral_mse_loss as mse_triton

def benchmark_primitive(stmt, globals_dict, label, sub_label, description, min_run_time=4.0):
    # 增加预热次数和最小运行时间，确保更稳定的统计结果
    t = benchmark.Timer(
        stmt=stmt,
        globals=globals_dict,
        label=label,
        sub_label=sub_label,
        description=description,
    )
    # blocked_autorange 会自动进行预热并决定迭代次数，确保总运行时间达到 min_run_time
    return t.blocked_autorange(min_run_time=min_run_time)

def measure_peak_memory(stmt, globals_dict, device):
    """Measures the peak memory increase (in MB) for a single execution."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Run the statement once to capture memory
    exec(stmt, globals_dict)
    
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 * 1024)  # Convert to MB

def _warmup_gpu(device):
    """Run heavy computations to warm up GPU and wake it from idle state."""
    print("Warming up GPU to stabilize clocks...")
    for _ in range(50):
        a = torch.randn(4096, 4096, device=device)
        b = torch.randn(4096, 4096, device=device)
        _ = a @ b
    torch.cuda.synchronize()

def run_benchmarks():
    device = torch.device('cuda')
    _warmup_gpu(device)
    
    print("Starting primitive benchmarks. This may take a few minutes...\n")
    
    # 组合: (Box Size L, Images B, Poses Q)
    configs = [
        (64, 8, 128),
        (64, 8, 2048),
        (64, 16, 2048),
        (128, 8, 128),
        (128, 8, 2048),
        (128, 16, 2048),
        (256, 8, 128),
        (256, 8, 2048),
        (256, 16, 2048),
    ]
    results = []
    mem_results = []
    
    for L, B_images, Q_poses in configs:
        sub_label = f"Box={L}, Img={B_images}, Pose={Q_poses}"
        print(f"Preparing and benchmarking: {sub_label}")
        
        # ==========================================
        # 1. Forward projection (受 Poses 数量影响)
        # ==========================================
        vol = torch.randn(1, L, L, L, 2, dtype=torch.float32, device=device)
        rot = torch.randn(1, Q_poses, 3, 3, dtype=torch.float32, device=device)
        # 确保旋转矩阵是正交的 (Orthogonalize)
        u, _, v = torch.svd(rot)
        rot = torch.matmul(u, v.transpose(-1, -2))
        
        globals_proj = {'vol': vol, 'rot': rot, 'project_torch': project_torch, 'project_triton': project_triton}
        
        # Memory measurement
        mem_torch_proj = measure_peak_memory('project_torch(vol, rot, channel_last=True)', globals_proj, device)
        mem_triton_proj = measure_peak_memory('project_triton(vol, rot, channel_last=True)', globals_proj, device)
        mem_results.append((sub_label, '1. Forward projection', mem_torch_proj, mem_triton_proj))
        
        res_torch_proj = benchmark_primitive('project_torch(vol, rot, channel_last=True)', globals_proj, '1. Forward projection', sub_label, 'PyTorch')
        res_triton_proj = benchmark_primitive('project_triton(vol, rot, channel_last=True)', globals_proj, '1. Forward projection', sub_label, 'Triton')
        results.extend([res_torch_proj, res_triton_proj])
        
        # ==========================================
        # 2. Backprojection (受 Images 数量影响)
        # ==========================================
        N = B_images
        image = torch.randn(N, L, L, dtype=torch.complex64, device=device)
        ctf = torch.randn(N, L, L, dtype=torch.float32, device=device)
        noise = torch.rand(L, L, dtype=torch.float32, device=device) + 0.1
        image_idx = None
        vol_idx = torch.zeros(N, dtype=torch.int64, device=device)
        trans = torch.zeros(N, 2, dtype=torch.float32, device=device)
        prob = torch.ones(N, dtype=torch.float32, device=device)
        radius = L // 2
        
        # 预先分配累加张量
        vol_num = torch.zeros(1, L, L, L, dtype=torch.complex64, device=device)
        vol_den = torch.zeros(1, L, L, L, dtype=torch.float32, device=device)
        
        # 为 N 个 Image 分配旋转矩阵
        rot_bp = torch.randn(N, 3, 3, dtype=torch.float32, device=device)
        
        globals_bp = {
            'image': image, 'ctf': ctf, 'noise': noise, 'image_idx': image_idx, 'vol_idx': vol_idx,
            'rot': rot_bp, 'trans': trans, 'prob': prob, 'radius': radius,
            'vol_num': vol_num, 'vol_den': vol_den,
            'backproject_torch': backproject_torch, 'backproject_triton': backproject_triton
        }
        stmt_bp = ("f(image, ctf, noise, image_index=image_idx, volume_index=vol_idx, "
                   "rotation=rot, translation=trans, probability=prob, radius=radius, "
                   "volume_numerator=vol_num, volume_denominator=vol_den)")
        
        # Memory measurement
        mem_torch_bp = measure_peak_memory(stmt_bp.replace('f(', 'backproject_torch('), globals_bp, device)
        mem_triton_bp = measure_peak_memory(stmt_bp.replace('f(', 'backproject_triton('), globals_bp, device)
        mem_results.append((sub_label, '2. Backprojection', mem_torch_bp, mem_triton_bp))
        
        res_torch_bp = benchmark_primitive(stmt_bp.replace('f(', 'backproject_torch('), globals_bp, '2. Backprojection', sub_label, 'PyTorch')
        res_triton_bp = benchmark_primitive(stmt_bp.replace('f(', 'backproject_triton('), globals_bp, '2. Backprojection', sub_label, 'Triton')
        results.extend([res_torch_bp, res_triton_bp])
        
        # ==========================================
        # 3. Weighted spectral MSE (受 Images, Poses, Translations 共同影响)
        # ==========================================
        D = L * L
        T_trans = 16  # 模拟真实的平移候选数量 (e.g. 4x4 grid)
        
        input_t = torch.randn(B_images, Q_poses, D, dtype=torch.complex64, device=device)
        target_t = torch.randn(B_images, T_trans, D, dtype=torch.complex64, device=device)
        weight = torch.rand(D, dtype=torch.float32, device=device)
        
        globals_mse = {
            'input_t': input_t, 'target_t': target_t, 'weight': weight,
            'mse_torch': mse_torch, 'mse_triton': mse_triton
        }
        
        # Memory measurement for MSE
        try:
            mem_torch_mse = measure_peak_memory('mse_torch(input_t, target_t, weight=weight, reduction="sum", spectral_reduction="sum")', globals_mse, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                mem_torch_mse = float('inf')
            else:
                raise e
        mem_triton_mse = measure_peak_memory('mse_triton(input_t, target_t, weight=weight, reduction="sum", spectral_reduction="sum")', globals_mse, device)
        mem_results.append((sub_label, '3. Weighted spectral MSE', mem_torch_mse, mem_triton_mse))
        
        try:
            res_torch_mse = benchmark_primitive('mse_torch(input_t, target_t, weight=weight, reduction="sum", spectral_reduction="sum")', globals_mse, '3. Weighted spectral MSE', sub_label, 'PyTorch')
            results.append(res_torch_mse)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    [!] PyTorch OOM on MSE (Box={L}, Img={B_images}, Pose={Q_poses})")
            else:
                raise e
                
        res_triton_mse = benchmark_primitive('mse_triton(input_t, target_t, weight=weight, reduction="sum", spectral_reduction="sum")', globals_mse, '3. Weighted spectral MSE', sub_label, 'Triton')
        results.append(res_triton_mse)
        
    print("\n" + "="*80)
    print("Peak Memory Allocation (MB)")
    print("="*80)
    print(f"{'Configuration':<35} | {'Primitive':<25} | {'PyTorch (MB)':<12} | {'Triton (MB)':<12}")
    print("-" * 90)
    for cfg, prim, pt_mem, tr_mem in mem_results:
        pt_str = "OOM" if pt_mem == float('inf') else f"{pt_mem:.1f}"
        tr_str = f"{tr_mem:.1f}"
        print(f"{cfg:<35} | {prim:<25} | {pt_str:<12} | {tr_str:<12}")
        
    print("\n" + "="*80)
    print("Benchmark Results (Time per call)")
    print("="*80)
    # 打印中位数(median)和 IQR 而不是单纯的均值，这能有效抵抗偶尔的系统卡顿带来的离群点
    compare = benchmark.Compare(results)
    compare.print()

if __name__ == '__main__':
    run_benchmarks()