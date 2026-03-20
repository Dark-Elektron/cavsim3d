from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import gmres
import multiprocessing

# ============================================================
# THREAD CONTROL
# ============================================================

def _init_worker(threads_per_process):
    """Initializer for worker processes - limits threads."""
    os.environ["OMP_NUM_THREADS"] = str(threads_per_process)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_process)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_process)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_process)
    
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=threads_per_process)
    except ImportError:
        pass


class ParalleliseCode:
    """Context manager with dynamic scheduling and thread control."""
    
    def __init__(self, max_workers=None, threads_per_process=1):
        self.max_workers = max_workers
        self.threads_per_process = threads_per_process
        self.executor = None
    
    def __enter__(self):
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker,
            initargs=(self.threads_per_process,)
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)
        return False
    
    def map(self, func, iterable):
        items = list(iterable)
        futures = {self.executor.submit(func, item): i for i, item in enumerate(items)}
        results = [None] * len(items)
        
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
        
        return results
    
    def map_with_progress(self, func, iterable, desc="Processing"):
        items = list(iterable)
        total = len(items)
        futures = {self.executor.submit(func, item): i for i, item in enumerate(items)}
        results = [None] * total
        completed = 0
        
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            completed += 1
            pct = 100 * completed / total
            bar_len = int(pct / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f"\r  {desc}: [{bar}] {completed}/{total} ({pct:.0f}%)", end="", flush=True)
        
        print()
        return results


# ============================================================
# EM SOLVER
# ============================================================

def create_helmholtz_matrix(n, wavenumber_k):
    h = 1.0 / (n + 1)
    main_diag = -2.0 * np.ones(n)
    off_diag = 1.0 * np.ones(n - 1)
    L1D = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr') / (h ** 2)
    
    I = eye(n, format='csr')
    L2D = kron(I, L1D) + kron(L1D, I)
    
    N = n * n
    A = L2D + (wavenumber_k ** 2) * eye(N, format='csr')
    A = A + 1j * 0.01 * wavenumber_k * eye(N, format='csr')
    
    return A


def em_solve_at_frequency(frequency_index):
    pid = os.getpid()
    cutoff_index = 10
    
    np.random.seed(frequency_index)
    
    if frequency_index < cutoff_index:
        n = 30
        wavenumber_k = 1.0 + frequency_index * 0.5
        maxiter = 500
    else:
        above_cutoff = frequency_index - cutoff_index
        n = 35 + above_cutoff * 3
        wavenumber_k = 6.0 + above_cutoff * 2.0
        maxiter = 1000 + above_cutoff * 100
    
    A = create_helmholtz_matrix(n, wavenumber_k)
    b = np.zeros(n * n, dtype=complex)
    b[n * n // 2] = 1.0
    
    iteration_count = [0]
    def callback(xk):
        iteration_count[0] += 1
    
    start = time.perf_counter()
    solution, info = gmres(A, b, rtol=1e-6, maxiter=maxiter,
                           restart=min(100, n*n),
                           callback=callback, callback_type='x')
    elapsed = time.perf_counter() - start
    
    return {
        'freq_index': frequency_index,
        'pid': pid,
        'time': elapsed,
        'iterations': iteration_count[0],
        'converged': info == 0,
    }


# ============================================================
# BENCHMARK
# ============================================================

if __name__ == "__main__":
    
    NUM_CORES = multiprocessing.cpu_count()
    
    print("=" * 70)
    print("FINDING OPTIMAL CONFIGURATION FOR YOUR SYSTEM")
    print("=" * 70)
    print(f"\nDetected cores/threads: {NUM_CORES}")
    print(f"Current OpenBLAS threads: 16 (from your output)")
    print("\nTesting different configurations...")
    
    frequencies = list(range(0, 20))
    
    # Configurations to test (workers × threads ≤ NUM_CORES is ideal)
    configs = [
        # (workers, threads_per_process)
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
        (2, 8),
        (4, 8),
        (8, 8),
        (16, 8),
        (2, 7),
        (7, 2),
        (14, 1),
        (1, 14)
    ]
    
    results_summary = []
    
    # Sequential baseline
    print(f"\n{'─' * 70}")
    print("Sequential (baseline):")
    print(f"{'─' * 70}")
    start = time.perf_counter()
    seq_results = [em_solve_at_frequency(f) for f in frequencies]
    seq_time = time.perf_counter() - start
    print(f"  Time: {seq_time:.2f}s")
    
    results_summary.append({
        'config': 'Sequential',
        'workers': 1,
        'threads': 16,
        'total_threads': 16,
        'time': seq_time,
        'speedup': 1.0
    })
    
    # Test parallel configurations
    for workers, threads in configs:
        total = workers * threads
        desc = f"{workers} workers × {threads} thread{'s' if threads > 1 else ''} = {total} total"
        
        print(f"\n{'─' * 70}")
        print(f"{desc}")
        print(f"{'─' * 70}")
        
        try:
            start = time.perf_counter()
            
            with ParalleliseCode(max_workers=workers, threads_per_process=threads) as parallel:
                results = parallel.map_with_progress(em_solve_at_frequency, frequencies)
            
            elapsed = time.perf_counter() - start
            speedup = seq_time / elapsed
            
            results_summary.append({
                'config': desc,
                'workers': workers,
                'threads': threads,
                'total_threads': total,
                'time': elapsed,
                'speedup': speedup
            })
            
            print(f"  Time: {elapsed:.2f}s | Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    # =========================================================
    # RESULTS SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Sort by time
    results_summary.sort(key=lambda x: x['time'])
    best = results_summary[0]
    
    print(f"\n  {'Configuration':<40} | {'Time':>7} | {'Speedup':>7}")
    print("  " + "─" * 60)
    
    for i, r in enumerate(results_summary):
        marker = " ★ BEST" if r == best else ""
        time_bar_len = int((best['time'] / r['time']) * 15) if r['time'] > 0 else 0
        time_bar = '█' * time_bar_len
        print(f"  {r['config']:<40} | {r['time']:>6.2f}s | {r['speedup']:>6.2f}x {marker}")
    
    # Visual comparison
    print("\n" + "─" * 70)
    print("Visual Comparison (shorter = faster):")
    print("─" * 70)
    
    max_time = max(r['time'] for r in results_summary)
    for r in results_summary:
        bar_len = int((r['time'] / max_time) * 40)
        bar = '█' * bar_len
        marker = " ★" if r == best else ""
        print(f"  {r['config']:<40} | {bar}{marker}")
    
    # =========================================================
    # RECOMMENDATION
    # =========================================================
    print("\n" + "=" * 70)
    print("RECOMMENDATION FOR YOUR SYSTEM")
    print("=" * 70)
    
    # Find best parallel config (excluding sequential)
    parallel_results = [r for r in results_summary if r['workers'] > 1]
    best_parallel = min(parallel_results, key=lambda x: x['time'])
    
    print(f"""
    Your system: {NUM_CORES} threads available
    
    Best configuration found:
    ┌────────────────────────────────────────────────────────────┐
    │  Workers (processes):     {best_parallel['workers']:<4}                            │
    │  Threads per process:     {best_parallel['threads']:<4}                            │
    │  Total threads:           {best_parallel['total_threads']:<4}                            │
    │  Time:                    {best_parallel['time']:.2f}s                          │
    │  Speedup vs sequential:   {best_parallel['speedup']:.2f}x                          │
    └────────────────────────────────────────────────────────────┘
    
    Use this in your code:
    
        with ParalleliseCode(max_workers={best_parallel['workers']}, threads_per_process={best_parallel['threads']}) as parallel:
            results = parallel.map(your_solver, frequencies)
    """)
    
    # Efficiency analysis
    print("─" * 70)
    print("Efficiency Analysis:")
    print("─" * 70)
    
    theoretical_max = NUM_CORES  # Theoretical max speedup
    efficiency = (best_parallel['speedup'] / theoretical_max) * 100
    
    print(f"""
    Theoretical max speedup:  {theoretical_max:.1f}x (if perfectly parallel)
    Achieved speedup:         {best_parallel['speedup']:.2f}x
    Parallel efficiency:      {efficiency:.1f}%
    
    Note: Efficiency < 100% is normal due to:
      - Process creation overhead
      - Uneven workload distribution
      - Memory bandwidth limits
      - OS scheduling overhead
    """)