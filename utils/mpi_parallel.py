"""
mpi_parallel.py - Optimized parallel computation for frequency sweeps

Uses chunked parallelization to minimize data transfer overhead.
"""

import os
import sys
import time
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def parallel_frequency_solve(
    frequencies: np.ndarray,
    K_matrix,
    M_matrix,
    B_matrix: np.ndarray,
    n_workers: int = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Solve frequency sweep in parallel using chunked approach.
    
    Divides frequencies into chunks - each worker gets matrices ONCE
    and processes multiple frequencies. Much faster than per-frequency!
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() // 2)
    
    n_freqs = len(frequencies)
    n_excitations = B_matrix.shape[1]
    
    # Don't use more workers than frequencies
    n_workers = min(n_workers, n_freqs)
    
    if show_progress:
        print(f"    Parallel solve: {n_freqs} frequencies, {n_workers} workers")
        print(f"    Matrix size: {K_matrix.shape[0]} DOFs")
    
    # Divide frequencies into chunks
    freq_chunks = np.array_split(np.arange(n_freqs), n_workers)
    
    if show_progress:
        chunk_sizes = [len(c) for c in freq_chunks]
        print(f"    Chunk sizes: {chunk_sizes}")
    
    # Prepare work items - one per worker with its chunk of frequencies
    work_items = []
    for worker_id, chunk_indices in enumerate(freq_chunks):
        if len(chunk_indices) == 0:
            continue
        work_items.append((
            worker_id,
            chunk_indices,
            frequencies[chunk_indices],
            K_matrix,
            M_matrix,
            B_matrix,
        ))
    
    t_start = time.time()
    
    # Execute in parallel
    Z_matrix = np.zeros((n_freqs, n_excitations, n_excitations), dtype=complex)
    
    if show_progress:
        print(f"    Starting {len(work_items)} worker tasks...")
    
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        # Submit all chunks
        futures = [executor.submit(_solve_frequency_chunk, item) for item in work_items]
        
        # Collect results as they complete
        for future in futures:
            try:
                worker_id, chunk_indices, Z_chunk = future.result()
                
                # Store results
                for local_idx, global_idx in enumerate(chunk_indices):
                    Z_matrix[global_idx, :, :] = Z_chunk[local_idx]
                
                if show_progress:
                    elapsed = time.time() - t_start
                    print(f"    Worker {worker_id} done: {len(chunk_indices)} frequencies in {elapsed:.1f}s")
                    
            except Exception as e:
                print(f"    Worker error: {e}")
                import traceback
                traceback.print_exc()
    
    if show_progress:
        elapsed = time.time() - t_start
        rate = n_freqs / elapsed if elapsed > 0 else 0
        print(f"    Total: {elapsed:.2f}s ({rate:.1f} freq/s)")
    
    return Z_matrix


def _solve_frequency_chunk(args: Tuple) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Worker function to solve a CHUNK of frequencies.
    
    This is much more efficient than solving one frequency at a time
    because the matrices are only transferred once per worker.
    """
    import scipy.sparse.linalg as spla
    import numpy as np
    
    worker_id, chunk_indices, frequencies, K, M, B = args
    
    n_freqs_local = len(frequencies)
    n_excitations = B.shape[1]
    ndof = K.shape[0]
    
    # Pre-allocate result array for this chunk
    Z_chunk = np.zeros((n_freqs_local, n_excitations, n_excitations), dtype=complex)
    
    # Solve each frequency in this chunk
    for local_idx, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq
        
        # Build system matrix: A = K - ω²M
        A = K - omega**2 * M
        
        x_all = np.zeros((ndof, n_excitations), dtype=complex)
        
        # Solve for each excitation
        for col in range(n_excitations):
            rhs = omega * B[:, col]
            try:
                x_all[:, col] = spla.spsolve(A, rhs)
            except Exception:
                x_all[:, col], _ = spla.gmres(A, rhs, atol=1e-10, maxiter=1000)
        
        # Z extraction: Z = 1j * B^H @ X
        Z_chunk[local_idx, :, :] = 1j * (B.T.conj() @ x_all)
    
    return worker_id, chunk_indices, Z_chunk


# =============================================================================
# SIMPLE PARALLEL MAP (for other use cases)
# =============================================================================

def parallel_map(
    func: Callable,
    items: List[Any],
    n_workers: int = None,
    show_progress: bool = True,
) -> List[Any]:
    """Apply function to items in parallel."""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() // 2)
    
    n_items = len(items)
    n_workers = min(n_workers, n_items)
    
    if show_progress:
        print(f"parallel_map: {n_items} items, {n_workers} workers")
    
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        results = list(executor.map(func, items))
    
    if show_progress:
        print(f"  Done")
    
    return results


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

class MPIParallel:
    """Simple parallel context manager for backward compatibility."""
    
    def __init__(self, n_processes=None, threads_per_process=1, verbose=True, **kwargs):
        self.n_processes = n_processes or max(1, mp.cpu_count() // 2)
        self.threads_per_process = threads_per_process
        self.verbose = verbose
        self.rank = 0
        self.size = self.n_processes
        self._results = {}
    
    def __enter__(self):
        os.environ['OMP_NUM_THREADS'] = str(self.threads_per_process)
        os.environ['MKL_NUM_THREADS'] = str(self.threads_per_process)
        if self.verbose:
            print(f"MPIParallel: {self.n_processes} workers")
        return self
    
    def __exit__(self, *args):
        return False
    
    @property
    def is_root(self): return True
    
    @property  
    def is_active(self): return True
    
    @property
    def gathered(self): return self._results
    
    def collect(self, key, value): self._results[key] = value
    def gather(self, root=None): return self._results
    def barrier(self): pass
    def broadcast(self, data, root=None): return data
    def log(self, msg, all_ranks=False): print(f"[Parallel] {msg}")