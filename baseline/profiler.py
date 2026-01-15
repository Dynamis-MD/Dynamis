"""
CUDA Event-based Profiler for distributed MD simulations.

Provides low-overhead timing using CUDA events instead of synchronization barriers.
"""

import os
import time
import torch


class CUDAEventProfiler:
    """
    CUDA Events-based profiler for detailed per-step timing analysis.
    
    Uses CUDA events instead of torch.cuda.synchronize() for each start/stop,
    reducing synchronization overhead. Only one sync is done at step_done().
    
    Usage:
        profiler = CUDAEventProfiler(enabled=True)
        
        # In each step:
        profiler.start("halo_exchange")
        exchanger.exchange(atoms)
        profiler.stop("halo_exchange")
        
        profiler.step_done()  # End of step (single sync here)
        
        # After episode:
        profiler.report()
    """
    
    def __init__(self, enabled: bool = True, rank: int = 0):
        self.enabled = enabled
        self.rank = rank
        self.step_times = {}          # {category: [times_ms]}
        self.current_step_events = {} # {category: (start_event, end_event)}
        self.step_count = 0
        self.episode_start_time = None
        
    def start_episode(self):
        """Call at the start of each episode."""
        self.step_times = {}
        self.step_count = 0
        self.episode_start_time = time.time()
        
    def record(self, category: str):
        """Context manager for recording time of a category."""
        return _ProfilerContext(self, category)
    
    def start(self, category: str):
        """Start timing for a category using CUDA events."""
        if not self.enabled:
            return
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.current_step_events[category] = (start_event, end_event)
    
    def stop(self, category: str):
        """Stop timing for a category (records end event, no sync)."""
        if not self.enabled:
            return
        if category not in self.current_step_events:
            return
        _, end_event = self.current_step_events[category]
        end_event.record()
    
    def step_done(self):
        """Call at the end of each MD step. Single sync and collect all times."""
        if not self.enabled:
            return
        
        # Single synchronization for all events
        torch.cuda.synchronize()
        
        # Collect elapsed times from CUDA events
        for category, (start_evt, end_evt) in self.current_step_events.items():
            # Skip events that weren't fully recorded (missing stop call)
            # query() returns True if event has been recorded and completed
            if not end_evt.query():
                continue
            elapsed = start_evt.elapsed_time(end_evt)  # ms
            if category not in self.step_times:
                self.step_times[category] = []
            self.step_times[category].append(elapsed)
        
        self.current_step_events = {}
        self.step_count += 1
    
    def report(self):
        """Print profiling report for this episode."""
        if not self.enabled or self.step_count == 0:
            return
            
        episode_time = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        rank_prefix = f"[Rank {self.rank}] " if self.rank > 0 else ""
        
        print(f"\n{rank_prefix}===== CUDA Event Profiling Report ({self.step_count} steps) =====")
        print(f"{'Category':<30} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Total (ms)':<12}")
        print("-" * 78)
        
        total_tracked = 0.0
        for category, times in sorted(self.step_times.items()):
            if times:
                avg = sum(times) / len(times)
                min_t = min(times)
                max_t = max(times)
                total = sum(times)
                total_tracked += total
                print(f"{category:<30} {avg:<12.2f} {min_t:<12.2f} {max_t:<12.2f} {total:<12.2f}")
        
        print("-" * 78)
        avg_per_step = total_tracked / self.step_count if self.step_count > 0 else 0
        episode_ms = episode_time * 1000
        untracked = episode_ms - total_tracked
        
        print(f"{'Tracked Total':<30} {avg_per_step:<12.2f} {'':<12} {'':<12} {total_tracked:<12.2f}")
        print(f"{'Episode Total':<30} {episode_ms/self.step_count:<12.2f} {'':<12} {'':<12} {episode_ms:<12.2f}")
        print(f"{'Untracked':<30} {untracked/self.step_count:<12.2f} {'':<12} {'':<12} {untracked:<12.2f}")
        print(f"{'Steps/s':<30} {self.step_count / episode_time:.2f}")
        print()
        
    def get_summary_dict(self) -> dict:
        """Return summary as dict for logging."""
        if not self.enabled or self.step_count == 0:
            return {}
        
        summary = {"step_count": self.step_count}
        for category, times in self.step_times.items():
            if times:
                summary[f"{category}_avg_ms"] = sum(times) / len(times)
                summary[f"{category}_total_ms"] = sum(times)
        return summary
    
    def save_to_csv(self, output_dir: str, episode: int = 0):
        """Save profiling results to CSV file for this rank.
        
        Args:
            output_dir: Base output directory (will create profiling/ subdir)
            episode: Episode number for filename
        """
        if not self.enabled or self.step_count == 0:
            return
        
        import csv
        
        # Create profiling directory
        profiling_dir = os.path.join(output_dir, "profiling")
        os.makedirs(profiling_dir, exist_ok=True)
        
        # CSV filename: profiling_episode{N}_rank{R}.csv
        csv_path = os.path.join(profiling_dir, f"profiling_episode{episode}_rank{self.rank}.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['category', 'avg_ms', 'min_ms', 'max_ms', 'total_ms', 'count'])
            
            # Data rows
            for category, times in sorted(self.step_times.items()):
                if times:
                    writer.writerow([
                        category,
                        f"{sum(times) / len(times):.4f}",
                        f"{min(times):.4f}",
                        f"{max(times):.4f}",
                        f"{sum(times):.4f}",
                        len(times)
                    ])
            
            # Summary row
            total_tracked = sum(sum(t) for t in self.step_times.values())
            episode_time = time.time() - self.episode_start_time if self.episode_start_time else 0
            writer.writerow([
                '_total_tracked',
                f"{total_tracked / self.step_count:.4f}",
                '', '',
                f"{total_tracked:.4f}",
                self.step_count
            ])
            writer.writerow([
                '_episode_wall_time_ms',
                '', '', '',
                f"{episode_time * 1000:.4f}",
                ''
            ])


class _ProfilerContext:
    """Context manager for CUDAEventProfiler.record()."""
    
    def __init__(self, profiler: 'CUDAEventProfiler', category: str):
        self.profiler = profiler
        self.category = category
        
    def __enter__(self):
        self.profiler.start(self.category)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop(self.category)
        return False


# Alias for backward compatibility
StepProfiler = CUDAEventProfiler
