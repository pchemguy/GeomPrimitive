import time
from pathlib import Path

from colorama import Fore, Style


class RunSummary:
    """Collects runtime statistics and prints a colorized summary footer."""
  
    def __init__(self, total_jobs: int, log_path: Path):
        self.start_time = time.time()
        self.total_jobs = total_jobs
        self.failed_jobs = 0
        self.completed_jobs = 0
        self.log_path = log_path
  
    def record_result(self, success: bool):
        self.completed_jobs += 1
        if not success:
            self.failed_jobs += 1
  
    def finalize(self):
        end_time = time.time()
        duration = end_time - self.start_time
        ok = self.total_jobs - self.failed_jobs
        throughput = (ok / duration) if duration > 0 else 0.0
  
        sep = Style.BRIGHT + Fore.WHITE
        title = Style.BRIGHT + Fore.CYAN
        thr_color = Fore.GREEN if self.failed_jobs == 0 else Fore.RED
        reset = Style.RESET_ALL
  
        lines = [
            "",
            f"{sep}{'=' * 78}{reset}",
            f"{title}RUN SUMMARY{reset}",
            f"{sep}{'=' * 78}{reset}",
            f"Start Time : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}",
            f"End Time   : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}",
            f"Duration   : {duration:.2f} seconds",
            f"Jobs Total : {self.total_jobs}",
            f"Jobs OK    : {ok}",
            f"Jobs Failed: {self.failed_jobs}",
            f"Throughput : {thr_color}{throughput:.2f} images/sec{reset}",
            f"Log File   : {self.log_path.resolve()}",
            f"{sep}{'=' * 78}{reset}",
            "",
        ]
        for line in lines:
            print(line)
