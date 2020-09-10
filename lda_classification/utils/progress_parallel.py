from tqdm.auto import tqdm
from joblib import Parallel


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, chunk_size=None, *args,
                 **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self.chunk_size = chunk_size
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        unit_scale = self.chunk_size if self.chunk_size is not None else False
        with tqdm(disable=not self._use_tqdm, total=self._total,
                  unit_scale=unit_scale) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
