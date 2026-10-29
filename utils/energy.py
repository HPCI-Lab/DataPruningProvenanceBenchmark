"""Energy accounting (joules) for pruning, training and evaluation.

Sources (chosen automatically, reported in the `energy_source` CSV column):
  GPU  nvml      NVML hardware energy counter (Volta+), exact.
       nvml-poll NVML power sampled every `poll_s` and integrated (older GPUs).
       none      no NVIDIA GPU / NVML -> GPU energy reported as 0 (flagged).
  CPU  rapl      Intel/AMD RAPL package (+DRAM) counters from /sys/class/powercap  (may need:
                 sudo chmod -R a+r /sys/class/powercap/intel-rapl*). Whole-socket energy, includes other processes.
       estimate  process CPU-seconds * cpu_tdp_w / n_cores (a rough model, NOT a measurement).
Run on an otherwise idle machine; `idle_power_w` (measured at start) lets the plots subtract the idle floor.

Attribution of PRUNING energy. Several pruners cache expensive work across ratios (pixel features, the incremental
k-center order, proxy-model training, KNN-Shapley values). Charging only the cost of each call would make later ratios look
free, so cached work is booked in a ledger and every selection reports two numbers:
  marginal   energy spent inside this call, excluding cache building
  standalone marginal + the full cost of every cached item this selection depends on (transitively)
             = what you would pay to run this method for this ratio on its own  (used for `total_energy_j`)
Pruners register cached work with `setup_scope(key)` (on build/extension) and `touch(key)` (on reuse).
"""
import os
import glob
import time
import threading
import contextlib
import torch


class Cost:
    __slots__ = ("gpu_j", "cpu_j", "s")

    def __init__(self, gpu_j=0.0, cpu_j=0.0, s=0.0):
        self.gpu_j, self.cpu_j, self.s = gpu_j, cpu_j, s

    @property
    def j(self):
        return self.gpu_j + self.cpu_j

    def __add__(self, o):
        return Cost(self.gpu_j + o.gpu_j, self.cpu_j + o.cpu_j, self.s + o.s)

    def __sub__(self, o):
        return Cost(self.gpu_j - o.gpu_j, self.cpu_j - o.cpu_j, self.s - o.s)

    def __iadd__(self, o):
        self.gpu_j += o.gpu_j; self.cpu_j += o.cpu_j; self.s += o.s
        return self

    def clamp(self):
        return Cost(max(self.gpu_j, 0.0), max(self.cpu_j, 0.0), max(self.s, 0.0))


class _Meas:
    def __init__(self):
        self.cost = Cost()


class NullTracker:
    source = "disabled"
    idle_power_w = 0.0

    @contextlib.contextmanager
    def measure(self):
        m = _Meas()
        t0 = time.perf_counter()
        yield m
        m.cost = Cost(0.0, 0.0, time.perf_counter() - t0)

    def measure_idle(self, seconds):
        return 0.0


class EnergyTracker:
    def __init__(self, device="cpu", cpu_tdp_w=65.0, poll_s=0.1, rapl_root="/sys/class/powercap"):
        self.device = str(device)
        self.cpu_tdp_w, self.poll_s, self.rapl_root = cpu_tdp_w, poll_s, rapl_root
        self.idle_power_w = 0.0
        self._lock = threading.Lock()
        self._gpu_j = 0.0
        self._cpu_j = 0.0
        self._last_t = time.perf_counter()
        self._gpu_mode = "none"
        self._init_gpu()
        self._init_cpu()
        self.source = f"gpu:{self._gpu_mode}|cpu:{self._cpu_mode}"
        if self._gpu_mode == "nvml-poll" or self._cpu_mode == "rapl":
            self._stop = threading.Event()
            threading.Thread(target=self._loop, daemon=True).start()

    # ---------------- GPU
    def _init_gpu(self):
        self._nv = None
        if not (self.device.startswith("cuda") and torch.cuda.is_available()):
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            idx = torch.cuda.current_device() if ":" not in self.device else int(self.device.split(":")[1])
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if vis and all(v.strip().isdigit() for v in vis.split(",")):
                idx = int(vis.split(",")[idx])
            self._nv, self._pynvml = pynvml.nvmlDeviceGetHandleByIndex(idx), pynvml
            try:
                self._gpu_base = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nv)       # mJ
                self._gpu_mode = "nvml"
            except Exception:
                pynvml.nvmlDeviceGetPowerUsage(self._nv)
                self._gpu_mode = "nvml-poll"
        except Exception:
            self._nv, self._gpu_mode = None, "none"

    # ---------------- CPU
    def _init_cpu(self):
        self._rapl = []
        try:
            for d in glob.glob(os.path.join(self.rapl_root, "intel-rapl*:*")):
                try:
                    name = open(os.path.join(d, "name")).read().strip()
                    if not (name.startswith("package") or name == "dram"):
                        continue
                    e = int(open(os.path.join(d, "energy_uj")).read())
                    mx = int(open(os.path.join(d, "max_energy_range_uj")).read())
                    self._rapl.append({"p": os.path.join(d, "energy_uj"), "last": e, "max": mx, "acc": 0.0})
                except Exception:
                    continue
        except Exception:
            pass
        self._cpu_mode = "rapl" if self._rapl else "estimate"
        self._ncores = os.cpu_count() or 1

    # ---------------- accumulation
    def _poll_once(self):
        with self._lock:
            now = time.perf_counter()
            if self._gpu_mode == "nvml-poll":
                self._gpu_j += self._pynvml.nvmlDeviceGetPowerUsage(self._nv) / 1000.0 * (now - self._last_t)
            self._last_t = now
            for r in self._rapl:                                   # wrap-safe accumulation
                try:
                    e = int(open(r["p"]).read())
                except Exception:
                    continue
                r["acc"] += (e - r["last"]) % (r["max"] + 1)
                r["last"] = e

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            try:
                self._poll_once()
            except Exception:
                pass

    def _read(self):
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._poll_once()
        if self._gpu_mode == "nvml":
            gpu = (self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nv) - self._gpu_base) / 1000.0
        else:
            gpu = self._gpu_j
        if self._cpu_mode == "rapl":
            cpu = sum(r["acc"] for r in self._rapl) / 1e6
        else:
            cpu = time.process_time() * self.cpu_tdp_w / self._ncores
        return gpu, cpu, time.perf_counter()

    @contextlib.contextmanager
    def measure(self):
        m = _Meas()
        g0, c0, t0 = self._read()
        try:
            yield m
        finally:
            g1, c1, t1 = self._read()
            m.cost = Cost(g1 - g0, c1 - c0, t1 - t0)

    def measure_idle(self, seconds=5.0):
        """Average idle power (W) of GPU+CPU over `seconds`, no workload. Stored in self.idle_power_w."""
        with self.measure() as m:
            time.sleep(seconds)
        # the CPU 'estimate' mode only counts this process's CPU time, which is ~0 while sleeping -> ~0 W (consistent)
        self.idle_power_w = m.cost.j / max(m.cost.s, 1e-9)
        return self.idle_power_w


# ====================================================================== global tracker + ledger
_TRACKER = NullTracker()
_SCOPES = []          # active scopes (innermost last)
SETUP = {}            # key -> {"cost": Cost, "deps": set(keys)}


def set_tracker(t):
    global _TRACKER
    _TRACKER = t if t is not None else NullTracker()
    SETUP.clear()


def get_tracker():
    return _TRACKER


class _Scope:
    def __init__(self, kind):
        self.kind, self.touched, self.setup_cost, self.child_incl = kind, set(), Cost(), Cost()


def touch(key):
    """Record that the current selection depends on cached work `key` (call on a cache HIT)."""
    for sc in _SCOPES:
        sc.touched.add(key)


@contextlib.contextmanager
def setup_scope(key):
    """Wrap building / extending cached work. Cost is booked to the ledger (accumulating on repeated entry)."""
    sc = _Scope("setup")
    _SCOPES.append(sc)
    with _TRACKER.measure() as m:
        try:
            yield
        finally:
            pass
    _SCOPES.remove(sc)
    excl = (m.cost - sc.child_incl).clamp()
    ent = SETUP.setdefault(key, {"cost": Cost(), "deps": set()})
    ent["cost"] += excl
    ent["deps"] |= (sc.touched - {key})
    if _SCOPES and _SCOPES[-1].kind == "setup":
        _SCOPES[-1].child_incl += m.cost
    for p in _SCOPES:
        p.touched.add(key)
        p.touched |= sc.touched
        if p.kind == "prune":
            p.setup_cost += excl


class PruneResult:
    def __init__(self):
        self.marginal = Cost()
        self.standalone = Cost()


@contextlib.contextmanager
def prune_scope():
    """Wrap one `select_subset` call -> PruneResult(marginal, standalone) filled on exit."""
    res = PruneResult()
    sc = _Scope("prune")
    _SCOPES.append(sc)
    with _TRACKER.measure() as m:
        try:
            yield res
        finally:
            pass
    _SCOPES.remove(sc)
    res.marginal = (m.cost - sc.setup_cost).clamp()
    seen, stack = set(), list(sc.touched)
    total = Cost()
    while stack:
        k = stack.pop()
        if k in seen or k not in SETUP:
            continue
        seen.add(k)
        total += SETUP[k]["cost"]
        stack.extend(SETUP[k]["deps"])
    res.standalone = res.marginal + total


ENERGY_FIELDS = [
    "energy_source", "idle_power_w",
    "prune_time_s", "prune_energy_j", "prune_energy_marginal_j", "prune_gpu_j", "prune_cpu_j",
    "train_time_s", "train_energy_j", "train_gpu_j", "train_cpu_j",
    "eval_time_s", "eval_energy_j",
    "total_energy_j",
]
_NAN = float("nan")


def energy_row(prune: PruneResult = None, train: Cost = None, eval_: Cost = None) -> dict:
    """CSV columns. total_energy_j = standalone pruning + training (the cost of PRODUCING the model;
    evaluation is reported separately because it is benchmark overhead)."""
    r = {k: _NAN for k in ENERGY_FIELDS}
    r["energy_source"] = _TRACKER.source
    r["idle_power_w"] = _TRACKER.idle_power_w
    if prune is not None:
        s = prune.standalone
        r.update(prune_time_s=s.s, prune_energy_j=s.j, prune_energy_marginal_j=prune.marginal.j,
                 prune_gpu_j=s.gpu_j, prune_cpu_j=s.cpu_j)
    if train is not None:
        r.update(train_time_s=train.s, train_energy_j=train.j, train_gpu_j=train.gpu_j, train_cpu_j=train.cpu_j)
    if eval_ is not None:
        r.update(eval_time_s=eval_.s, eval_energy_j=eval_.j)
    if prune is not None and train is not None:
        r["total_energy_j"] = prune.standalone.j + train.j
    return r
