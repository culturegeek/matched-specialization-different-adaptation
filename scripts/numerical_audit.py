"""Explicit numeric tolerance for newly recomputed floating tensors only."""
import platform
import numpy as np

class TensorAudit:
    def __init__(self, tolerance=0.0):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        self.tolerance=float(tolerance)
        self.groups={}

    def compare(self, actual, expected, kind, label):
        a=actual.detach().cpu().numpy() if hasattr(actual, 'detach') else np.asarray(actual)
        b=expected.detach().cpu().numpy() if hasattr(expected, 'detach') else np.asarray(expected)
        assert a.shape==b.shape and a.dtype==b.dtype, (label, 'shape/dtype mismatch')
        assert np.isfinite(a).all() and np.isfinite(b).all(), (label, 'nonfinite')
        if kind=='v':
            assert (a>=0).all() and (b>=0).all(), (label, 'negative second moment')
        same=bool(np.array_equal(a,b))
        delta=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0.0
        scale=float(np.max(np.abs(b))) if b.size else 0.0
        bound=0.0 if self.tolerance==0 else (1e-12+self.tolerance*scale if kind=='v' else self.tolerance*(1+scale))
        g=self.groups.setdefault(kind,dict(checks=0,bitwise_equal_checks=0,max_absolute_difference=0.0,worst_bound_fraction=0.0))
        g['checks']+=1;g['bitwise_equal_checks']+=int(same)
        g['max_absolute_difference']=max(g['max_absolute_difference'],delta)
        if bound:g['worst_bound_fraction']=max(g['worst_bound_fraction'],delta/bound)
        assert delta<=bound, (label, kind, 'numeric mismatch', delta, bound)
        return same

    def report(self):
        return dict(tolerance=self.tolerance,mode='strict' if self.tolerance==0 else 'explicit_tolerance',groups=self.groups,
                    scope='Only recomputed or replayed floating weights/m/v; hashes, schedules, step counts and transplant identity remain exact.',
                    bounds=dict(weights_and_m='tolerance * (1 + max(abs(reference)))',v='1e-12 + tolerance * max(abs(reference)); zero bound in strict mode'),
                    python=platform.python_version(),numpy=np.__version__)
