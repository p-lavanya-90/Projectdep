import numpy as np
import warnings

def apply_compat():
    """Patches NumPy 2.x with removed aliases."""
    patched = []
    
    # Restoring np.in1d which librosa still calls in some versions
    if not hasattr(np, "in1d"):
        def _in1d(ar1, ar2, assume_unique=False, invert=False, **kwargs):
            return np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert)
        np.in1d = _in1d
        patched.append("in1d")

    # Restoring np.trapz (renamed to np.trapezoid)
    if not hasattr(np, "trapz"):
        if hasattr(np, "trapezoid"):
            np.trapz = np.trapezoid
            patched.append("trapz")

    # Restoring types
    type_map = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str
    }
    for name, ty in type_map.items():
        if not hasattr(np, name):
            setattr(np, name, ty)
            patched.append(name)

    if patched:
        warnings.warn(f"NumPy 2.x compatibility shim applied for: {', '.join(patched)}", stacklevel=2)

apply_compat()
