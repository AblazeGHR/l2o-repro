# Stub for `pymetis` on Windows.
#
# Why: the Neural-QAOA-Squared official code imports `pymetis` at the top of
# `competitors/QAOA-in-QAOA/utilities.py`, but pymetis ships no Windows wheel and
# cannot be compiled on Windows (`sys/resource.h` is a Linux-only header). A
# grep across the repository shows the module is imported but never called by
# the QAOA^2 baseline, so an importable stub is functionally equivalent.
#
# Any real call will raise a clear error instead of silently misbehaving.

def __getattr__(name):
    raise AttributeError(
        f"pymetis is not available on Windows (this is a stub); "
        f"'{name}' is not implemented"
    )


def __dir__():
    return []
