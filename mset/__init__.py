"""MSET: multi-scale exact test (Shimonishi and Kawashima, AAAI 2020)

mset.core: p-values of all windows and the distinctive periods and values (re-exported here)
mset.plot: computation and plots of the distinctive periods and values (needs matplotlib)
"""

from .core import (
    TIES,
    MSET_Error,
    MSET_cat,
    MSET_ord,
    all_multinom_cases,
    calc_pvalue_exact,
    check_values,
    find_dperiod,
    find_dperiod_value,
    is_significant,
    multinom_p,
    obs_to_probs,
)
