def efficiency(cosh, cosl, mkpi, qsq):

    eff = (
        1
        - 0.2 * cosh**2
        + 0.1 * cosh
        - 0.5 * cosl**2
        + 0.005 * mkpi
        + 0.005 * qsq
        + 0.05 * cosl**2 * qsq
    )
    return eff
