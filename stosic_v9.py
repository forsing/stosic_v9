from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v9.py — 7-node krug (K=7 / prilagodjenje 7/39) — Cyclical monotonicity (7/39)

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/cyclical_monotonicity.pvs
  — π optimalan za c IFF support(π) je c-ciklično monoton
  — two_cyclic_monotone?: c(i1,j1)+c(i2,j2) ≤ c(i1,j2)+c(i2,j1)
  — thm_cs_implies_2cyclic; Villani: ne možeš poboljšati cikličnom preraspodelom

Mapiranje na 7/39:
  x_k = ko-pojavljivanje na celom CSV; c(i,j)=||x_i-x_j||²
  za svaki uzastopni par kola A→B (oba po 7):
    min-cost assignment A→B (optimalni plan) → support je 2-CM
  Π[i,j] += 1 za svaku matched ivicu
  skor[j] = Σ_{i ∈ last} Π(i,j)
  next = top 7; bez randoma; stop ako 7 uzastopnih
"""

from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from stosic_v1 import EPS, MAX_NUM, N_PICK, load_draws
from stosic_v2 import top7_from_freq
from stosic_v8 import cooccurrence_features, cost_matrix, is_degenerate_consecutive


def optimal_matching_support(
    sources: List[int], targets: List[int], C: np.ndarray
) -> List[tuple[int, int]]:
    """Min-cost bijekcija A→B; support optimalnog plana ⇒ 2-CM (Villani/Stosić)."""
    n = N_PICK
    sub = np.zeros((n, n), dtype=np.float64)
    for a, i in enumerate(sources):
        for b, j in enumerate(targets):
            sub[a, b] = C[i, j]
    row_ind, col_ind = linear_sum_assignment(sub)
    return [(sources[int(r)], targets[int(c)]) for r, c in zip(row_ind, col_ind)]


def accumulate_cm_support(draws: np.ndarray, C: np.ndarray) -> np.ndarray:
    Pi = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
    for t in range(len(draws) - 1):
        src = [int(n) - 1 for n in draws[t]]
        tgt = [int(n) - 1 for n in draws[t + 1]]
        for i, j in optimal_matching_support(src, tgt, C):
            Pi[i, j] += 1.0
    return Pi


def predict_next(draws: np.ndarray) -> List[int]:
    C = cost_matrix(cooccurrence_features(draws))
    Pi = accumulate_cm_support(draws, C)
    skor = np.zeros(MAX_NUM, dtype=np.float64)
    for n in draws[-1]:
        skor += Pi[int(n) - 1, :]
    # blagi tie-break: empirija (deterministički), ne menja CM red ako je jasan
    nu = np.zeros(MAX_NUM, dtype=np.float64)
    for d in draws:
        for n in d:
            nu[int(n) - 1] += 1.0
    nu = nu / max(nu.sum(), EPS)
    combo = top7_from_freq(skor + EPS * nu)
    if is_degenerate_consecutive(combo):
        combo = top7_from_freq(nu)
    return combo


def main():
    draws = load_draws()
    next_combo = predict_next(draws)
    if is_degenerate_consecutive(next_combo):
        raise SystemExit("degenerisan next (uzastopni) — zaustavljen pre ispisa")
    print(next_combo)


if __name__ == "__main__":
    main()



"""
[7, x, 14, y, 23, z, 35]
"""



"""
v9 = cyclical_monotonicity. Iz PVS (2-CM → optimalni matching), pa proveravam da nije degenerisano / duplikat.

v9: cyclical_monotonicity — za svaki A→B min-cost matching (support = 2-CM), skor iz last.
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
