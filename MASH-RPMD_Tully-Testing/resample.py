"""
resample.py -- the stage update of MASH-RPMD_Tully-Testing in plain numpy: the weight of every
trajectory endpoint, and the histogram resampling of the signed endpoint weights into the next
stage's points.  Imported by iterate.py; nothing here touches the trajectories' files.

--- representation ---------------------------------------------------------------------------------

A stage's density operator rho(q,p) = rho_I I + rho_z sigma_z + rho_x sigma_x + rho_y sigma_y is a set
of points (q_i, p_i), each carrying a weight vector w_i = (w_I, w_z, w_x, w_y) (COMPONENTS order):

    int dq dp rho_k(q,p) f(q,p)  ~  sum_i w_k,i f(q_i, p_i)

so the weights are signed and already carry the phase-space measure.  'shared' points carry all four
weights; 'separate' points are one-hot (one nonzero component each).

--- the update (one stage, seg_time) ---------------------------------------------------------------

Point i spawns n_spin trajectories, each with its own spin S^(0) drawn uniformly on the whole sphere.
With S(t), q(t) the trajectory at time t, the new components are (the factor 1/2 makes the zero-time
map the identity, e.g. 6 <S_x S_x> = 2 over the sphere, and conserves Tr rho = 2 int rho_I on average)

    rho_I(t) = 1/2 sum_j a_P,j                / n_spin    deposited at (q_j(t), p_j(t))
    rho_z(t) = 1/2 sum_j a_P,j sgn(S_z,j(t))  / n_spin
    rho_x(t) = 1/2 sum_j a_C,j S_x,j(t)       / n_spin
    rho_y(t) = 1/2 sum_j a_C,j S_y,j(t)       / n_spin

    a_P = 4 (w_I |S_z^(0)| + w_z sgn(S_z^(0))|S_z^(0)| + w_x S_x^(0) + w_y S_y^(0))     (population-type finals)
    a_C = 4 (w_I + w_z sgn(S_z^(0))) + 6 (w_x S_x^(0) + w_y S_y^(0))                     (coherence-type finals)

term by term the 16 integrals of the iteration (w of the trajectory's source point).  Beads: S is
the bead mean, sgn(S_z) the bead mean of sign(S_z) (all beads share one spin under the centroid
approximation, so this is only a convention), q the centroid.

--- resampling: histogram cancellation of the signed weights ----------------------------------------

The endpoints form a signed point set W_j (n, 4).  Positive and negative weights that land close
together should cancel before anything is drawn, otherwise the draws keep both signs and the
total weight sum_j |W_j| (hence the variance) grows every stage.  So:

  1. grid the endpoints on a (q, p) histogram (`bins` = (nq, np) over their range) and add the
     weights per bin: B_b = sum_{j in b} W_j.  Opposite signs inside a cell cancel here.
  2. draw n_out bins by stratified (low-variance) resampling with probability
        shared   : p_b = sum_k |B_k,b| / sum_b' sum_k |B_k,b'|     and weight vector B_b / (n_out p_b)
        separate : per component k, p_b = |B_k,b| / sum |B_k|    and weight sgn(B_k,b) sum|B_k| / n_out (one-hot)
     Both are unbiased: the expected resampled measure is the binned one.
  3. place each draw ('placement'): 'endpoint' -- a real endpoint of the bin, chosen with
     probability proportional to its |W| (keeps the points on the dynamical support); 'uniform' --
     uniformly inside the bin (exact sampling of the piecewise-constant histogram); 'center'.

bins=None uses no grid: only endpoints at exactly the same (q, p) share a cell (e.g. the n_pool_init
samples of a delta, or trajectories that never left one surface).  That is unbiased, with no binning
error, but cancels nothing between nearby points, so sum|W| grows from stage to stage.  A grid trades
a bias of order one cell width for that cancellation.
"""

import numpy as np

COMPONENTS = ('I', 'z', 'x', 'y')


def heaviside(x):
    """h(x) with h(0) = 1/2."""
    return 0.5 * (1.0 + np.sign(x))


# ------------------------------------------------------------------ the update weights
def source_factors(w, S0):
    """
    a_P, a_C of every trajectory (see the module docstring).
    w  : (n, 4) weight vector of each trajectory's source point
    S0 : (Sx0, Sy0, sgnz0, Sz0), each (n,): the spin at the start of the stage -- bead means, with
         sgnz0 the bead mean of sign(S_z)
    """
    w = np.asarray(w, float)
    wI, wz, wx, wy = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
    Sx0, Sy0, sg0, Sz0 = S0
    aP = 4.0 * (wI * np.abs(Sz0) + wz * sg0 * np.abs(Sz0) + wx * Sx0 + wy * Sy0)
    aC = 4.0 * (wI + wz * sg0) + 6.0 * (wx * Sx0 + wy * Sy0)
    return aP, aC


def endpoint_weights(aP, aC, S, n_spin):
    """
    (W_I, W_z, W_x, W_y) of every trajectory at one time, shape (n, 4).
    S : (Sx, Sy, sgnz) at that time, each (n,) -- bead means, sgnz the bead mean of sign(S_z)
    """
    Sx, Sy, sg = S
    c = 0.5 / float(n_spin)
    return c * np.stack([aP, aP * sg, aC * Sx, aC * Sy], axis=-1)


# ------------------------------------------------------------------ histogram
def _edges(x, n):
    # every point at one value (a delta) gives zero-width cells: _index puts all of them in the last
    # one, and 'uniform'/'center' placement then returns that value exactly
    return np.linspace(float(np.min(x)), float(np.max(x)), int(n) + 1)


def _index(x, edges):
    """Bin of each x; the upper edge belongs to the last bin."""
    return np.clip(np.searchsorted(edges, x, side='right') - 1, 0, edges.size - 2)


def stratified(mass, n, rng):
    """
    Counts per cell of n stratified-resampling draws with probability proportional to mass: one
    independent uniform in each of n equal-probability strata.  Unbiased and low-variance like
    systematic resampling, but without its single shared offset, which aliases with any periodic
    pattern in the cell order (e.g. alternating signs) and can then draw every point from one sign.
    """
    mass = np.asarray(mass, float)
    cum = np.cumsum(mass / mass.sum())
    cum[-1] = 1.0
    u = (np.arange(n) + rng.random(n)) / n
    return np.bincount(np.searchsorted(cum, u, side='right'), minlength=mass.size)


def _place(drawn, cell, q, p, lw, rng, placement, eq=None, ep=None, nqp=None):
    """
    Locations of the draws: `drawn` (n_out,) cell ids, `cell` (n,) cell of every endpoint, lw (n,)
    the endpoint placement weights.
    """
    if placement == 'endpoint' or eq is None:
        order = np.argsort(cell, kind='stable')
        cw = np.cumsum(lw[order])
        first = np.searchsorted(cell[order], drawn, side='left')
        last = np.searchsorted(cell[order], drawn, side='right') - 1
        lo = np.where(first > 0, cw[np.maximum(first - 1, 0)], 0.0)
        hi = cw[last]
        u = lo + rng.random(drawn.size) * (hi - lo)
        k = np.clip(np.searchsorted(cw, u, side='right'), first, last)
        j = order[k]
        return q[j].copy(), p[j].copy()
    iq, ip = np.divmod(drawn, nqp[1])
    if placement == 'center':
        return 0.5 * (eq[iq] + eq[iq + 1]), 0.5 * (ep[ip] + ep[ip + 1])
    if placement == 'uniform':
        return (eq[iq] + rng.random(drawn.size) * (eq[iq + 1] - eq[iq]),
                ep[ip] + rng.random(drawn.size) * (ep[ip + 1] - ep[ip]))
    raise ValueError(f"placement must be 'endpoint', 'uniform' or 'center', got {placement!r}")


def resample(q, p, W, n_out, bins=(64, 64), sharing='shared', placement='endpoint', rng=None):
    """
    Resample the signed endpoint set (q, p, W) into new points.

    q, p : (n,) endpoint position and momentum (centroid);  W : (n, 4) endpoint weights
    n_out: draws ('shared') or draws per nonzero component ('separate')
    bins : (nq, np) grid over the endpoints' range, or None (cells = exactly coincident points)
    Returns (q_new, p_new, w_new (n_new, 4), info) with info the diagnostics (per component):
        raw_l1 = sum|W|, binned_l1 = sum|B| (raw - binned = what cancelled), net = sum W (exactly
        preserved by binning), out_l1 / out_net of the drawn weights, occupied cells, distinct draws.
    """
    rng = np.random.default_rng() if rng is None else rng
    q = np.asarray(q, float); p = np.asarray(p, float); W = np.asarray(W, float)
    absW = np.abs(W)
    live = absW.sum(axis=1) > 0.0
    if not live.any():
        raise ValueError('every endpoint weight is zero: nothing to resample')
    q, p, W, absW = q[live], p[live], W[live], absW[live]

    if bins is None:
        # no grid: only points at exactly the same (q, p) share a cell (zero width, so no bias)
        cell = np.unique(np.stack([q, p], axis=1), axis=0, return_inverse=True)[1].ravel()
        ncell, eq, ep, nqp = int(cell.max()) + 1, None, None, None
    else:
        nqp = (int(bins[0]), int(bins[1]))
        eq, ep = _edges(q, nqp[0]), _edges(p, nqp[1])
        cell = _index(q, eq) * nqp[1] + _index(p, ep)
        ncell = nqp[0] * nqp[1]
    B = np.stack([np.bincount(cell, weights=W[:, k], minlength=ncell) for k in range(4)], axis=1)

    if sharing == 'shared':
        mass = np.abs(B).sum(axis=1)
        drawn = np.repeat(np.arange(ncell), stratified(mass, n_out, rng))
        w_new = B[drawn] * (mass.sum() / (n_out * mass[drawn]))[:, None]
        q_new, p_new = _place(drawn, cell, q, p, absW.sum(axis=1), rng, placement, eq, ep, nqp)
    elif sharing == 'separate':
        qs, ps, ws = [], [], []
        for k in range(4):
            mass = np.abs(B[:, k])
            if mass.sum() == 0.0:
                continue
            drawn = np.repeat(np.arange(ncell), stratified(mass, n_out, rng))
            wk = np.zeros((drawn.size, 4))
            wk[:, k] = np.sign(B[drawn, k]) * mass.sum() / n_out
            qk, pk = _place(drawn, cell, q, p, absW[:, k], rng, placement, eq, ep, nqp)
            qs.append(qk); ps.append(pk); ws.append(wk)
        q_new, p_new, w_new = np.concatenate(qs), np.concatenate(ps), np.concatenate(ws)
    else:
        raise ValueError(f"sharing must be 'shared' or 'separate', got {sharing!r}")

    info = dict(raw_l1=absW.sum(axis=0), binned_l1=np.abs(B).sum(axis=0), net=W.sum(axis=0),
                out_l1=np.abs(w_new).sum(axis=0), out_net=w_new.sum(axis=0),
                n_occupied=int((np.abs(B).sum(axis=1) > 0).sum()), n_cells=int(ncell),
                gridded=bins is not None,
                n_distinct=int(np.unique(np.stack([q_new, p_new]), axis=1).shape[1]),
                n_endpoints=int(q.size))
    return q_new, p_new, w_new, info


# ------------------------------------------------------------------ stage 0
def sample_rho_init(spec, n_pool, rng):
    """
    A signed pool (q, p, W) drawn from the initial components: n_pool samples per nonzero component,
    one-hot weights coef / n_pool.  spec[k] is None or {'coef', 'q0', 'p0', 'ga'}: a normalized delta
    at (q0, p0) for ga None, else MASH-RPMD_Tully's Gaussian q ~ N(q0, 1/(2 ga)), p ~ N(p0, ga/2).
    """
    qs, ps, Ws = [], [], []
    for k, name in enumerate(COMPONENTS):
        c = spec.get(name)
        if c is None or float(c['coef']) == 0.0:
            continue
        if c.get('ga') is None:
            qk = np.full(n_pool, float(c['q0']))
            pk = np.full(n_pool, float(c['p0']))
        else:
            ga = float(c['ga'])
            qk = rng.normal(float(c['q0']), np.sqrt(1.0 / (2.0 * ga)), n_pool)
            pk = rng.normal(float(c['p0']), np.sqrt(ga / 2.0), n_pool)
        Wk = np.zeros((n_pool, 4))
        Wk[:, k] = float(c['coef']) / n_pool
        qs.append(qk); ps.append(pk); Ws.append(Wk)
    if not qs:
        raise ValueError('rho_init has no nonzero component')
    return np.concatenate(qs), np.concatenate(ps), np.concatenate(Ws)
