#!/usr/bin/env python3
"""
================================================================================
7D-O2 Closed-Loop Respiratory Control Model — Paper-Accurate Python Translation
================================================================================

Source:
    Diekman CO, Thomas PJ, Wilson CG (2024).
    "COVID-19 and silent hypoxemia in a minimal closed-loop model of the
    respiratory rhythm generator."
    Biological Cybernetics, 118(3-4):145-163.
    DOI: 10.1007/s00422-024-00989-w  |  PMID: 38884785

Original MATLAB:
    ModelDB #229640  (Diekman et al. 2017)
    ModelDB #2015954 (Silent hypoxemia extension, 2024)
    GitHub: https://github.com/ModelDBRepository/229640

Translation:
    - Every equation references Appendix A (Eqs. 7-28)
    - MATLAB ode15s -> scipy Radau (both implicit stiff solvers)
    - Tolerances: rtol = atol = 1e-9 (identical)
    - SH parameters from Results p.7-9: phi=0.24, theta_g=70, sigma_g=36, [Hb]=250

Usage:
    python respiratory_model.py                        # normoxia only
    python respiratory_model.py --compare              # normoxia vs SH + L2 norms
    python respiratory_model.py --compare --output-dir figures

Requirements: numpy, scipy, matplotlib
================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json, os, sys, argparse, time

# =============================================================================
# MATPLOTLIB — publication quality
# =============================================================================
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.linewidth': 1.2, 'lines.linewidth': 1.8,
})

COL_N = '#c0392b'  # normoxia red (matches paper Fig 6)
COL_S = '#2471a3'  # SH blue (matches paper Fig 6)
COL_G = '#e0e0e0'  # grid

# =============================================================================
# PARAMETERS — exact values from Appendix A (Eqs 7-28) and Results (p.7-9)
# =============================================================================

def normoxia_params():
    """Original 7D-O2 model. Source: Appendix A, Eqs 7-28; closedloop.m ModelDB #229640."""
    return dict(
        # CPG: Butera-Rinzel-Smith (Eqs 7-16)
        C=21.0,             # pF
        gK=11.2,            # nS
        gNaP=2.8,           # nS
        gNa=28.0,           # nS
        gL=2.8,             # nS
        EK=-85.0,           # mV
        ENa=50.0,           # mV
        EL=-65.0,           # mV
        Etonic=0.0,         # mV
        theta_n=-29.0, sigma_n=-4.0,      # K+ activation
        theta_p=-40.0, sigma_p=-6.0,      # NaP activation
        theta_h=-48.0, sigma_h=6.0,       # NaP inactivation
        theta_m=-34.0, sigma_m=-5.0,      # fast Na+ activation
        tau_n_bar=10.0,     # ms
        tau_h_bar=10000.0,  # ms
        # Motor pool (Eqs 17-18)
        ra=0.001, rd=0.001, Tmax=1.0, VT=2.0, Kp=5.0,
        # Lung volume (Eq 19) — E1=0.4 L, E2=0.0025 ms^-1
        E1=0.4, E2=0.0025, vol0=2.0,
        # Lung oxygen (Eq 20)
        PextO2=(760.0-47.0)*0.21, tau_LB=500.0, R_gas=62.364, Temp=310.0,
        # Blood oxygen (Eqs 21-27)
        M=8e-6, Hb=150.0, volB=5.0, betaO2=0.03, c=2.5, K=26.0,
        # Chemosensation (Eq 28)
        phi=0.3, theta_g=85.0, sigma_g=30.0,
    )

def sh_params():
    """Silent hypoxemia working model. Source: Results p.7-9, Fig 5B purple curve.
    Quote (p.9): 'Thus we will consider the model with [Hb]=250 as our working model.'"""
    p = normoxia_params()
    p['phi'] = 0.24       # reduced from 0.3  (blunted chemosensory gain)
    p['theta_g'] = 70.0   # shifted from 85   (lower activation threshold)
    p['sigma_g'] = 36.0   # widened from 30   (broader sigmoid)
    p['Hb'] = 250.0       # increased from 150 (polycythemia)
    return p

# =============================================================================
# ODE RIGHT-HAND SIDE — Eqs 7-28
# =============================================================================

DEFAULT_INITS = [-56.8172, 9.5344e-04, 0.7454, 2.0026e-04, 2.0525, 98.9638, 97.7927]

def closedloop_rhs(t, u, p):
    """
    7D-O2 model RHS.
    State: [V, n, h, alpha, volL, PAO2, PaO2]
    Each line references the equation number from Appendix A.
    """
    V, n, h, alpha, volL, PAO2, PaO2 = u
    PaO2 = max(PaO2, 1e-6)

    # Gating functions (Eqs 15-16)
    p_inf = 1.0 / (1.0 + np.exp((V - p['theta_p']) / p['sigma_p']))
    m_inf = 1.0 / (1.0 + np.exp((V - p['theta_m']) / p['sigma_m']))
    n_inf = 1.0 / (1.0 + np.exp((V - p['theta_n']) / p['sigma_n']))
    h_inf = 1.0 / (1.0 + np.exp((V - p['theta_h']) / p['sigma_h']))
    tau_n = p['tau_n_bar'] / np.cosh((V - p['theta_n']) / (2.0 * p['sigma_n']))
    tau_h = p['tau_h_bar'] / np.cosh((V - p['theta_h']) / (2.0 * p['sigma_h']))

    # Ionic currents (Eqs 10-14)
    IK   = p['gK']   * n**4 * (V - p['EK'])                          # Eq 10
    INaP = p['gNaP'] * p_inf * h * (V - p['ENa'])                    # Eq 11
    INa  = p['gNa']  * m_inf**3 * (1.0 - n) * (V - p['ENa'])        # Eq 12
    IL   = p['gL']   * (V - p['EL'])                                  # Eq 13

    # Chemosensory feedback (Eq 28)
    gtonic = p['phi'] * (1.0 - np.tanh((PaO2 - p['theta_g']) / p['sigma_g']))
    Itonic = gtonic * (V - p['Etonic'])                               # Eq 14

    # Motor pool (Eqs 17-18)
    T_conc = p['Tmax'] / (1.0 + np.exp(-(V - p['VT']) / p['Kp']))

    # Lung volume (Eq 19)
    dvolL = p['E1'] * alpha - p['E2'] * (volL - p['vol0'])

    # Conversion factors (Eqs 26-27)
    zeta = p['volB'] / 22400.0
    eta  = p['Hb'] * 1.36

    # Hemoglobin saturation and derivative (Eqs 24-25)
    cc, KK = p['c'], p['K']
    SaO2     = PaO2**cc / (PaO2**cc + KK**cc)                        # Eq 24
    dSaO2_dP = cc * PaO2**(cc-1) * (
        1.0/(PaO2**cc + KK**cc) - PaO2**cc/(PaO2**cc + KK**cc)**2)   # Eq 25

    # Oxygen fluxes (Eqs 22-23)
    JLB = (PAO2 - PaO2) / p['tau_LB'] * (volL / (p['R_gas'] * p['Temp']))  # Eq 22
    JBT = p['M'] * zeta * (p['betaO2'] * PaO2 + eta * SaO2)               # Eq 23

    # Lung oxygen (Eq 20)
    dvolL_pos = max(0.0, dvolL)
    dPAO2 = ((p['PextO2'] - PAO2) / volL) * dvolL_pos - (PAO2 - PaO2) / p['tau_LB']

    # Blood oxygen (Eq 21)
    denom = max(zeta * (p['betaO2'] + eta * dSaO2_dP), 1e-15)
    dPaO2 = (JLB - JBT) / denom

    return [
        (-IK - INaP - INa - IL - Itonic) / p['C'],   # dV/dt     Eq 7
        (n_inf - n) / tau_n,                           # dn/dt     Eq 8
        (h_inf - h) / tau_h,                           # dh/dt     Eq 9
        p['ra'] * T_conc * (1.0 - alpha) - p['rd'] * alpha,  # dalpha/dt Eq 17
        dvolL,                                         # dvolL/dt  Eq 19
        dPAO2,                                         # dPAO2/dt  Eq 20
        dPaO2,                                         # dPaO2/dt  Eq 21
    ]

# =============================================================================
# SIMULATION
# =============================================================================

def run_sim(params, tf=15000.0, inits=None):
    """Integrate with scipy Radau (equivalent to MATLAB ode15s for stiff systems)."""
    if inits is None:
        inits = list(DEFAULT_INITS)
    sol = solve_ivp(
        fun=lambda t, u: closedloop_rhs(t, u, params),
        t_span=[0, tf], y0=inits,
        method='Radau', rtol=1e-9, atol=1e-9, max_step=5.0,
        dense_output=True,
    )
    if not sol.success:
        print(f"  WARNING: {sol.message}", file=sys.stderr)
    return sol

# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

def compute_gtonic(PaO2, p):
    return p['phi'] * (1.0 - np.tanh((PaO2 - p['theta_g']) / p['sigma_g']))

def compute_SaO2(PaO2, p):
    P = np.maximum(PaO2, 1e-6)
    return P**p['c'] / (P**p['c'] + p['K']**p['c'])

# =============================================================================
# L2 NORM
# =============================================================================

def l2_norm(t1, y1, t2, y2, vi, N=5000):
    ts, te = max(t1[0],t2[0]), min(t1[-1],t2[-1])
    tc = np.linspace(ts, te, N)
    i1 = np.interp(tc, t1, y1[vi])
    i2 = np.interp(tc, t2, y2[vi])
    d = i1 - i2
    dt = (te - ts) / (N - 1)
    return tc, i1, i2, d, np.sqrt(np.sum(d**2) * dt / (te - ts))

# =============================================================================
# FIGURES
# =============================================================================

def fig1_normoxia(sol, p, path):
    """Replicate paper Figure 1: 6-panel normoxia time series."""
    t = sol.t / 1000.0
    V, n, h, alpha, volL, PAO2, PaO2 = sol.y
    gt = compute_gtonic(PaO2, p)

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(r"Figure 1 — 7D-O$_2$ Normoxia ($M=8\times10^{-6}$ ms$^{-1}$)",
                 fontsize=15, fontweight='bold', y=0.98)
    for a, y, yl, ylm in [
        (ax[0,0], V, r'$V$ (mV)', [-70,20]),
        (ax[0,1], alpha, r'$\alpha$', None),
        (ax[0,2], volL, r'vol$_\mathrm{L}$ (L)', [1.9,3.1]),
        (ax[1,0], gt, r'$g_\mathrm{tonic}$ (nS)', None),
        (ax[1,1], PaO2, r'$P_\mathrm{a}$O$_2$ (mmHg)', None),
        (ax[1,2], PAO2, r'$P_\mathrm{A}$O$_2$ (mmHg)', None),
    ]:
        a.plot(t, y, 'k', lw=1.5)
        a.set_ylabel(yl); a.set_xlabel(r'$t$ (s)'); a.set_xlim([0, t[-1]])
        if ylm: a.set_ylim(ylm)
        a.grid(True, alpha=0.2, color=COL_G)
    plt.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")

def fig6_comparison(sn, pn, ss, ps, path):
    """Replicate paper Figure 6 structure: normoxia (red) vs SH (blue)."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Figure 6 — Normoxia (red) vs Silent Hypoxemia (blue)",
                 fontsize=14, fontweight='bold', y=0.99)
    for i, (vi, yl, tt) in enumerate([
        (0, r'$V$ (mV)', 'Membrane Potential'),
        (4, r'vol$_\mathrm{L}$ (L)', 'Lung Volume'),
        (6, r'$P_\mathrm{a}$O$_2$ (mmHg)', 'Arterial O$_2$'),
    ]):
        a = ax.flat[i]
        a.plot(sn.t/1000, sn.y[vi], color=COL_N, lw=1.3, label='Normoxia')
        a.plot(ss.t/1000, ss.y[vi], color=COL_S, lw=1.3, label='SH ([Hb]=250)')
        a.set_ylabel(yl); a.set_xlabel(r'$t$ (s)'); a.set_title(tt, fontsize=12)
        a.legend(frameon=False); a.grid(True, alpha=0.2, color=COL_G)
    a = ax[1,1]
    a.plot(sn.t/1000, compute_SaO2(sn.y[6], pn)*100, color=COL_N, lw=1.3, label='Normoxia')
    a.plot(ss.t/1000, compute_SaO2(ss.y[6], ps)*100, color=COL_S, lw=1.3, label='SH')
    a.axhline(80, color='gray', ls='--', lw=0.8)
    a.set_ylabel(r'SaO$_2$ (%)'); a.set_xlabel(r'$t$ (s)'); a.set_title('Oxygen Saturation')
    a.legend(frameon=False); a.grid(True, alpha=0.2, color=COL_G)
    plt.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")

def fig_l2(sn, ss, path):
    """L2 norm bar chart + PaO2 overlay."""
    names = ['V','n','h',r'$\alpha$','volL',r'PAO$_2$',r'PaO$_2$']
    keys = ['V','n','h','alpha','volL','PAO2','PaO2']
    norms = {}
    for i, k in enumerate(keys):
        _, _, _, _, v = l2_norm(sn.t, sn.y, ss.t, ss.y, i)
        norms[k] = v

    tc, i1, i2, d, l2p = l2_norm(sn.t, sn.y, ss.t, ss.y, 6)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(r"L$_2$ Norm Analysis: Normoxia vs Silent Hypoxemia", fontsize=14, fontweight='bold')

    bars = a1.bar(names, list(norms.values()), color=[COL_S]*4+[COL_N]*3, edgecolor='white', lw=0.8)
    for b, v in zip(bars, norms.values()):
        a1.text(b.get_x()+b.get_width()/2, b.get_height()+max(norms.values())*0.02,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    a1.set_ylabel(r'L$_2$ Norm'); a1.set_title('Per-Variable'); a1.grid(True, axis='y', alpha=0.2)

    a2.plot(tc/1000, i1, color=COL_N, lw=1.2, label='Normoxia')
    a2.plot(tc/1000, i2, color=COL_S, lw=1.2, label='SH')
    a2.fill_between(tc/1000, i1, i2, alpha=0.12, color=COL_S)
    a2.set_ylabel(r'$P_\mathrm{a}$O$_2$ (mmHg)'); a2.set_xlabel(r'$t$ (s)')
    a2.set_title(f'PaO$_2$ | L$_2$ = {l2p:.3f}'); a2.legend(frameon=False); a2.grid(True, alpha=0.2)

    plt.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")
    return norms

def fig_dissociation(path):
    """Oxyhemoglobin dissociation curves for various K (replicates Fig 4A style)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    PO2 = np.linspace(0, 150, 500)
    for K, col, lbl in [(26, COL_N, 'K=26 (default)'), (14, COL_S, 'K=14'),
                         (20, '#7d3c98', 'K=20'), (32, '#117a65', 'K=32')]:
        p = normoxia_params(); p['K'] = K
        ax.plot(PO2, compute_SaO2(PO2, p)*100, color=col, lw=2, label=lbl)
    ax.axhline(80, color='gray', ls='--', lw=0.8)
    ax.set_xlabel(r'$P_\mathrm{a}$O$_2$ (mmHg)'); ax.set_ylabel(r'SaO$_2$ (%)')
    ax.set_title('Oxyhemoglobin Dissociation Curve (Eq. 24)', fontsize=14, fontweight='bold')
    ax.legend(frameon=False); ax.grid(True, alpha=0.2); ax.set_xlim([0,150]); ax.set_ylim([0,105])
    plt.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")

def fig_sigmoid(path):
    """Chemosensory sigmoid for normoxia and SH (replicates Fig 3A style)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    PO2 = np.linspace(0, 150, 500)
    pn, ps = normoxia_params(), sh_params()
    ax.plot(PO2, compute_gtonic(PO2, pn), color=COL_N, lw=2,
            label=rf'Normoxia ($\phi$=0.3, $\theta_g$=85, $\sigma_g$=30)')
    ax.plot(PO2, compute_gtonic(PO2, ps), color=COL_S, lw=2,
            label=rf'SH ($\phi$=0.24, $\theta_g$=70, $\sigma_g$=36)')
    ax.set_xlabel(r'$P_\mathrm{a}$O$_2$ (mmHg)'); ax.set_ylabel(r'$g_\mathrm{tonic}$ (nS)')
    ax.set_title('Chemosensory Feedback (Eq. 28)', fontsize=14, fontweight='bold')
    ax.legend(frameon=False); ax.grid(True, alpha=0.2); ax.set_xlim([0,150])
    plt.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")

# =============================================================================
# EXPORT
# =============================================================================

def export_csv(sol, p, path):
    t = sol.t; V,n,h,a,vL,PA,Pa = sol.y
    g = compute_gtonic(Pa, p); S = compute_SaO2(Pa, p)*100
    np.savetxt(path, np.column_stack([t,V,n,h,a,vL,PA,Pa,g,S]), delimiter=',',
               header="t_ms,V_mV,n,h,alpha,volL_L,PAO2_mmHg,PaO2_mmHg,gtonic_nS,SaO2_pct",
               comments='', fmt='%.8e')
    print(f"  Exported: {path}")

def export_json(p, path):
    with open(path, 'w') as f: json.dump(p, f, indent=2)
    print(f"  Exported: {path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="7D-O2 Model — Paper-Accurate Python")
    ap.add_argument('--compare', action='store_true', help='Run normoxia vs SH comparison')
    ap.add_argument('--tf', type=float, default=15000.0, help='Final time in ms')
    ap.add_argument('--output-dir', type=str, default='figures', help='Output directory')
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    out = lambda f: os.path.join(a.output_dir, f)

    print("="*72)
    print("  7D-O2 Closed-Loop Respiratory Control Model")
    print("  Diekman, Thomas & Wilson (2017, 2024)")
    print("  Paper-Accurate Python Translation")
    print("="*72)

    # 1. Normoxia
    print(f"\n[1] Normoxia (tf={a.tf} ms)...")
    t0 = time.time()
    pn = normoxia_params(); sn = run_sim(pn, a.tf)
    print(f"    {len(sn.t)} steps in {time.time()-t0:.1f}s")
    print(f"    PaO2 = {sn.y[6,-1]:.1f} mmHg, SaO2 = {compute_SaO2(sn.y[6,-1],pn)*100:.1f}%")

    # 2. Normoxia figures + data
    print("\n[2] Generating normoxia figures...")
    fig1_normoxia(sn, pn, out("fig1_normoxia.png"))
    fig_dissociation(out("fig_dissociation.png"))
    fig_sigmoid(out("fig_chemosensory.png"))
    export_csv(sn, pn, out("normoxia_timeseries.csv"))
    export_json(pn, out("normoxia_params.json"))

    if a.compare:
        # 3. SH simulation
        print(f"\n[3] Silent Hypoxemia (phi=0.24, theta_g=70, sigma_g=36, [Hb]=250)...")
        t0 = time.time()
        ps = sh_params(); ss = run_sim(ps, a.tf)
        print(f"    {len(ss.t)} steps in {time.time()-t0:.1f}s")
        print(f"    PaO2 = {ss.y[6,-1]:.1f} mmHg, SaO2 = {compute_SaO2(ss.y[6,-1],ps)*100:.1f}%")
        export_csv(ss, ps, out("sh_timeseries.csv"))
        export_json(ps, out("sh_params.json"))

        # 4. Comparison figures
        print("\n[4] Comparison figures...")
        fig6_comparison(sn, pn, ss, ps, out("fig6_comparison.png"))
        norms = fig_l2(sn, ss, out("fig_l2_analysis.png"))

        # 5. L2 summary
        print("\n" + "="*50)
        print("  L2 Norms: Normoxia vs Silent Hypoxemia")
        print("="*50)
        for k, v in norms.items():
            print(f"  {k:>8s}:  {v:.6f}")
        print("="*50)
    else:
        print("\n  (Use --compare for SH comparison)")

    print(f"\nAll outputs: {os.path.abspath(a.output_dir)}/")
    print("Done.\n")

if __name__ == '__main__':
    main()
