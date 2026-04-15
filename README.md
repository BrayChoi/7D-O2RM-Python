# 7D-O₂ Closed-Loop Respiratory Control Model

**COVID-19 and Silent Hypoxemia — Interactive Research Tool**

MATLAB → Python translation of the 7D-O₂ model from Diekman, Thomas & Wilson (2024), with an interactive browser-based simulator, L₂ norm cross-validation, and publication-quality figure generation.

## Paper Reference

Diekman CO, Thomas PJ, Wilson CG (2024). COVID-19 and silent hypoxemia in a minimal closed-loop model of the respiratory rhythm generator. *Biological Cybernetics*, 118(3-4):145–163. DOI: [10.1007/s00422-024-00989-w](https://doi.org/10.1007/s00422-024-00989-w)

- Original MATLAB code: [ModelDB #229640](https://modeldb.science/229640)
- Silent hypoxemia extension: [ModelDB #2015954](https://modeldb.science/2015954)

## Quick Start

### Web Application (React + Vite)

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Python Model (standalone)

```bash
# Normoxia only
python python/respiratory_model.py

# Normoxia vs Silent Hypoxemia comparison with L₂ norms
python python/respiratory_model.py --compare

# All outputs
python python/respiratory_model.py --compare --output-dir ./figures
```

**Dependencies:** numpy, scipy, matplotlib

## Deploy to Vercel

### Option A: GitHub Integration (recommended)
1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → Import Project → Select your repo
3. Vercel auto-detects Vite — click Deploy

### Option B: CLI
```bash
npm install -g vercel
npm run build
vercel
```

## Project Structure

```
├── index.html              # Entry point
├── package.json            # Dependencies
├── vite.config.js          # Vite configuration
├── vercel.json             # Vercel deployment config
├── public/
│   └── favicon.svg
├── src/
│   ├── main.jsx            # React entry
│   └── App.jsx             # Full application (all pages)
└── python/
    └── respiratory_model.py  # Paper-accurate Python implementation
```

## Model Parameters

### Normoxia (Original 7D-O₂)
| Parameter | Value | Equation |
|-----------|-------|----------|
| φ (phi) | 0.3 nS | Eq. 28 |
| θg | 85 mmHg | Eq. 28 |
| σg | 30 mmHg | Eq. 28 |
| [Hb] | 150 g/L | Eq. 27 |

### Silent Hypoxemia (SH Working Model)
| Parameter | Value | Source |
|-----------|-------|--------|
| φ (phi) | 0.24 nS | Results p.7 |
| θg | 70 mmHg | Results p.7 |
| σg | 36 mmHg | Results p.7 |
| [Hb] | 250 g/L | Fig. 5B, p.9 |

## Acknowledgments

Based entirely on the work of Casey O. Diekman (NJIT), Peter J. Thomas (Case Western Reserve), and Christopher G. Wilson (Loma Linda University). Original research supported by NIH and NSF grants.
