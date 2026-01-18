
# Thomas Le Sellier De Chezelles — Notes, Code & Papers

[![Made with PythonFramework
!Build
[![LicenseMIT

> A personal monorepo with:
> - **AI** — my MDL/Occam‑inspired training objective (natural logs), Sine‑MLP experiments, and other ML work.
> - **School** — CentraleSupélec coursework, projects, and utilities.
> - **Maths** — problem sets, notes, and solutions (incl. RMS — *Revue Mathématiques Spéciales* corrections).

---

## ✨ Highlights

- **Paper (LaTeX)**: *Minimum‑Description‑Length Training with Natural‑Log Curvature Penalty — An Engineering Spec*  
  (ReLU by default, Sine for smooth fields; exact loss with `ln(det(I+H))`, batch‑surrogate with `ln(1+2||ĝ||²)`.)
- **Code (PyTorch)**: Sine‑MLP regression — compare **baseline MSE** vs **MDL batch surrogate** on a noisy `sin + sin` target.
- **Maths**: curated exercises/solutions and notes; currently includes RMS corrections.

---

## 🗂️ Repository Structure

```text
.
├─ ai/
│  ├─ papers/
│  │  └─ mdl-natural-log/         # LaTeX sources for the MDL/Occam paper
│  │     ├─ main.tex              # main entry point (Sections 0–11)
│  │     ├─ figs/                 # exported figures (optional)
│  │     └─ makefile              # optional: latexmk targets (see below)
│  ├─ experiments/
│  │  └─ train_mdl_vs_mse.py      # Sine-MLP: MDL batch loss vs MSE (AdamW)
│  └─ utils/                      # helpers (optional)
│
├─ school/
│  ├─ courses/                    # course notes, projects, reports
│  └─ tools/                      # scripts, small utilities for coursework
│
├─ maths/
│  ├─ rms/                        # RMS (Revue Mathématiques Spéciales) corrections
│  └─ notes/                      # personal math notes, problem sets
│
├─ .gitignore
├─ LICENSE                        # MIT (or your choice)
└─ README.md
