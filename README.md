# Manuscript Plot Reproduction: Figures 1, 2, 3, and 4

This repository contains scripts for reproducing key statistical analyses and plots from the manuscript. Specifically, it includes:

- **Figure 1**: Performance in the object location memory and object location arrangement task.
- **Figure 2**: Map-like representations in the right hippocampus (rHC) and their correlation with general fluid intelligence (gf).
- **Figure 3**: 2D-ness of distance representations in the rHC and behavioural distance estimates.
- **Figure 4**: Item familiarity signals in the rHC.

---

## 📁 Project Structure

project-root/
│
├── data/
│ └── data.csv # Raw dataset used for analysis
│
├── plots/ # Output directory for all figures
│ └── ...
│
├── figure_1.py # Script for Figure 1
├── figure_2.py # Script for Figure 2
├── figure_3.py # Script for Figure 3 
├── figure_4.py # Script for Figure 4
│
├── README.md
└── requirements.txt

---

## ✅ Requirements

Use the included `requirements.txt` file to install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Scripts

All paths are relative to ~/owncloud/projects/publication. Update if necessary.

Ensure your working directory is set to the project root. Then execute any script, for example:

```bash
python figure_1.py
```

Figures will be saved in the plots/ folder.


