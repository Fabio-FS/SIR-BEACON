```
# Project Structure


├── .gitignore
├── .lprof
├── .vscode
│   └── settings.json
├── Long term goals.md
├── Pol_Hom.afdesign
├── Pol_Hom_backup.afdesign
├── README.md
├── README_for_Model.md
├── README_for_visualization.md
├── Summary of the results.md
├── figures
│   ├── Fig_1.jpg
│   ├── no_labels
│   │   ├── Fig_1
│   │   ├── Fig_2
│   │   ├── Fig_3
│   │   └── unsure or SI
│   └── with_labels
├── notebooks
│   ├── Fig_0_test.ipynb
│   ├── Fig_1_SIRM_old.ipynb
│   ├── Fig_Intro.ipynb
│   ├── SIR-mask.ipynb
│   ├── SIRM.ipynb
│   ├── SIRT.ipynb
│   ├── SIRV.ipynb
│   ├── Sophias_data.ipynb
│   ├── data_homophily.csv
│   └── plot_functions.py
├── print_tree.ipynb
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── models
│   │   ├── SIRM
│   │   │   ├── __init__.py
│   │   │   └── integrated.py
│   │   ├── SIRM_D
│   │   │   ├── __init__.py
│   │   │   ├── dynamic.py
│   │   │   ├── integrated.py
│   │   │   └── sweep.py
│   │   ├── SIRT
│   │   │   ├── __init__.py
│   │   │   └── integrated.py
│   │   ├── SIRT_D
│   │   │   ├── __init__.py
│   │   │   ├── dynamic.py
│   │   │   ├── integrated.py
│   │   │   └── sweep.py
│   │   ├── SIRV
│   │   │   ├── __init__.py
│   │   │   ├── dynamic.py
│   │   │   ├── integrated.py
│   │   │   └── sweep.py
│   │   ├── SIRV_D
│   │   │   ├── __init__.py
│   │   │   ├── dynamic.py
│   │   │   ├── integrated.py
│   │   │   └── sweep.py
│   │   ├── __init__.py
│   │   ├── consolidated_dynamics.py
│   │   ├── consolidated_sweeps.py
│   │   ├── mask_SIR
│   │   │   ├── __init__.py
│   │   │   └── integrated.py
│   │   └── mask_SIR_D
│   │       ├── __init__.py
│   │       ├── dynamic.py
│   │       ├── integrated.py
│   │       └── sweep.py
│   └── utils
│       ├── Contact_Matrix.py
│       ├── R0.py
│       ├── __init__.py
│       ├── batch_sweep.py
│       └── consolidated_batch_sweep.py
├── tests
│   ├── 001_test_imports.ipynb
│   ├── 002_test_SIR.ipynb
│   ├── 003_test_SIRB.ipynb
│   ├── 004_test_contacts.ipynb
│   ├── 005_test_contacts.ipynb
│   ├── 006_test_sweep.ipynb
│   ├── 007_compare_sweep.ipynb
│   ├── 008_Sinkhorn_normalization.ipynb
│   ├── 009_Is_orientation_h_correct.ipynb
│   ├── Debugging.ipynb
│   ├── __init__.py
│   ├── test_optimized_sweep.py
│   └── test_sweep.py
└── tree.md
```