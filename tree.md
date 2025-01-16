```
# Project Structure

├── .git
│   ├── FETCH_HEAD
│   ├── HEAD
│   ├── config
│   ├── description
│   ├── hooks
│   │   ├── applypatch-msg.sample
│   │   ├── commit-msg.sample
│   │   ├── fsmonitor-watchman.sample
│   │   ├── post-update.sample
│   │   ├── pre-applypatch.sample
│   │   ├── pre-commit.sample
│   │   ├── pre-merge-commit.sample
│   │   ├── pre-push.sample
│   │   ├── pre-rebase.sample
│   │   ├── pre-receive.sample
│   │   ├── prepare-commit-msg.sample
│   │   ├── push-to-checkout.sample
│   │   ├── sendemail-validate.sample
│   │   └── update.sample
│   ├── info
│   │   └── exclude
│   ├── objects
│   │   ├── info
│   │   └── pack
│   └── refs
│       ├── heads
│       └── tags
├── .gitignore
├── .lprof
├── .vscode
│   └── settings.json
├── Long term goals.md
├── Pol_Hom.afdesign
├── Pol_Hom.afdesign~lock~
├── README.md
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
│   ├── Fig_1.ipynb
│   ├── Fig_2.ipynb
│   ├── Fig_3.ipynb
│   └── plot_functions.py
├── print_tree.ipynb
├── setup.py
├── src
│   ├── __init__.py
│   ├── models
│   │   ├── SIRM
│   │   │   ├── __init__.py
│   │   │   ├── dynamic.py
│   │   │   └── sweep.py
│   │   ├── SIRM_D
│   │   │   ├── __init__.py
│   │   │   ├── dynamic.py
│   │   │   └── sweep.py
│   │   ├── SIRT
│   │   │   ├── __init__.py
│   │   │   ├── dynamic.py
│   │   │   └── sweep.py
│   │   ├── SIRV
│   │   │   ├── __init__.py
│   │   │   ├── dynamic.py
│   │   │   └── sweep.py
│   │   └── __init__.py
│   └── utils
│       ├── Contact_Matrix.py
│       ├── R0.py
│       ├── __init__.py
│       └── batch_sweep.py
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