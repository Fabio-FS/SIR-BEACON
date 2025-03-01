```
# Project Structure

├── .git
│   ├── COMMIT_EDITMSG
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
│   ├── index
│   ├── info
│   │   └── exclude
│   ├── logs
│   │   ├── HEAD
│   │   └── refs
│   │       ├── heads
│   │       │   └── main
│   │       └── remotes
│   │           └── origin
│   │               └── main
│   ├── objects
│   │   ├── 07
│   │   │   ├── 755bfd225196a484904ba88340c8e712ca935f
│   │   │   └── 77fc1f1b7dd0b421db2e8b8c1fe232290ed2c8
│   │   ├── 09
│   │   │   └── bb1cfea8ecc2fcc4365ea7354ff6f195f310e8
│   │   ├── 0d
│   │   │   └── 9865b692b90262f8692e3f4af1413e650c2152
│   │   ├── 0e
│   │   │   └── e2ea495e5cf0249c773672ed2d82054068bf41
│   │   ├── 16
│   │   │   ├── 43b10c72701d6369bd99517c4d5c70c9caa03f
│   │   │   ├── b7023b86f5d8763f080ac6a83741680a637e96
│   │   │   └── fa98d7aebc98bb2e78c252ef25f4e422c356d1
│   │   ├── 1b
│   │   │   └── 5d22207057a25ee8527587519a59abae41b4c6
│   │   ├── 1e
│   │   │   └── 482284a59dda7004b4ea82dda3195603903ea9
│   │   ├── 20
│   │   │   └── 4bc21acddb7399daa1bfd34f15684f534b0b8f
│   │   ├── 27
│   │   │   └── c6c3c9285a8eac0ffdee6d8b8902c77dee4f6e
│   │   ├── 31
│   │   │   └── 8c017c4987f78c4bd270c96957e3b1cf02c3bb
│   │   ├── 33
│   │   │   └── c7d90176a1f0a9a73fde5b251ac08e10c4a3f4
│   │   ├── 34
│   │   │   └── 07a1c797aa70fd3b4926f8a63a408059df32a9
│   │   ├── 37
│   │   │   └── a80f864cee25bb0dc54f1ba573f22c89add799
│   │   ├── 40
│   │   │   └── 73ec9060975ba4fc68fad2abbc4e47118d95bf
│   │   ├── 43
│   │   │   └── c4557a98ce933931a71819bf98e0f300574540
│   │   ├── 4a
│   │   │   └── 93754bb4bf64313c6521d797b7422fa48f241b
│   │   ├── 55
│   │   │   └── 5944fb6326ded938a59066788f345cce472e6d
│   │   ├── 57
│   │   │   └── 22eea6cc24b1d2cd1d9fe25f4c4871255dc4b1
│   │   ├── 59
│   │   │   └── 006c05574cd14d9cc00d55db71d548942b9cfe
│   │   ├── 5a
│   │   │   ├── 8ce2f9dc8d01b096a16d49ac67d5a36c567703
│   │   │   └── a39451a7acdf6895e7e16120e89856ebc889a3
│   │   ├── 6e
│   │   │   └── 613d0bf0aca6e00942d10ada455bcee45b6e1c
│   │   ├── 71
│   │   │   └── a348ed4366b55cc016b8460724577ebf4bf81e
│   │   ├── 74
│   │   │   └── 8f57eecb6375c65f17035fe97ad380638d0c44
│   │   ├── 7c
│   │   │   └── dba4637f6dd03324a60c7092190a32daa08441
│   │   ├── 80
│   │   │   └── 1e147c7cf95811fbcc36ce57c9074353ea5529
│   │   ├── 81
│   │   │   └── d859bc607c61340878ef5dc03b788d8f65f6ae
│   │   ├── 8f
│   │   │   └── b52d6b28f77c764db739eb6d430872a57ca28f
│   │   ├── 94
│   │   │   └── d80d784b79c27e3ac577b74f72687c3600637c
│   │   ├── a5
│   │   │   ├── 292afa85cac4796f20b7e846e769a5baf7a7a4
│   │   │   └── e948fef779ac9253cfb661f472ef318de4be8d
│   │   ├── a9
│   │   │   └── f6baa8f51c3d5747e3f89b4fe254f39f821c07
│   │   ├── b1
│   │   │   └── 09b493fa7eca2289a27ee8d0fc79225299d2c6
│   │   ├── b6
│   │   │   ├── 46b2239dfef12e724d124e5a9cec54a6fc7a66
│   │   │   └── 7ef9a371477be285fa8f3a19a61dc6d94a8dbb
│   │   ├── bb
│   │   │   └── 962b3e3d05c171820b29a3760fbd098b20d997
│   │   ├── c0
│   │   │   ├── 3fec11f845ffa8f311dd98921e10ced45a6530
│   │   │   └── 439b0f705fd704ee3779e8ba1e02d5a085ea98
│   │   ├── c6
│   │   │   └── d53497555b64c7f9ab80ab71fcd32d63e836e1
│   │   ├── c7
│   │   │   └── c8f3609d02f7bd9669b0338e33f586f5a6a4ee
│   │   ├── c8
│   │   │   └── 84782d537738516a9d54b50a292f91327f78fa
│   │   ├── cd
│   │   │   └── 1fd475fbe630f0919ab883cf393449932b7153
│   │   ├── d7
│   │   │   └── d97718fa55ad521aa880ed0ecc8eeca1c67a5c
│   │   ├── db
│   │   │   └── a508c23343af6e3406e4ea3f14c6ae1c41f89d
│   │   ├── dc
│   │   │   ├── 1a39600da84f1f1b6e2cbedd9d5e6e314ccfb9
│   │   │   └── f3ed2ccad8f460755d7136af2d5d17e513e3e2
│   │   ├── e1
│   │   │   ├── 544a9902c06f78db95de52521695a1b715bb32
│   │   │   └── b82ef4705243e91472fa380b71c7e89b8f6117
│   │   ├── e2
│   │   │   └── 631b2978b681f035260d3be7582e1346aa8656
│   │   ├── e3
│   │   │   └── 053e015cfa0528638aea4b8a678a30d33aa7c5
│   │   ├── e6
│   │   │   └── 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│   │   ├── e9
│   │   │   └── a3f2a6264f1a99ffd78d4e3be3d3bb04dc5e7b
│   │   ├── f3
│   │   │   └── c84aa6d8bb276b19321f39d9d94866fb8c8fdc
│   │   ├── f5
│   │   │   └── b34f5da02cef9a1e02cf0bf822f65aa83a0fe5
│   │   ├── info
│   │   └── pack
│   └── refs
│       ├── heads
│       │   └── main
│       ├── remotes
│       │   └── origin
│       │       └── main
│       └── tags
├── .gitignore
├── .lprof
├── .vscode
│   └── settings.json
├── Long term goals.md
├── Pol_Hom.afdesign
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
│   ├── Fig_1_SIRM_old.ipynb
│   ├── Fig_2.ipynb
│   ├── Fig_3.ipynb
│   ├── Fig_4.ipynb
│   ├── Fig_Intro.ipynb
│   └── plot_functions.py
├── print_tree.ipynb
├── requirements.txt
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
│   │   ├── __init__.py
│   │   ├── mask_SIR
│   │   │   ├── __init__.py
│   │   │   ├── dynamic.py
│   │   │   └── sweep.py
│   │   └── mask_SIR_D
│   │       ├── __init__.py
│   │       ├── dynamic.py
│   │       └── sweep.py
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
├── tree copy.md
└── tree.md
```