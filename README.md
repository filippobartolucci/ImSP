# Image Specific Protection Against Manipulation
Official implementation of the ICIAP 2025 paper ["Image Specific Protection Against Manipulation"](https://link.springer.com/chapter/10.1007/978-3-032-10185-3_52)

# Overview
<p align="center">
 <img src="./architecture.jpg" alt="teaser" width="90%" />
</p>

This work introduces a proactive defense framework to safeguard images against manipulations performed by Generative Models (GMs). Unlike traditional passive detection methods, which attempt to identify manipulations after they occur, our approach embeds image-specific protection signals directly into the image before sharing it.

# Trained models
You can download the pretrained checkpoints from the link below:
 * [Checkpoints](https://drive.google.com/file/d/1Xs_0sDi7BGUogeuZFilhtJNx5RrFgM0Y/view?usp=share_link)

# Dataset
We rely on the test set split used in MaLP.
Please download the dataset directly from their repository:
* [MaLP Repository](https://github.com/vishal3477/pro_loc)

# Authors
* [Filippo Bartolucci](https://github.com/filippobartolucci)
* Giuseppe Lisanti

# Cite
If you use this source code please cite the following works:

**Image Specific Protection Against Manipulation**
```
@InProceedings{ImSP,
author="Bartolucci, Filippo
and Lisanti, Giuseppe",
editor="Rodol{\`a}, Emanuele
and Galasso, Fabio
and Masi, Iacopo",
title="Image Specific Protection Against Manipulation",
booktitle="Image Analysis and Processing -- ICIAP 2025",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="660--671",
}
```

**[Perturb, Attend, Detect and Localize (PADL): Robust Proactive Image Defense](https://github.com/filippobartolucci/PADL)**
```
@ARTICLE{10980274,
  author={Bartolucci, Filippo and Masi, Iacopo and Lisanti, Giuseppe},
  journal={IEEE Access}, 
  title={Perturb, Attend, Detect, and Localize (PADL): Robust Proactive Image Defense}, 
  year={2025},
  volume={13},
  pages={81755-81768},
  doi={10.1109/ACCESS.2025.3565824}}

```

