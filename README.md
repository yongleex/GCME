# Gamma Correction with Maximum Entropy (GCME) 

[![preprint](https://img.shields.io/static/v1?label=Journal&message=Signal_Processing&color=B31B1B)](https://doi.org/10.1016/j.sigpro.2021.108427)
[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2007.02246&color=B31B1B)](http://arxiv.org/abs/2007.02246)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository contains code for the accepted paper *[Blind Inverse Gamma Correction with Maximized Differential Entropy](https://doi.org/10.1016/j.sigpro.2021.108427)*. 
The best-restored image 
<img src="https://render.githubusercontent.com/render/math?math=I^\gamma">
 is assumed with the largest entropy value. Our GCME method obtains a closed-form solution via differential entropy and change-of-variables rule. As a result, our GCME is an exact (non-approximate), accurate, and fast gamma correction algorithm.


![results](https://github.com/yongleex/GCME/blob/5d6a2be83f622543f7e04b9c4b15527086bff3d4/images/results.png)

### Motivation
Maximum entropy has been proved to be an effective image prior because most good images should contain sufficient information. 
Our motivation is very simple, the best gamma should maximize the entropy of transformed image.

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\gamma^* = \arg\max Entropy(I^\gamma)">
</p>

Fortunately, we found a closed-form solution to this optimization problem.

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\gamma^* = -\frac{1}{\int_0^1 p_I(u)\ln(u)du}">
</p>

which can be efficiently computed with one line Python script for an input image $I$ (pixel intensity range $(0,1)$)
```python
gamma = -1/np.nanmean(np.log(I))
```
More info is referred to the [paper](https://doi.org/10.1016/j.sigpro.2021.108427).

### Install dependencies
```
conda install numpy matplotlib opencv seaborn
conda install -c conda-forge glob2
# conda install -c conda-forge nibabel   # read MRI images		
conda install -c conda-forge pydicom   # read CT images
conda install pip
pip install nibabel
```

### BibTeX

```
@article{Lee2021GCME,
		title = {Blind Inverse Gamma Correction with Maximized Differential Entropy},
		author = {Yong Lee and Shaohua Zhang and Miao Li and Xiaoyu He}
		journal = {Signal Processing},
		pages = {108427},
		year = {2021},
		issn = {0165-1684},
		doi = {https://doi.org/10.1016/j.sigpro.2021.108427},
		url = {https://www.sciencedirect.com/science/article/pii/S0165168421004643},
}
```

## Contact
Yong Lee (Email: yongli.cv@gmail.com) @2021-12-08

