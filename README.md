This is an unofficial implementation of "Fast RobustSTL: Efficient and Robust Seasonal-Trend Decomposition for Time Series with Complex Patterns".
This approach supports the decomposition of multiple seasonalities.
Note that the gradient descent-based algorithm (Adam) is used rather than GADMM to decompose multiple seasonalities due to incomplete information in the original paper.


## Installation
`pip install --upgrade git+https://github.com/ariaghora/fast-robust-stl.git`


## Usage
```python
from frstl import fast_robustSTL

# season lengths -> len = m
season_lens = [24, 168, 672]

# trend extraction regularization factors -> len = 2
trend_regs = [10.0, 10.0]

# season regularization factors -> len = 3 * m
season_regs = [[0.001, 1, 10],    # lambda 1, 2, 3 for seasonality  1
               [0.0001, 100, 1],  # lambda 1, 2, 3 for seasonality  2
               [500, 0.01, 10]]   # lambda 1, 2, 3 for seasonality  3

# weighting factor for each non-local seasonal filter (Eq. 10) -> len = m
alphas = [1., 1., 1.]

# normalization factor (Eq. 10)
z = 3

# denoising weights -> len = 2
denoise_ds = [1.0, 1.0]

# season decomposition weights -> len = 2
season_ds = [50.0, 1.0]

K = 3
H = 5

input_ori, trends_hat, multiple_seas, remainders_hat =\
    fast_robustSTL(input, season_lens, trend_regs, season_regs, alphas, z, denoise_ds, season_ds, K, H)
```

For a pre-run colab notebook plese click [this link](https://colab.research.google.com/drive/17Mddx2PuqpkyPLDQmbQMGYm-zc9fGFR-?usp=sharing).
