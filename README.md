This is an unofficial implementation of "Fast RobustSTL: Efficient and Robust Seasonal-Trend Decomposition for Time Series with Complex Patterns".
This approach supports the decomposition of multiple seasonalities.
Note that the gradient descent-based algorithm (Adam) is used rather than GADMM to decompose multiple seasonalities due to incomplete information in the original paper.


## Installation
`pip install --upgrade git+https://github.com/ariaghora/fast-robust-stl.git`


## Usage
```python
from frstl import fast_robustSTL

input_ori, trends_hat, multiple_seas, remainders_hat =\
    fast_robustSTL(input, season_lens, trend_regs, season_regs, alphas, z, denoise_ds, season_ds, K, H)
```