import numpy as np
import torch
from .utils import *
from .l1 import l1
from cvxopt import matrix
from tqdm.auto import tqdm


def denoise_step(sample, H=3, dn1=1., dn2=1.):
    def get_denoise_value(idx):
        start_idx, end_idx = get_neighbor_idx(len(sample), idx, H)
        idxs = np.arange(start_idx, end_idx)
        weight_sample = sample[idxs]

        weights = np.array(list(map(lambda j: bilateral_filter(
            j, idx, sample[j], sample[idx], dn1, dn2), idxs)))
        return np.sum(weight_sample * weights)/np.sum(weights)

    idx_list = np.arange(len(sample))
    denoise_sample = np.array(list(map(get_denoise_value, idx_list)))
    return denoise_sample


def seasonality_extraction(sample, season_lens, alphas, z, K=2, H=5, ds1=50., ds2=1.):
    sample_len = len(sample)
    idx_list = np.arange(sample_len)

    def get_season_value(idx, season_len):
        idxs = get_season_idx(sample_len, idx, season_len, K, H)
        if idxs.size == 0:
            return sample[idx]

        weight_sample = sample[idxs]
        weights = np.array(list(map(lambda j: bilateral_filter(
            j, idx, sample[j], sample[idx], ds1, ds2), idxs)))
        season_value = np.sum(weight_sample * weights)/np.sum(weights)
        return season_value

    seasons_tilda = 0.
    for season_len, alpha in zip(season_lens, alphas):
        seasons_tilda += alpha * np.array(
            [get_season_value(idx, season_len) for idx in idx_list])
    seasons_tilda /= z

    return seasons_tilda


def trend_extraction(sample, season_len, reg1=10., reg2=0.5):
    sample_len = len(sample)
    season_diff = sample[season_len:] - sample[:-season_len]
    g = season_diff[:, None]

    assert len(season_diff) == (sample_len - season_len)
    q = np.concatenate([season_diff, np.zeros([sample_len*2-3])])
    q = np.reshape(q, [len(q), 1])

    M = get_toeplitz([sample_len-season_len, sample_len-1],
                     np.ones([season_len]))
    D = get_toeplitz([sample_len-2, sample_len-1], np.array([1, -1]))
    P = np.concatenate([M, reg1*np.eye(sample_len-1), reg2*D], axis=0)

    '''
    l1 approximation solver
    '''
    q = matrix(q)
    P = matrix(P)
    delta_trends = l1(P, q)

    relative_trends = get_relative_trends(delta_trends)

    return sample-relative_trends, relative_trends


def adjustment(sample, relative_trends, seasons_tilda, season_len):
    num_season = int(len(sample)/season_len)

    trend_init = np.mean(seasons_tilda[:season_len*num_season])

    trends_hat = relative_trends + trend_init
    seasons_hat = seasons_tilda - trend_init
    remainders_hat = sample - trends_hat - seasons_hat
    return [trends_hat, seasons_hat, remainders_hat]


def check_converge_criteria(prev_remainders, remainders):
    diff = np.sqrt(np.mean(np.square(remainders-prev_remainders)))
    if diff < 1e-6:
        return True
    else:
        return False


def decompose_multiple_seasonal_components(seasons_hat, season_lens, season_regs, max_iter):
    m = len(season_lens)
    N = len(seasons_hat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seasons_predict = torch.empty(N, m).double().to(device)
    seasons_predict.requires_grad = True
    torch.nn.init.xavier_uniform_(seasons_predict)

    seasons_hat = torch.from_numpy(seasons_hat).to(device)

    D = get_toeplitz([N-2, N-1], np.array([1, -1]))
    D = torch.from_numpy(D).double().to(device)

    D2 = get_toeplitz([N-2, N], np.array([1, -2, 1]))
    D2 = torch.from_numpy(D2).double().to(device)

    DTis = [get_toeplitz([N-2*slen, N], np.array(
        [1, *([0]*(slen-1)), -2, *([0]*(slen-1)), 1])) for slen in season_lens]
    DTis = [torch.from_numpy(DTi).double().to(device) for DTi in DTis]

    season_regs = torch.from_numpy(np.array(season_regs)).double().to(device)

    sse = torch.nn.MSELoss(reduction='sum').to(device)  # **SUM** squared error
    opt = torch.optim.Adam([seasons_predict], lr=0.001)

    print('Extracting multiple seasonalities...')
    pbar = tqdm(range(max_iter))
    for j in pbar:
        opt.zero_grad()

        # following terms are those in Eq. 17
        term1 = sse(seasons_hat, seasons_predict.sum(1))
        term2 = (torch.norm(
            D @ seasons_predict[1:], p=1, dim=0, keepdim=True) * season_regs[:, 0]).sum()
        term3 = (torch.norm(D2 @ seasons_predict, p=1, dim=0,
                 keepdim=True) * season_regs[:, 1]).sum()
        term4 = 0

        for k in range(m):
            term4 += torch.norm(DTis[k] @ seasons_predict[:,
                                k], p=1) * season_regs[k, 2]

        loss = term1 + term2 + term3 + term4
        loss.backward()
        opt.step()

        pbar.set_description(f'Loss: {loss.item()}')

        # if j % 1000 == 0:
        #     print(f'loss at {j}/{max_iter-1}: {loss.item()}')

    res = seasons_predict.detach().cpu().numpy()

    return res


def quick_viz(plots, labels):
    assert len(plots) == len(labels), 'length of plots != length of labels'
    n = len(plots)

    for i, (plot, label) in enumerate(zip(plots, labels)):
        plt.subplot(n, 1, i+1)
        plt.plot(plot, label=label)
        plt.legend()
    plt.show()


def fast_robustSTL(input, season_lens, trend_regs, season_regs, alphas, z, denoise_ds, season_ds, K=2, H=5, max_iter=5000):
    sample = input
    trial = 1
    while True:
        '''
        Step 1: Denoising using bilateral filter
        '''
        dn1, dn2 = denoise_ds
        denoised_sample = denoise_step(sample, H, dn1, dn2)

        '''
        Step 2: Trend extraction by Robust Sparse Model
        '''
        reg1, reg2 = trend_regs
        season_len = np.max(season_lens)
        detrend_sample, relative_trends =\
            trend_extraction(denoised_sample, season_len, reg1, reg2)

        ''' 
        Step 3: Seasonal component extraction by non-local seasonal filter
        '''
        ds1, ds2 = season_ds
        seasons_tilda =\
            seasonality_extraction(
                detrend_sample, season_lens, alphas, z, K, H, ds1, ds2)

        trends_hat, seasons_hat, remainders_hat =\
            adjustment(sample, relative_trends, seasons_tilda, season_len)

        if trial != 1:
            converge = check_converge_criteria(
                previous_remainders, remainders_hat)
            if converge:
                break

        trial += 1
        previous_remainders = remainders_hat[:]
        sample = trends_hat + seasons_hat + remainders_hat

    '''
    Step 5:
    Further decomposition of multiple seasonal components.
    This is executed after the multiple seasons hat is discovered.
    Need suggestion.
    '''

    if len(season_lens) == 1:
        multiple_seas = seasons_hat
    else:
        multiple_seas =\
            decompose_multiple_seasonal_components(
                seasons_hat, season_lens, season_regs, max_iter)

    return [input, trends_hat, multiple_seas, remainders_hat]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    N = 1500
    y1 = sinewave(N, 24, 1)
    y2 = sinewave(N, 168, 1.5)
    y3 = sinewave(N, 672, 2)
    y = y1+y2+y3+np.random.normal(0, 0.2, N)

    y[672:] += 10  # simulate abrupt change from the middle of sample onwards

    # season lengths -> len = m
    season_lens = [24, 168, 672]
    # trend extraction regularization factors -> len = 2
    trend_regs = [1.0, 10.0]
    # season regularization factors -> len = 3 * m
    season_regs = [[0.001, 1, 10],    # reg 1, 2, 3 for seasonality  1
                   [0.0001, 100, 1],   # reg 1, 2, 3 for seasonality  2
                   [500, 0.01, 10]]  # reg 1, 2, 3 for seasonality  3

    alphas = [0.01, 0.01, 1]
    # normalization factor (Eq. 10)
    z = 3
    # denoising weights -> len = 2
    denoise_ds = [1.0, 1.0]
    # season decomposition weights -> len = 2
    season_ds = [50.0, 1.0]
    K = 3
    H = 5

    result = fast_robustSTL(y,
                            season_lens=season_lens,
                            trend_regs=trend_regs,
                            season_regs=season_regs,
                            alphas=alphas,
                            z=z,
                            denoise_ds=denoise_ds,
                            season_ds=season_ds,
                            K=K,
                            H=H)

    input_ori, trends_hat, multiple_seas, remainders_hat = result

    quick_viz([input_ori, trends_hat], ['input', 'trend'])
    quick_viz([y1, y2, y3], ['sinewave 1', 'sinewave 2', 'sinewave 3'])
    quick_viz([*multiple_seas.T, remainders_hat],
              [f'seasonality {i+1}' for i in range(multiple_seas.shape[1])] + ['remainders'])
