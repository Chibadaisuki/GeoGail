import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

zeta = 0.6
alpha = -1.3
bin_num = 1000

br_list = np.load('./preprocess_data/geolife/burst_rank_list.npy')
log_10_br_list = np.log10(br_list)
freqs, bins = np.histogram(log_10_br_list, bins=bin_num)
bins_median = (10 ** bins[: -1] + 10 ** bins[1:]) / 2
freqs = freqs / (10 ** bins[: -1] - 10 ** bins[1:])
freqs = freqs / np.sum(freqs)

p_k = zeta * bins_median ** (alpha)
#plt.close()
plt.xscale("log")
plt.yscale("log")
plt.plot(bins_median, freqs, 'o')
plt.plot(bins_median, p_k)
plt.legend(['real data', 'fit curve alpha={}'.format(alpha)])
#plt.close()
plt.savefig('./alpha_fit_result.jpg')
#plt.close()
