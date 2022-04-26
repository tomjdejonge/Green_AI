import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import FactorAnalysis, PCA
import tensorly as tl
from sklearn.preprocessing import normalize
from tensorly import unfold as tl_unfold
from tensorly.decomposition import parafac

from utils import *

time_factor = np.load("C:/Users/tommo/Downloads/BEP_GREEN_AI/time_factor.npy")
neuron_factor = np.load("C:/Users/tommo/Downloads/BEP_GREEN_AI/neuron_factor.npy")
trial_factor = np.load("C:/Users/tommo/Downloads/BEP_GREEN_AI/trial_factor.npy")
latent = np.load("C:/Users/tommo/Downloads/BEP_GREEN_AI/latent.npy")
observed = np.load("C:/Users/tommo/Downloads/BEP_GREEN_AI/observed.npy")

factors_actual = (normalize(time_factor), normalize(neuron_factor), normalize(trial_factor))

# Specify the tensor and the rank
X, rank = observed, 3

# Perform CP decompositon using TensorLy
factors_tl = parafac(X, rank=rank)

# Reconstruct M, with the result of each library
M_tl = reconstruct(factors_tl)

# Compute the reconstruction error
rec_error_tl = np.mean((X-M_tl)**2)

# plot the decomposed factors from TensorLy
plot_factors(factors_tl, d=3)
plt.suptitle("Factors computed with TensorLy", y=1.1, fontsize=20);



