#%%
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Random state.
RS = 20150101

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
digits = load_digits()
#%%
# We first reorder the data points according to the handwritten numbers.
X = np.vstack([digits.data[digits.target==i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])
digits_proj = TSNE(random_state=RS).fit_transform(X)
#%%
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", np.unique(colors).shape[0]))

    # We create a scatter plot.
    f = plt.figure(figsize=(20, 20))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=4,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    # for i in range(np.unique(colors).shape[0]):
        # Position of each label.
        # xtext, ytext = np.median(x[colors == i, :], axis=0)
        # txt = ax.text(xtext, ytext, str(i), fontsize=24)
        # txt.set_path_effects([
        #     PathEffects.Stroke(linewidth=5, foreground="w"),
        #     PathEffects.Normal()])
        # txts.append(txt)

    return f, ax, sc, txts
#%%
scatter(digits_proj, y)
# plt.savefig('digits_tsne-generated.png', dpi=120)
plt.show()


# %%
l_feas = np.load('logs/tmp/l_feas.npy')
u_feas = np.load('logs/tmp/u_feas.npy')
y_l = np.zeros(l_feas.shape[0],dtype='int8')
y_u = np.ones(u_feas.shape[0],dtype='int8')
feas_all = np.vstack([l_feas,u_feas])
y_all = np.hstack([y_l,y_u])
# %%
proj = TSNE(random_state=RS).fit_transform(feas_all)

# %%
scatter(proj, y_all)
plt.savefig('logs/tmp/u_l_feas_tsne-generated.jpg', dpi=300,facecolor='white')
plt.show()

# %%
