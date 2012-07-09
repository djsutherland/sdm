#!/usr/bin/env python
from __future__ import print_function

import sdm
import numpy as np
import progressbar as pbar

neg = [np.random.normal(0, 1, (12, 4)) for _ in range(100)]
pos = [np.random.normal(1, .2, (12, 4)) for _ in range(100)]
bags = neg + pos
labels = [0] * len(neg) + [1] * len(pos)

widgets = ['Progress: ', pbar.Percentage(), ' ',
        pbar.Bar(marker=pbar.RotatingMarker()),
        ' ', pbar.ETA()]
def pbar_wrapper():
    pb = []
    def inner(x):
        if not pb:
            pb.append(pbar.ProgressBar(widgets=widgets, maxval=x).start())
            pb.append(x)
        else:
            pb[0].update(pb[1] - x)
        if x == 0:
            pb[0].finish()
    return inner

dfs = ['renyi:.8', 'hellinger']
divs = sdm.get_divs(bags, div_funcs=dfs,
        num_threads=0, cv_threads=0,
        show_progress=50,
        print_progress=pbar_wrapper(),
)

for df, df_divs in zip(dfs, divs):
    acc = sdm.crossvalidate_divs(df_divs, labels)
    print("{} accuracy: {:.0%}".format(df, acc))
