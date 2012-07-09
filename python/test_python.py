#!/usr/bin/env python
from __future__ import print_function

import sdm
import numpy as np
import progressbar as pbar
import random

neg = [np.random.normal(0, 1, (12, 4)) for _ in range(50)]
pos = [np.random.normal(1, .2, (12, 4)) for _ in range(50)]
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

print()
print("Training model")
sdm.set_log_level('debug')
model = sdm.train(bags, labels)
print("trained!", model)

test_neg = [np.random.normal(0, 1, (12, 4)) for _ in range(5)]
test_pos = [np.random.normal(1, .2, (12, 4)) for _ in range(5)]

print("Testing a negative:", model.predict(test_neg[0]))
print("Testing a positive:", *(model.predict(test_pos[0], get_dec_values=True)))
print("Testing a bunch:", model.predict(test_neg + test_pos))
print("\nTesting a bunch with dec values:")
preds, vals = model.predict(test_neg + test_pos, get_dec_values=True)
print(preds)
print(vals)

#model.free()
#print("freed!")
