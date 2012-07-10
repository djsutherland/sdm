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

df_kernels = [('renyi:.8', 'gaussian'), ('linear', 'polynomial')]
divs = sdm.get_divs(bags, div_funcs=[df for df, kernel in df_kernels],
        num_threads=0, cv_threads=0,
        show_progress=50,
        print_progress=pbar_wrapper(),
)

for (df, kernel), df_divs in zip(df_kernels, divs):
    acc = sdm.crossvalidate_divs(df_divs, labels, kernel=kernel)
    print("{}/{} accuracy: {:.0%}".format(df, kernel, acc))

print()
print("Training model")
sdm.set_log_level('info')
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

print()
print("Transducting with linear...")
preds = sdm.transduct(bags, labels, test_neg + test_pos,
                      div_func='linear', kernel='linear')
print(preds)

print()
trans_divs = sdm.get_divs(bags + test_neg + test_pos, div_funcs=['renyi:1.01'])
td = trans_divs.copy()
print("Transducting with precomputed renyi:1.01...")
preds = sdm.transduct(bags, labels, test_neg + test_pos,
        div_func='renyi:1.01', divs=trans_divs[0])
print(preds)
assert np.all(trans_divs[0] == td)

# TODO: test passing divs in

#model.free()
#print("freed!")
