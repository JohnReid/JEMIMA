#!/usr/bin/env python

#
# Copyright John Reid 2014
#

import jemima as jem
import jemima.importancesampling as jis
from jemima import wmers

from itertools import imap
import pandas as pd
import pandas.rpy.common as com
import seqan.traverse
import numpy as npy
import numpy.random as rdm
from numpy import log, exp

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import rpy2.robjects.lib.ggplot2 as ggplot2
import rpy2.robjects as ro
from rpy2.robjects.vectors import IntVector, FloatVector, StrVector
from rpy2.robjects.packages import importr
grdevices = importr('grDevices')

lambda_ = .01
numsites = 50
pseudocount = 1.
runx1pwm = npy.array((
    (0.384615,  0.076923,  0.115385,  0.423077),
    (0.461538,  0.076923,  0.038462,  0.423077),
    (0.153846,  0.269231,  0.038462,  0.538462),
    (0.038462,  0.038462,  0.000000,  0.923077),
    (0.076923,  0.000000,  0.884615,  0.038462),
    (0.076923,  0.307692,  0.000000,  0.615385),
    (0.000000,  0.000000,  1.000000,  0.000000),
    (0.000000,  0.000000,  1.000000,  0.000000),
    (0.000000,  0.038462,  0.000000,  0.961538),
    (0.307692,  0.076923,  0.000000,  0.615385),
    (0.500000,  0.076923,  0.153846,  0.269231),
))
runx1withpc = jem.addpseudocounts(runx1pwm, numsites, pseudocount)
W = len(runx1withpc)

#logo(runx1pwm, 'runx1')
#logo(runx1withpc, 'runx1-pc')

logging.info('Loading sequences')
#seqs = seqan.StringDNASet(('AAAAAAAA', 'ACGTACGT', 'TATATATA'))
numbases, seqs, ids = seqan.readFastaDNA('T00759-small.fa')

logging.info('Building index')
index = seqan.IndexStringDNASetESA(seqs)

logging.info('Calculating true Zn sums')
calculateZn = jem.createZncalculatorFn(runx1withpc, lambda_)
summer = jis.ZnSumCb(W)
sumvisitor = jis.ZnCalcVisitor(W, calculateZn, summer)
seqan.traverse.topdownhistorytraversal(index.topdownhistory(), sumvisitor)
logging.info('Sums:\n%s', summer.sums)
trueZnsum = summer.sums[0].sum()
logging.info('True sum: %s', trueZnsum)
#logo(normalisepwm(sumvisitor.sums), 'learnt')

logging.info('Counting W-mers')
Ws = [W]
Wmercounts = npy.zeros((2*len(index), len(Ws)), dtype=npy.uint)
numWmers = wmers.countWmersMulti(index.topdownhistory(), Ws, Wmercounts)[0]
logging.info('Got %d %d-mers', numWmers, W)
childWmerfreqs = npy.zeros((2*len(index), len(Ws), jem.SIGMA))
childWmerfreqs = jem.normalisepwm(childWmerfreqs)
wmers.countWmerChildren(index.topdownhistory(), W, Wmercounts, childWmerfreqs)
sumestimator = jis.makesumestimator(numWmers)
1/0

numsamples = 3000
rdm.seed(1)
logging.info('Importance sampling using binding site model')
samplebs, cbbs = jis.importancesample(
    index, jem.createpwmlikelihoodfn(runx1withpc), W, childWmerfreqs[:,0], calculateZn, numsamples, model='BS')
#logging.info('Importance sampling using binding site model squared')
#samplebs2, cbbs2 = importancesample(
#    createpwmlikelihoodsquaredfn(runx1withpc), W, childWmerfreqs[:,0], calculateZn, numsamples, model='BS2')
logging.info('Importance sampling using background model')
samplebg, cbbg = jis.importancesample(
    index, jem.bglikelihoodfn, W, childWmerfreqs[:,0], calculateZn, numsamples, model='BG')
#samples = pd.concat((samplebs, samplebs2, samplebg))
samples = pd.concat((samplebs, samplebg))
samples.index = npy.arange(len(samples))  # Need unique indices for conversion to R dataframe
samplesgrouped = samples.groupby(['model'])

logging.info('Analysing variance')
variances = samplesgrouped['Zweighted'].aggregate(npy.var)
logging.info('Variances:\n%s', variances)
logging.info('Variance ratio: %f', variances['BG'] / variances['BS'])
logging.info('Estimate: %s', sumestimator(samples['Zweighted']))
logging.info('Estimates:\n%s', samplesgrouped['Zweighted'].aggregate(sumestimator))
logging.info('True sum: %s', trueZnsum)

# Examine how close each estimate of the pwm was
pwmbs = jem.normalisepwm(cbbs.cb.summer.sums)
pwmbg = jem.normalisepwm(cbbg.cb.summer.sums)
pwmtrue = jem.normalisepwm(summer.sums)
logging.info('BS PWM distance/base: %f', npy.linalg.norm(pwmtrue - pwmbs, ord=1) / W)
logging.info('BG PWM distance/base: %f', npy.linalg.norm(pwmtrue - pwmbg, ord=1) / W)

# Plot sampled Z
logging.info('Plotting sampled Zn')
grdevices.png(file="sampled-Z.png", width=4, height=3, units="in", res=300)
rsamples = com.convert_to_r_dataframe(samples)
pp = ggplot2.ggplot(rsamples) + \
    ggplot2.aes_string(x='Z', color='factor(model)') + \
    ggplot2.scale_colour_discrete(name="model") + \
    ggplot2.geom_density() + \
    ggplot2.scale_x_log10()
    #ggplot2.scale_x_continuous(limits=FloatVector((0, 1)))
pp.plot()
grdevices.dev_off()

# Plot likelihood ratios
logging.info('Plotting likelihood ratios from binding site samples')
grdevices.png(file="sampled-ratios.png", width=4, height=3, units="in", res=300)
rsamplesbs = com.convert_to_r_dataframe(samples[samples['model'] == 'BS'])
pp = ggplot2.ggplot(rsamplesbs) + \
    ggplot2.aes_string(x='lr') + \
    ggplot2.geom_density() + \
    ggplot2.scale_x_log10()
pp.plot()
grdevices.dev_off()
