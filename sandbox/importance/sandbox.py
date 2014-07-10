#!/usr/bin/env python

#
# Copyright John Reid 2014
#

import jemima
reload(jemima)
import jemima.importancesampling
reload(jemima.importancesampling)
import jemima as jem
import jemima.importancesampling as jis
from jemima import wmers

import pandas as pd
import pandas.rpy.common as com
import seqan.traverse
import numpy as npy
import numpy.random as rdm

import logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

import matplotlib as mpl
mpl.use('agg')

import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects.packages import importr
grdevices = importr('grDevices')


def makedf(iscb, Zncalculator, **kwargs):
    """Make a pandas data frame containing the Zs and importance ratios
    from the callback."""
    # Create data frame
    kwargs.update({
        'Z' : map(Zncalculator, iscb.Xns),
        'ir': iscb.irs,
    })
    df = pd.DataFrame(kwargs)
    df['Zweighted'] = df['Z'] * df['ir']
    return df


numseedsites = 20
pseudocount = 1.
W = 11
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
runx1withpc = jem.addpseudocounts(runx1pwm, numseedsites, pseudocount)
assert W == len(runx1withpc)
# jem.logo(runx1pwm, 'runx1')
# jem.logo(runx1withpc, 'runx1-pc')

logging.info('Loading sequences')
# seqs = seqan.StringDNASet(('AAAAAAAA', 'ACGTACGT', 'TATATATA'))
numbases, seqs, ids = seqan.readFastaDNA('T00759-small.fa')
logging.info('Loaded %d bases from %d sequences', numbases, len(seqs))
lambda_ = len(seqs) / float(numbases)

logging.info('Building index')
index = seqan.IndexStringDNASetESA(seqs)

logging.info('Counting W-mers')
Ws = [W]
Wmercounts = npy.zeros((2*len(index), len(Ws)), dtype=npy.uint)
numWmers = wmers.countWmersMulti(index.topdownhistory(), Ws, Wmercounts)[0]
logging.info('Got %d %d-mers', numWmers, W)
childWmerfreqs = npy.zeros((2*len(index), len(Ws), jem.SIGMA))
wmers.countWmerChildren(index.topdownhistory(), W, Wmercounts, childWmerfreqs)
childWmerfreqs = jem.normalisearray(childWmerfreqs)
sumestimator = jis.makesumestimator(numWmers)

logging.info('Importance sampling using background model to find one seed')
rdm.seed(1)
memocb = jis.importancesample(
    index, W, childWmerfreqs[:, 0], jem.bglikelihoodfn,
    numsamples=1, callback=jis.ISCbMemo())
pwm = jem.pwmfromWmer(memocb.Xns[0], numseedsites, 1.)
pwmlikelihoodfn = jem.createpwmlikelihoodfn(pwm)
calculateZn = jem.createZncalculatorFn(pwm, lambda_)
jem.logo(pwm, 'seed')

numsamples = 3000
distsbs = []
distsbg = []
truesums = []
varratios = []
for iteration in xrange(1):
    logging.debug('Calculating true Zn sums')
    summer = jis.ZnSumCb(W)
    sumvisitor = jis.ZnCalcVisitor(W, calculateZn, summer)
    seqan.traverse.topdownhistorytraversal(index.topdownhistory(), sumvisitor)
    logging.debug('Sums:\n%s', summer.sums)
    trueZnsum = summer.sums[0].sum()
    truesums.append(trueZnsum)
    logging.info('True sum: %s', trueZnsum)

    logging.debug('Importance sampling using binding site model')
    cbbs = jis.importancesample(
        index, W, childWmerfreqs[:, 0], pwmlikelihoodfn,
        numsamples, jis.ISMemoCbAdaptor(jis.ZnSumCb(W), calculateZn))
    samplebs = makedf(cbbs, calculateZn, model='BS')
    logging.debug('Variances:\n%s', cbbs.cb.variances())

    logging.debug('Importance sampling using background model')
    cbbg = jis.importancesample(
        index, W, childWmerfreqs[:, 0], jem.bglikelihoodfn,
        numsamples, jis.ISMemoCbAdaptor(jis.ZnSumCb(W), calculateZn))
    samplebg = makedf(cbbg, calculateZn, model='BG')

    # Examine how close each estimate of the pwm was
    pwmbs = jem.normalisearray(cbbs.cb.sums)
    pwmbg = jem.normalisearray(cbbg.cb.sums)
    pwmtrue = jem.normalisearray(summer.sums)
    lambda_ = trueZnsum / float(numbases)
    calculateZn = jem.createZncalculatorFn(pwmtrue, lambda_)
    pwmlikelihoodfn = jem.createpwmlikelihoodfn(pwmtrue)
    distperbasebs = npy.linalg.norm(pwmtrue - pwmbs, ord=1) / W
    distperbasebg = npy.linalg.norm(pwmtrue - pwmbg, ord=1) / W
    distsbs.append(distperbasebs)
    distsbg.append(distperbasebg)
    logging.info('BS PWM distance/base: %f', distperbasebs)
    logging.info('BG PWM distance/base: %f', distperbasebg)
    jem.logo(pwmbs, '%04d-BS' % iteration)
    jem.logo(pwmbg, '%04d-BG' % iteration)
    jem.logo(pwmtrue, '%04d-true' % iteration)

    logging.debug('Analysing variance')
    samples = pd.concat((samplebs, samplebg))
    # Need unique indices for conversion to R dataframe
    samples.index = npy.arange(len(samples))
    samplesgrouped = samples.groupby(['model'])
    variances = samplesgrouped['Zweighted'].aggregate(npy.var)
    # logging.info('Variances:\n%s', variances)
    varianceratio = variances['BG'] / variances['BS']
    varratios.append(varianceratio)
    logging.info('Variance ratio: %f', varianceratio)
    logging.debug('Estimate: %s', sumestimator(samples['Zweighted']))
    logging.debug('Estimates:\n%s',
                  samplesgrouped['Zweighted'].aggregate(sumestimator))
    logging.debug('True sum: %s', trueZnsum)

emdf = pd.DataFrame({
    'BSdists' : distsbs,
    'BGdists' : distsbg,
    'truesums' : truesums,
    'varratios' : varratios,
})

# Plot sampled Z
logging.info('Plotting sampled Zn')
grdevices.png(file="sampled-Z.png", width=4, height=3, units="in", res=300)
rsamples = com.convert_to_r_dataframe(samples)
pp = ggplot2.ggplot(rsamples) + \
    ggplot2.aes_string(x='Z', color='factor(model)') + \
    ggplot2.scale_colour_discrete(name="model") + \
    ggplot2.geom_density() + \
    ggplot2.scale_x_log10()
# ggplot2.scale_x_continuous(limits=FloatVector((0, 1)))
pp.plot()
grdevices.dev_off()

# Plot likelihood ratios
logging.info('Plotting likelihood ratios from binding site samples')
grdevices.png(file="sampled-ratios.png",
              width=4, height=3, units="in", res=300)
rsamplesbs = com.convert_to_r_dataframe(samples[samples['model'] == 'BS'])
pp = ggplot2.ggplot(rsamplesbs) + \
    ggplot2.aes_string(x='ir') + \
    ggplot2.geom_density() + \
    ggplot2.scale_x_log10()
pp.plot()
grdevices.dev_off()
