#!/usr/bin/Rscript

# install.packages("datatable")
library(data.table)
library(dplyr)
library(ggplot2)

#
# Load data
statsdf <- data.table(
    read.csv(gzfile("statsdf-10000-0025.csv.gz")),
    key=c("seed", "iteration"))
dim(statsdf)
class(statsdf)
sapply(statsdf, class)

has.KL <- ! (is.na(statsdf[["KLestimatetrue"]]) | is.na(statsdf[["KLtrueestimate"]]))
positive.KL <- (0 < statsdf[["KLestimatetrue"]]) & (0 < statsdf[["KLtrueestimate"]])
min(statsdf$distperbase)
min(statsdf[has.KL]$distperbase)
min(statsdf[has.KL]$KLtrueestimate)
min(statsdf[has.KL]$KLestimatetrue)
min(statsdf[positive.KL & has.KL]$distperbase)
alpha <- .1

ggplot(
    data=statsdf[has.KL & positive.KL],
    aes(x=KLtrueestimate, y=KLestimatetrue, color=method)) +
    geom_point(alpha=alpha) +
    scale_x_log10() + scale_y_log10()

ggplot(
    data=statsdf[has.KL & positive.KL],
    aes(x=numsamples, y=distperbase, color=method)) +
    geom_point(alpha=alpha) +
    scale_x_log10() + scale_y_log10()

ggplot(
    data=statsdf[has.KL & positive.KL],
    aes(x=factor(iteration), y=distperbase, color=method)) +
    geom_boxplot(outlier.size=1) +
    scale_y_log10()

ggplot(
    data=statsdf[has.KL & positive.KL],
    aes(x=lambdatrue, y=distperbase, color=method)) +
    geom_point(alpha=alpha) +
    scale_x_log10() + scale_y_log10()

# sum(statsdf[,method[1],by="seed,iteration"]$V1 == "PWMweights")
# sum(statsdf[,method[2],by="seed,iteration"]$V1 == "uniformweights")
cmp.dist <- statsdf[,
    list(
        distratio = distperbase[2] / distperbase[1],
        durationratio = duration[2] / duration[1],
        varratio = var[2] / var[1]),
    by="seed,iteration"]

# Plot the distance ratio without outliers
# compute lower and upper whiskers
ylim1 = boxplot.stats(log10(cmp.dist$distratio))$stats[c(1, 5)]
# scale y limits based on ylim1
ggplot(
    data=cmp.dist,
    aes(x=factor(iteration), y=distratio)) +
    geom_boxplot(outlier.size=1) + scale_y_log10() +
    coord_cartesian(ylim = c(.9, 1.1)*10**ylim1)

# Plot the variance ratio without outliers
ylim1 = boxplot.stats(log10(cmp.dist$varratio))$stats[c(1, 5)]
ggplot(
    data=cmp.dist,
    aes(x=factor(iteration), y=varratio)) +
    geom_boxplot(outlier.size=1) + scale_y_log10() +
    coord_cartesian(ylim = c(.9, 1.1)*10**ylim1)

# Plot the variance ratio against the distance ratio
ggplot(
    data=cmp.dist,
    aes(x=distratio, y=varratio, color=iteration)) +
    geom_hline(yintercept=1, alpha=alpha) +
    geom_vline(xintercept=1, alpha=alpha) +
    geom_point(alpha=alpha) +
    scale_x_log10() + scale_y_log10() +
    scale_colour_gradientn(colours=rainbow(4)) +
    coord_cartesian(xlim=c(1e-2, 1e2), ylim=c(1e5, 1e-5))

ggplot(data=cmp.dist, aes(x=durationratio)) + geom_density() + scale_x_log10()
