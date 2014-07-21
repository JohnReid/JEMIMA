#!/usr/bin/Rscript

# install.packages("datatable")
library(data.table)
library(dplyr)
library(reshape2)
library(ggplot2)
library(GGally)

#
# Load data
#
# filename <- "statsdf-10000-0025.csv.gz"
# filename <- "statsdf-00003-0005.csv.gz"
filename <- "statsdf-00048-0020.csv.gz"
# filename <- "statsdf-12000-0020.csv.gz"
statsdf <- data.table(read.csv(gzfile(filename)), key=c("seed", "iteration"))
method.names <- levels(statsdf$method)
method.names
# Guess seedidx if not provided
if(! "seedidx" %in% names(statsdf)) {
    statsdf$seedidx <- (1:nrow(statsdf) + 1) %/% 2
}
dim(statsdf)
class(statsdf)
sapply(statsdf, class)

#
# Check how accurate our Z estimates are
#
statsdf %>%
    group_by(method, W) %>%
    summarise(mean(Znsumestimate / Znsumtrue), var(Znsumestimate / Znsumtrue))
facet.width = facet_wrap(~ W)
ggplot(
    # filter(statsdf, method=="PWMoccs" | method=="uniformoccs"),
    statsdf,
    aes(x=method, y=Znsumestimate / Znsumtrue)) +
    scale_y_log10() +
    geom_boxplot() + facet.width

#
# Check how close the estimated PWMs are to the truth
#
ggplot(statsdf, aes(x=distperbase, color=method)) + 
    geom_density() + facet.width +
    scale_x_log10()

has.KL <- ! (is.na(statsdf[["KLestimatetrue"]]) | is.na(statsdf[["KLtrueestimate"]]))
positive.KL <- (0 < statsdf[["KLestimatetrue"]]) & (0 < statsdf[["KLtrueestimate"]])
min(statsdf$distperbase)
min(statsdf[has.KL]$distperbase)
min(statsdf[has.KL]$KLtrueestimate)
min(statsdf[has.KL]$KLestimatetrue)
min(statsdf[positive.KL & has.KL]$distperbase)
alpha <- .1

#
# Investigate if directionality of KL has much effect
#
ggplot(
    data=statsdf[has.KL & positive.KL],
    aes(x=KLtrueestimate, y=KLestimatetrue, color=method)) +
    geom_point(alpha=alpha) +
    scale_x_log10() + scale_y_log10()

#
# Does number of samples correlate with PWM accuracy?
#
ggplot(
    data=statsdf[has.KL & positive.KL],
    aes(x=numsamples, y=distperbase, color=method)) +
    geom_point(alpha=alpha) +
    scale_x_log10() + scale_y_log10()

#
# How does the iteration affect the accuracy of the PWM?
#
ggplot(
    data=statsdf[has.KL & positive.KL],
    aes(x=factor(iteration), y=distperbase, color=method)) +
    geom_boxplot(outlier.size=1) +
    scale_y_log10()

# Get the distperbase for each method
distperbasedf <- statsdf %>%
    group_by(seedidx, iteration) %>%
    select(seedidx, iteration, method, distperbase) %>%
    dcast(seedidx + iteration ~ method, value.var="distperbase")

ggplot(distperbasedf, aes_string(x=method.names[1], y=method.names[2], color="iteration")) +
    geom_point(alpha=alpha)





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
    geom_hline(yintercept=1, alpha=.2) +
    geom_vline(xintercept=1, alpha=.2) +
    geom_point(alpha=alpha) +
    scale_x_log10() + scale_y_log10() +
    scale_colour_gradientn(colours=rainbow(4)) +
    coord_cartesian(xlim=c(1e-2, 1e2), ylim=c(1e5, 1e-5))

ggplot(data=cmp.dist, aes(x=durationratio)) + geom_density() + scale_x_log10()

pairs.plot <- ggpairs(
    distperbasedf,
    columns=method.names,
    axisLabels="show",
    diag=list(continuous="density"),
)
lims = c(0, max(statsdf$distperbase))
scatter.coords = coord_cartesian(xlim=lims, ylim=lims)
density.coords = coord_cartesian(xlim=lims)
for( rowidx in 1:length(method.names) ) {
    for( colidx in 1:rowidx ) {
        if( rowidx > colidx ) {
            coords <- scatter.coords
        } else if( rowidx == colidx ) {
            coords <- density.coords
        }
        pairs.plot <- putPlot(
            pairs.plot,
            getPlot(pairs.plot, rowidx, colidx) + coords,
            rowidx, colidx)
    }
}
pairs.plot
# print pairs.plot
