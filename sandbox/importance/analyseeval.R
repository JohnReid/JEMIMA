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
# filename <- "statsdf-12000-0020.csv.gz"
# filename <- "statsdf-00003-0005.csv.gz"
# filename <- "statsdf-00048-0020.csv.gz"
filename <- "statsdf-08000-0018.csv.gz"
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
    geom_boxplot() + facet.width + coord_cartesian(ylim=c(.5, 2.))

#
# Check how close the estimated PWMs are to the truth
#
distlims = c(1e-3, 1e0)
distxcoords = coord_cartesian(xlim=distlims)
distycoords = coord_cartesian(ylim=distlims)
distxycoords = coord_cartesian(xlim=distlims, ylim=distlims)
ggplot(statsdf, aes(x=distperbase, color=method)) + 
    geom_density() + facet.width +
    scale_x_log10() +
    distxcoords

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
    scale_x_log10(breaks=seq(min(statsdf$numsamples), max(statsdf$numsamples), by=100)) +
    scale_y_log10() +
    distycoords

#
# How does the iteration affect the accuracy of the PWM?
#
ggplot(
    data=statsdf[has.KL & positive.KL],
    aes(x=factor(iteration), y=distperbase, color=method)) +
    geom_boxplot(outlier.size=1) +
    scale_y_log10() + distycoords

# Get the distperbase for each method
distperbasedf <- statsdf %>%
    filter(iteration > 2) %>%
    group_by(seedidx, iteration) %>%
    select(seedidx, iteration, method, distperbase, W) %>%
    dcast(seedidx + iteration + W~ method, value.var="distperbase")
head(distperbasedf)
ggplot(distperbasedf, aes_string(x=method.names[1], y=method.names[2], color="iteration")) +
    geom_point(alpha=alpha) + facet.width +
    scale_x_log10() + scale_y_log10() +
    distxycoords

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
