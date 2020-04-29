#Script to analyze the differences b/w original GiNA output and GiNADeux output
#In case we override the workflow on the command-line

comparison_file	  <- 'comparison.csv'
output_file		  <- 'comparison.pvalues.csv'
args = commandArgs(trailingOnly=TRUE)
if(length(args)!=0) {
    for(i in 1:length(args)){
        eval(parse(text=args[[i]]))
    }
}

if(!require(ggplot2)){install.packages("ggplot2", repos = "http://mirror.las.iastate.edu/CRAN/", dependencies = TRUE)}
library(ggplot2)
#library(ggthemes)
#library(plotly)
#library(RColorBrewer)
if(!require(tidyverse)){install.packages("tidyverse", repos = "http://mirror.las.iastate.edu/CRAN/", dependencies = TRUE)}
#library(tidyverse)

comparisons <- read.csv(file=comparison_file)

getTTestPValue <- function(x) {
    ret = t.test(as.numeric(x))
    return(ret['p.value'])
}
indexes = c("picture", "numbering")
#Paired t-test
comparisons.m <- as.matrix(comparisons)
comparisons.stripped <- comparisons.m[,!(colnames(comparisons.m) %in% indexes)]
comparison.pvalues <- apply(comparisons.stripped, 2, getTTestPValue)
comparison.pvalues <- setNames(data.frame(comparison.pvalues), colnames(comparisons.stripped))
comparison.pvalues <- data.frame(c(setNames(vector(mode="character",length=2),indexes), comparison.pvalues))
comparison.pvalues.collated  <- rbind(comparisons, comparison.pvalues)
write.csv(comparison.pvalues.collated,output_file)
