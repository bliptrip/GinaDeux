#Script to analyze the differences b/w original GiNA output and GiNADeux output
#In case we override the workflow on the command-line

ginaUn_output     <- '/mnt/external/Cranberries/MatthewPhillips/GiNATestImages/GiNA_undistorted/GiNA_output.csv'
ginaDeux_output   <- 'GiNADeuxOutput/GiNADeux_output.csv'
output_file		  <- 'comparison.paired.pvalues.csv'
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

ginaUn   <- read.csv(file=ginaUn_output)
ginaDeux <- read.csv(file=ginaDeux_output)

indexes = c("picture", "numbering")
#Shared columns b/w two output datasets
shared   <- intersect(colnames(ginaUn), colnames(ginaDeux))
shared   <- shared[!(shared %in% indexes)]
ginaUn   <- ginaUn[,shared]
ginaDeux <- ginaDeux[,shared]
pvalues.m <- matrix(rep(0, length(shared)), nrow=1, ncol=length(shared))
colnames(pvalues.m) <- shared
pvalues.df <- data.frame(pvalues.m)
#Paired t-test
for(name in shared) {
    pvalues.df[,name] = t.test(as.numeric(ginaUn[,name]), as.numeric(ginaDeux[,name]), mu=0, paired=TRUE)['p.value']
}
write.csv(pvalues.df,output_file)
