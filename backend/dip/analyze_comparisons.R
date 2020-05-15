#Script to analyze the differences b/w original GiNA output and GiNADeux output
#In case we override the workflow on the command-line

comparison_file	  <- ''
input_file1       <- ''
input_file2       <- ''
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
update.packages("tidyverse", repos = "http://mirror.las.iastate.edu/CRAN/", ask = FALSE)
library(tidyverse)

indexes = c("picture", "numbering")

getTTestPValue <- function(x) {
    ret = t.test(as.numeric(x))
    return(ret['p.value'])
}
if (comparison_file != '') {
    #Regular t-test
    comparisons <- read.csv(file=comparison_file)
    comparisons.m <- as.matrix(comparisons)
    comparisons.stripped <- comparisons.m[,!(colnames(comparisons.m) %in% indexes)]
    comparison.pvalues <- apply(comparisons.stripped, 2, getTTestPValue)
    comparison.pvalues <- setNames(data.frame(comparison.pvalues), colnames(comparisons.stripped))
    comparison.pvalues <- data.frame(c(setNames(vector(mode="character",length=2),indexes), comparison.pvalues))
    comparison.pvalues.collated  <- rbind(comparisons, comparison.pvalues)
    write.csv(comparison.pvalues.collated,output_file)
    #pivot_longer() to make it easier to do box plots, and separate plots based on different categories
    #color_params <- c("B_med","B_var","G_med","G_var","R_med","R_var","bwColor_r","vbwColor_r")
    color_var_params <- c("B_var","G_var","R_var")
    comparisons_color_var_longer = comparisons %>%
                                pivot_longer(color_var_params, names_to='feature', values_to='differences')
    #Generate color_var boxplot
    g_color_var = ggplot(comparisons_color_var_longer, aes(feature,differences)) + 
        geom_boxplot() +
        labs(title="RGB Color Variance Comparison Boxplots: GiNA vs. GiNADeux", x="GiNA-derived Features", y="GiNA vs. GiNADeux Absolute Differences") +
        theme_minimal() +
        theme(plot.title = element_text(face="bold",size=24,vjust=2),
              axis.title = element_text(face="bold",size=16))

    ggsave(filename="comparisons_color_var_boxplot.png", plot=g_color_var, device="png", bg="white", width=33, height=25, units="cm", dpi=300)

    color_params <- c("B_med","G_med","R_med","bwColor_r","vbwColor_r")
    comparisons_color_longer = comparisons %>%
                                pivot_longer(color_params, names_to='feature', values_to='differences')
    #Generate color boxplot
    g_color = ggplot(comparisons_color_longer, aes(feature,differences)) + 
        geom_boxplot() +
        labs(title="RGB Median Value/BW Variance Comparison Boxplots: GiNA vs. GiNADeux", x="GiNA-derived Features", y="GiNA vs. GiNADeux Absolute Differences") +
        theme_minimal() +
        theme(plot.title = element_text(face="bold",size=24,vjust=2),
              axis.title = element_text(face="bold",size=16))
    ggsave(filename="comparisons_color_boxplot.png", plot=g_color, device="png", bg="white", width=33, height=25, units="cm", dpi=300)
    relative_params <- c("blobLength_r","blobSolidity_r","blobVolume_r","blobWidth_r","projectedArea_r","projectedPerimeter_r","LvsW_r","blobEccentricity_r")
    comparisons_relative_longer = comparisons %>%
                                    pivot_longer(relative_params, names_to='feature', values_to='differences')
    #Generate relative boxplot
    g_relative = ggplot(comparisons_relative_longer, aes(feature,differences)) + 
        geom_boxplot() +
        labs(title="Relative Features (normalized to size standards) Comparison Boxplots: GiNA vs. GiNADeux", x="GiNA-derived Features", y="GiNA vs. GiNADeux Absolute Differences") +
        theme_minimal() +
        theme(plot.title = element_text(face="bold",size=24,vjust=2),
              axis.title = element_text(face="bold",size=16))
    ggsave(filename="comparisons_relative_boxplot.png", plot=g_relative, device="png", bg="white", width=33, height=25, units="cm", dpi=300)
    real_params <- c("blobLength_r2","blobVolume_r2","blobWidth_r2","projectedArea_r2","projectedPerimeter_r2","locationX","locationY")
    comparisons_real_longer = comparisons %>% pivot_longer(real_params, names_to='feature', values_to='differences')
    #Generate real boxplot
    g_real = ggplot(comparisons_real_longer, aes(feature,differences)) + 
        geom_boxplot() +
        labs(title="Real-value (cm)  Comparison Boxplots: GiNA vs. GiNADeux", x="GiNA-derived Features", y="GiNA vs. GiNADeux Absolute Differences") +
        theme_minimal() +
        theme(plot.title = element_text(face="bold",size=24,vjust=2),
              axis.title = element_text(face="bold",size=16))
    ggsave(filename="comparisons_real_boxplot.png", plot=g_real, device="png", bg="white", width=33, height=25, units="cm", dpi=300)

} else if( (input_file1 != '') && (input_file2 != '') ) {
    #Paired t-test
    input1 <- read.csv(file=input_file1)
    input2 <- read.csv(file=input_file2)
    #Create box plots
}

