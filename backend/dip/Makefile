ifdef DEBUG
		DP=python3 -m pdb
else
		DP=
endif

G1_OUTPUT_DIR ?= /mnt/external/Cranberries/MatthewPhillips/GiNATestImages/GiNA_undistorted
G2_OUTPUT_DIR ?= ./GiNADeuxOutput

install:
		R CMD BATCH packagedeps.R

#Generate Neural Net and output analysis on a single image
nn_gen:
		$(DP) ./segment_test.py --algorithm neural --neural_model mp_test --foreground_image /mnt/external/Cranberries/MatthewPhillips/GiNATestImages/neural_Pics/foreground_small.jpg --background_image /mnt/external/Cranberries/MatthewPhillips/GiNATestImages/neural_Pics/background_small.jpg -i $(G1_OUTPUT_DIR)/BG_1-25.JPG -o BGP_1-25.bin.jpg -t BGP_1-25.csv

#Generate an analysis on a single image without generating a neural model (using previously generated, trained neural net model)
ginadeux:
		$(DP) ./segment_test.py --algorithm neural --neural_model mp_test --min_area 2500 --max_area 30000 --includes include_images.txt -t GiNADeux_output.csv -o $(G2_OUTPUT_DIR)

#Rerun GiNADeux only on its previously generated binary outputs (much faster than regenerating the segmentation through the neural net prediction)
ginadeux_bin:
		$(DP) ./segment_test.py --algorithm neural --neural_model mp_test --min_area 2500 --max_area 30000 --inputs_are_binary --includes include_images_binary.txt -t GiNADeux_output.binary.csv -o $(G2_OUTPUT_DIR)

#Rerun GiNADeux only on its previously generated binary outputs (much faster than regenerating the segmentation through the neural net prediction)
ginadeux_bin2:
		$(DP) ./segment_test.py --algorithm neural --neural_model mp_test --min_area 2500 --max_area 30000 --inputs_are_binary --includes include_images_binary.txt -t GiNADeux_output.binary.2.csv -o $(G2_OUTPUT_DIR)

#Run GiNADeux only on the original GiNA's binary outputs, to see how the calculated traits compare
ginadeux_binorig:
		$(DP) ./segment_test.py --algorithm neural --neural_model mp_test --min_area 2500 --max_area 30000 --inputs_are_binary --includes include_images_binary.orig.txt -t GiNADeux_output.orig.binary.csv -o $(G2_OUTPUT_DIR)


#Compare Matlab GiNA (original) output against GinaDeux
compare:
		$(DP) ./segment_diff.py --i1 $(G1_OUTPUT_DIR)/GiNA_output.csv --i2 $(G2_OUTPUT_DIR)/GiNADeux_output.csv -o comparison.csv

compare_bins:
		$(DP) ./segment_diff.py --i1 $(G1_OUTPUT_DIR)/GiNA_output.csv --i2 $(G2_OUTPUT_DIR)/GiNADeux_output.binary.csv -o comparison.binary.csv

compare_binorig:
		$(DP) ./segment_diff.py --i1 $(G1_OUTPUT_DIR)/GiNA_output.csv --i2 $(G2_OUTPUT_DIR)/GiNADeux_output.orig.binary.csv -o comparison.orig.binary.csv

compare_deuxs:
		$(DP) ./segment_diff.py --i1 $(G2_OUTPUT_DIR)/GiNADeux_output.csv --i2 $(G2_OUTPUT_DIR)/GiNADeux_output.binary.csv -o comparison.deuxs.binary.csv

compare_deuxs2:
		$(DP) ./segment_diff.py --i1 $(G2_OUTPUT_DIR)/GiNADeux_output.binary.csv --i2 $(G2_OUTPUT_DIR)/GiNADeux_output.binary.2.csv -o comparison.deuxs.binary.2.csv

bindiff:
		$(DP) ./binary_diff.py --i1 $(G1_OUTPUT_DIR)/BG_1-25.binary.JPG --i2 BGP_1-25.bin.jpg -o binary.diff.png

#Use R to assess significant differences using paired t-tests and generate plots indicating those with significant differences.
analyze:
	R CMD BATCH '--args comparison_file="comparison.csv" output_file="comparison.pvalues.csv"' analyze_comparisons.R

analyze_pair:
	R CMD BATCH '--args ginaUn_output="/mnt/external/Cranberries/MatthewPhillips/GiNATestImages/GiNA_undistorted/GiNA_output.csv" ginaDeux_output="GiNADeuxOutput/GiNADeux_output.csv" output_file="comparison.paired.pvalues.csv"' analyze_paired.R

analyze_bin:
	R CMD BATCH '--args comparison_file="comparison.binary.csv" output_file="comparison.binary.pvalues.csv"' analyze_comparisons.R

analyze_bin_pair:
	R CMD BATCH '--args ginaUn_output="/mnt/external/Cranberries/MatthewPhillips/GiNATestImages/GiNA_undistorted/GiNA_output.csv" ginaDeux_output="GiNADeuxOutput/GiNADeux_output.binary.csv" output_file="comparison.binary.paired.pvalues.csv"' analyze_paired.R

analyze_binorig:
	R CMD BATCH '--args comparison_file="comparison.orig.binary.csv" output_file="comparison.orig.binary.pvalues.csv"' analyze_comparisons.R

analyze_binorig_pair:
	R CMD BATCH '--args ginaUn_output="/mnt/external/Cranberries/MatthewPhillips/GiNATestImages/GiNA_undistorted/GiNA_output.csv" ginaDeux_output="GiNADeuxOutput/GiNADeux_output.orig.binary.csv" output_file="comparison.orig.binary.paired.pvalues.csv"' analyze_paired.R
