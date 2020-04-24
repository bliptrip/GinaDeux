#Generate Neural Net and output analysis on a single image
nn_gen:
		./segment_test.py --algorithm neural --neural_model mp_test --foreground_image /mnt/external/Cranberries/MatthewPhillips/GiNATestImages/neural_Pics/foreground_small.jpg --background_image /mnt/external/Cranberries/MatthewPhillips/GiNATestImages/neural_Pics/background_small.jpg -i /mnt/external/Cranberries/MatthewPhillips/GiNATestImages/GiNA_undistorted/BG_1-25.JPG -o BGP_1-25.bin.jpg -t BGP_1-25.csv

#Generate an analysis on a single image without generating a neural model (using previously generated, trained neural net model)
ginadeux:
		./segment_test.py --algorithm neural --neural_model mp_test --min_area 2500 --max_area 30000 --includes include_images.txt -t GiNADeux_output.csv -o GiNADeuxOutput

ginadeux_d:
		python3 -m pdb ./segment_test.py --algorithm neural --neural_model mp_test --min_area 2500 --max_area 30000 --includes include_images.txt -t GiNADeux_output.csv -o GiNADeuxOutput

#Compare Matlab GiNA (original) output against GinaDeux
compare:
		./segment_diff.py --i1 /mnt/external/Cranberries/MatthewPhillips/GiNATestImages/GiNA_undistorted/GiNA_output.csv --i2 ./GiNADeuxOutput/GiNADeux_output.csv -o comparison.csv

compare_d:
		python3 -m pdb ./segment_diff.py --i1 /mnt/external/Cranberries/MatthewPhillips/GiNATestImages/GiNA_undistorted/GiNA_output.csv --i2 ./GiNADeuxOutput/GiNADeux_output.csv -o comparison.csv

bindiff:
		./binary_diff.py --i1 /mnt/external/Cranberries/MatthewPhillips/GiNATestImages/GiNA_undistorted/BG_1-25.binary.JPG --i2 BGP_1-25.bin.jpg -o binary.diff.png

bindiff_d:
		python3 -m pdb ./binary_diff.py --i1 /mnt/external/Cranberries/MatthewPhillips/GiNATestImages/GiNA_undistorted/BG_1-25.binary.JPG --i2 BGP_1-25.bin.jpg -o binary.diff.png
