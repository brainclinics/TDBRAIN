# TDBRAIN
Code used in the paper The Two Decades Brainclinics Research Archive for Insights in Neurophysiology (TDBRAIN) database

# TD-BRAIN-processing-code
This python API can be used for automatic* EEG artifact removal (eye-blink) and detection. as well as processing and producing the figures in the TD-BRAIN manuscript. It loads the .csv files that are saved in the format described in the manuscript. If necessary it also repairs channels when there are too many artifacts within one channel. It wil mark the artifacts that it found in an artifact channel. This also prints out a .pdf report with the detected artifacts depicted in the ‘artifacts’ channel. 
In the next step (interprocessing.py) you can define if you want to remove them or not and if you want to segment the data into epochs. In this step you can also apply remontaging and filtering if wanted. Because it is an automatic routine and we apply it to >100 datasets at a time, we did accept there is a trade-off where we probably miss some artifacts for some subjects and/or detect too many for other subjects. 

The module is conceptually developed along the lines of: Arns, M. et al. EEG alpha asymmetry as a gender-specific predictor of outcome to acute treatment with different antidepressant medications in the randomized iSPOT-D study. Clin Neurophysio 127, 509–19 (2016) (Supplement) and https://www.fieldtriptoolbox.org/about/

*although it can also be adapted for each individual, in a more manual setting.

##### CONTENTS:
1. end2end_alphaPowerandiAPF.py: 
>>can be run from the terminal over all subjectdirectories in the sourcepath, will be saved in the sampe directory >>structure in the preprocpath and results will be saved in the resultspath
```
$ python end2end_alphaPowerandiAPF.py sourcepath 
```
>>for example:
```
$ python end2end_alphaPowerandiAPF.py '/NAS/database/BCD_OA_database/BCD_OA_std_csvnew/' 
```

2. autopreprocessing_pipeline.py:  
>>pipeline using autopreprocessing.py looping over subjects' folders. This pipeline    function takes as input the varargsin dictionary which should contain ’sourcepath’ (string), ’savepath’ (string)  and ‘condition’ (1d array of strings). ‘startsubject' and ‘subject' are optional inputs, where you can chose to start with a certain subject number (if you have stopped processing in the middle of a processing round for instance) or preprocess one subject indicated by number in array or idcode (first 8 digits, without session number). 

3. autopreprocessing_OA.py:        
>>sourcecode for the actual preprocessing

4. interprocessing.py:             
>>sourcecode for post-processing (segmentation etc)
5. pio.py:                          
>>contains FilepathFinder, a low level object used for reading all data fitting a requirement in all folders and subfolders

