from os import listdir
from os.path import isfile, join
import random
from pathlib import Path
import shutil

#from sklearn import  train_test_split


def writedataset(ImagesFiles,AnnotsFiles,Indexes , dirPath ):
    writeDir = join(mypath,dirPath)
    writeAnnotDir = join(mypath, dirPath+"Annot")

    Path(writeDir).mkdir(parents=True, exist_ok=True)
    Path(writeAnnotDir).mkdir(parents=True, exist_ok=True)

    txtFilePath = open(join(mypath,dirPath+".txt"), "w")

    for i in Indexes:
       imageFile      = ImagesFiles[i]
       imageAnnotFile = AnnotsFiles[i]

       imageSrc       = join(imagePath,imageFile )
       imageAnnotSrc  = join(annoPath,imageAnnotFile)

       imageDsc = join(writeDir, imageFile )
       AnnotDsc = join(writeAnnotDir, imageAnnotFile)

       txtline =  imageDsc +  " "  +  AnnotDsc + "\n"
       txtFilePath.writelines(txtline)

       shutil.copyfile(imageSrc, imageDsc)
       shutil.copyfile(imageAnnotSrc, AnnotDsc)

    txtFilePath.close()

def getTrainValTestRandomIndexes(n):
    randGeneratedIndexes = []
    for i in range(n):
        randGeneratedIndexes.append(i)
    random.shuffle(randGeneratedIndexes)

    step = int(n/3)

    trainIndexes = randGeneratedIndexes[0:step]
    valIndexes   = randGeneratedIndexes[step:2*step]
    testIndexes  = randGeneratedIndexes[2*step : n]

    return  trainIndexes, valIndexes, testIndexes



mypath =  "data\CairoSegdata"
imageDir = "image128128100"
annotDir = "annotat128128100"

trainDir ="train"
valDir   ="val"
testDir  ="test"


imagePath = join(mypath,imageDir)
annoPath = join(mypath,annotDir)

ImagesFiles = [f for f in sorted(listdir(imagePath)) if isfile(join(imagePath, f)) and f.endswith(".tif")]
AnnotsFiles = [f for f in sorted(listdir(annoPath)) if isfile(join(annoPath, f)) and f.endswith(".tif")]

trainIndexes, valIndexes, testIndexes = getTrainValTestRandomIndexes(len(ImagesFiles))

writedataset(ImagesFiles,AnnotsFiles,trainIndexes,trainDir)
writedataset(ImagesFiles,AnnotsFiles,valIndexes,valDir)
writedataset(ImagesFiles,AnnotsFiles,testIndexes,testDir)










m = 1

