from yake import KeywordExtractor as YakeKW
from kep.utility import getlanguage, CreateKeywordsFolder, LoadFiles
import os

class YAKE(object):
    def __init__(self, numOfKeywords, pathData, dataset_name):
        self.__lan = getlanguage(pathData + "/Datasets/" + dataset_name)
        self.__numOfKeywords = numOfKeywords
        self.__dataset_name = dataset_name
        self.__pathData = pathData
        self.__pathToDatasetName = self.__pathData + "/Datasets/" + self.__dataset_name
        self.__keywordsPath = self.__pathData + '/Keyphrases/YAKE/' + self.__dataset_name
        self.__algorithmName = "YAKE"

    def LoadDatasetFiles(self):
        # Gets all files within the dataset fold
        listFile = LoadFiles(self.__pathToDatasetName + '/docsutf8/*')
        print(f"\ndatasetID = {self.__dataset_name}; Number of Files = {len(listFile)}; Language of the Dataset = {self.__lan}")
        return listFile

    def CreateKeywordsOutputFolder(self):
        # Set the folder where keywords are going to be saved
        CreateKeywordsFolder(self.__keywordsPath)

    def runSingleDoc(self, text):
        #Get YAKE keywords
        # 1. create a YAKE extractor.
        extractor = YakeKW(lan = self.__lan, top = self.__numOfKeywords)


        try:
            # 2. get the numOfKeywords-highest scored candidates as keyphrases
            keywords = extractor.extract_keywords(text)
        except:
            keywords = []

        return keywords

    def runMultipleDocs(self, listOfDocs):
        self.CreateKeywordsOutputFolder()

        for j, doc in enumerate(listOfDocs):
            # docID keeps the name of the file (without the extension)
            docID = '.'.join(os.path.basename(doc).split('.')[0:-1])

            # Reads the File
            with open(doc, encoding='utf-8') as fil:
                text = fil.read()

            keywords = self.runSingleDoc(text)

            # Save the keywords; score (on Algorithms/NameOfAlg/Keywords/NameOfDataset
            with open(os.path.join(self.__keywordsPath, docID), 'w', encoding="utf-8") as out:
                for (key, score) in keywords:
                    out.write(f'{key} {score}\n')

            # Track the status of the task
            print(f"\rFile: {j + 1}/{len(listOfDocs)}", end='')

        print(f"\n100% of the Extraction Concluded")

    def ExtractKeyphrases(self):
        print(f"\n\n-----------------Extract Keyphrases--------------------------")
        listOfDocs = self.LoadDatasetFiles()
        self.runMultipleDocs(listOfDocs)
