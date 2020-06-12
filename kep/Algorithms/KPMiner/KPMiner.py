import pke
import os
from kep.utility import ComputeDF, getlanguage, CreateKeywordsFolder, LoadFiles

class KPMiner(object):
    def __init__(self, numOfKeywords, pathData, dataset_name, normalization):
        self.__lan = getlanguage(pathData + "/Datasets/" + dataset_name)
        self.__numOfKeywords = numOfKeywords
        self.__dataset_name= dataset_name
        self.__normalization = normalization
        self.__pathData = pathData
        self.__pathToDFFile = self.__pathData + "/Models/dfs/" + self.__dataset_name + '_dfs.gz'
        self.__pathToDatasetName = self.__pathData + "/Datasets/" + self.__dataset_name
        self.__keywordsPath = self.__pathData + '/Keyphrases/KPMiner/' + self.__dataset_name
        self.__algorithmName = "KPMiner"

    def ComputeDocumentFrequency(self):
        if self.__lan not in ['en', 'pt', 'fr', 'it', 'nl', 'de']:
            ComputeDF(self.__pathToDatasetName + '/docsutf8', 'en', self.__normalization, self.__pathToDFFile)
        else:
            ComputeDF(self.__pathToDatasetName + '/docsutf8', self.__lan, self.__normalization, self.__pathToDFFile)

    def LoadDatasetFiles(self):
        # Gets all files within the dataset fold
        listFile = LoadFiles(self.__pathToDatasetName + '/docsutf8/*')
        print(f"\ndatasetID = {self.__dataset_name}; Number of Files = {len(listFile)}; Language of the Dataset = {self.__lan}")
        return listFile

    def CreateKeywordsOutputFolder(self):
        # Set the folder where keywords are going to be saved
        CreateKeywordsFolder(self.__keywordsPath)

    def runSingleDoc(self, doc):
        #Get KPMiner keywords
        # 1. create a SingleRank extractor.
        extractor = pke.unsupervised.KPMiner()

        # 2. load the content of the document in a given language
        extractor.load_document(input=doc, language=self.__lan, normalization=self.__normalization)

        # 3. select {1-5}-grams that do not contain punctuation marks or
        #    stopwords as keyphrase candidates. Set the least allowable seen
        #    frequency to 5 and the number of words after which candidates are
        #    filtered out to 200.
        lasf = 5
        cutoff = 200
        extractor.candidate_selection(lasf=lasf, cutoff=cutoff)

        try:
            # 4. weight the candidates using KPMiner weighting function.
            df = pke.load_document_frequency_file(input_file=self.__pathToDFFile)
            alpha = 2.3
            sigma = 3.0
            extractor.candidate_weighting(df=df, alpha=alpha, sigma=sigma)

            # 5. get the numOfKeywords-highest scored candidates as keyphrases
            keywords = extractor.get_n_best(n=self.__numOfKeywords)
        except:
            keywords = []

        return keywords

    def runMultipleDocs(self, listOfDocs):
        self.CreateKeywordsOutputFolder()

        for j, doc in enumerate(listOfDocs):
            # docID keeps the name of the file (without the extension)
            docID = '.'.join(os.path.basename(doc).split('.')[0:-1])

            keywords = self.runSingleDoc(doc)

            # Save the keywords; score (on Algorithms/NameOfAlg/Keywords/NameOfDataset
            with open(os.path.join(self.__keywordsPath, docID), 'w', encoding="utf-8") as out:
                for (key, score) in keywords:
                    out.write(f'{key} {score}\n')

            # Track the status of the task
            print(f"\rFile: {j + 1}/{len(listOfDocs)}", end='')

        print(f"\n100% of the Extraction Concluded")

    def ExtractKeyphrases(self):
        print(f"\n------------------------------Compute DF--------------------------")
        self.ComputeDocumentFrequency()

        print(f"\n\n-----------------Extract Keyphrases--------------------------")
        listOfDocs = self.LoadDatasetFiles()
        self.runMultipleDocs(listOfDocs)

