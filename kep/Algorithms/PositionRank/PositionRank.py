import pke
from kep.utility import getlanguage, CreateKeywordsFolder, LoadFiles
import os

class PositionRank(object):
    def __init__(self, numOfKeywords, pathData, dataset_name, normalization):
        self.__normalization = normalization
        self.__lan = getlanguage(pathData + "/Datasets/" + dataset_name)
        self.__numOfKeywords = numOfKeywords
        self.__dataset_name = dataset_name
        self.__pathData = pathData
        self.__pathToDatasetName = pathData + "/Datasets/" + dataset_name
        self.__keywordsPath = self.__pathData + '/Keyphrases/PositionRank/' + self.__dataset_name
        self.__algorithmName = "PositionRank"

    def LoadDatasetFiles(self):
        # Gets all files within the dataset fold
        listFile = LoadFiles(self.__pathToDatasetName + '/docsutf8/*')
        print(f"\ndatasetID = {self.__dataset_name}; Number of Files = {len(listFile)}; Language of the Dataset = {self.__lan}")
        return listFile

    def CreateKeywordsOutputFolder(self):
        # Set the folder where keywords are going to be saved
        CreateKeywordsFolder(self.__keywordsPath)

    def runSingleDoc(self, doc):
        # define the valid Part-of-Speeches to occur in the graph
        pos = {'NOUN', 'PROPN', 'ADJ'}

        #define the grammar for selecting the keyphrase candidates
        grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

        #Get PositionRank keywords
        # 1. create a PositionRank extractor.
        extractor = pke.unsupervised.PositionRank()

        # 2. load the content of the document in a given language
        extractor.load_document(input=doc, language=self.__lan, normalization=self.__normalization)

        # 3. select the noun phrases up to 3 words as keyphrase candidates.
        extractor.candidate_selection(grammar=grammar,maximum_word_number=3)


        try:
            # 4. weight the candidates using the sum of their word's scores that are
            #    computed using random walk biaised with the position of the words
            #    in the document. In the graph, nodes are words (nouns and
            #    adjectives only) that are connected if they occur in a window of
            #    10 words.
            extractor.candidate_weighting(window=10, pos = pos)

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
        print(f"\n\n-----------------Extract Keyphrases--------------------------")
        listOfDocs = self.LoadDatasetFiles()
        self.runMultipleDocs(listOfDocs)
