import pke
import string
from kep.utility import load_stop_words, getlanguage, CreateKeywordsFolder, LoadFiles
import os

class MultiPartiteRank(object):
    def __init__(self, numOfKeywords, pathData, dataset_name):
        self.__lan = getlanguage(pathData + "/Datasets/" + dataset_name)
        self.__numOfKeywords = numOfKeywords
        self.__dataset_name = dataset_name
        self.__pathData = pathData
        self.__pathToDatasetName = pathData + "/Datasets/" + dataset_name
        self.__keywordsPath = self.__pathData + '/Keyphrases/MultiPartiteRank/' + self.__dataset_name
        self.__algorithmName = "MultiPartiteRank"

    def LoadDatasetFiles(self):
        # Gets all files within the dataset fold
        listFile = LoadFiles(self.__pathToDatasetName + '/docsutf8/*')
        print(f"\ndatasetID = {self.__dataset_name}; Number of Files = {len(listFile)}; Language of the Dataset = {self.__lan}")
        return listFile

    def CreateKeywordsOutputFolder(self):
        # Set the folder where keywords are going to be saved
        CreateKeywordsFolder(self.__keywordsPath)

    def runSingleDoc(self, doc):
        #Get MultiPartiteRank keywords
        # 1. create a MultipartiteRank extractor.
        extractor = pke.unsupervised.MultipartiteRank()

        # 2. load the content of the document in a given language
        # Test if lang exists in spacy models. If not considers model en
        extractor.load_document(input=doc, language=self.__lan)

        # 3. select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'NOUN', 'PROPN', 'ADJ'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += load_stop_words(self.__lan)
        extractor.candidate_selection(pos=pos, stoplist=stoplist)


        try:
            # 4. build the Multipartite graph and rank candidates using random walk,
            #    alpha controls the weight adjustment mechanism, see TopicRank for
            #    threshold/method parameters.
            extractor.candidate_weighting(alpha=1.1,threshold=0.74,method='average')

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