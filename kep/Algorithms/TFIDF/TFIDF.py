import pke
import string
import os
from kep.utility import load_stop_words, ComputeDF, getlanguage, CreateKeywordsFolder, LoadFiles

class TFIDF(object):
    def __init__(self, numOfKeywords, pathData, dataset_name, normalization):
        self.__lan = getlanguage(pathData + "/Datasets/" + dataset_name)
        self.__numOfKeywords = numOfKeywords
        self.__dataset_name = dataset_name
        self.__normalization = normalization
        self.__pathData = pathData
        self.__pathToDFFile = self.__pathData + "/Models/dfs/" + self.__dataset_name + '_dfs.gz'
        self.__pathToDatasetName = self.__pathData + "/Datasets/" + self.__dataset_name
        self.__keywordsPath = self.__pathData + '/Keyphrases/TFIDF/' + self.__dataset_name
        self.__algorithmName = "TFIDF"

    def ComputeDocumentFrequency(self):
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
        #Get TFIDF keywords
        # 1. create a TFIDF extractor.
        extractor = pke.unsupervised.TfIdf()

        # 2. load the content of the document in a given language
        # Test if lan exists in spacy models. If not considers model en
        if self.__lan not in ['en', 'pt', 'fr', 'it', 'nl', 'de']:
            extractor.load_document(input=doc, language='en', normalization=self.__normalization)
        else:
            extractor.load_document(input=doc, language=self.__lan, normalization=self.__normalization)

        # 3. select {1-3}-grams not containing punctuation marks as candidates.
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += load_stop_words(self.__lan)

        extractor.candidate_selection(n=3, stoplist=stoplist)

        try:
            # 4. weight the candidates using a `tf` x `idf`
            df = pke.load_document_frequency_file(input_file=self.__pathToDFFile)
            extractor.candidate_weighting(df=df)

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
