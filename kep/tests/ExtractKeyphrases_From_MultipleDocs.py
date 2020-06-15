from kep import Rake
from kep import YAKE
from kep import MultiPartiteRank
from kep import TopicalPageRank
from kep import TopicRank
from kep import PositionRank
from kep import SingleRank
from kep import TextRank
from kep import KPMiner
from kep import TFIDF
from kep import KEA

# Some algorithms have a normalization parameter which may be defined with None, stemming or lemmatization
normalization = None  # Other options: "stemming" (porter) and "lemmatization"

# Num of Keyphrases do Retrieve
numOfKeyphrases = 20

# List of Datasets

#Uncomment if you want to extract keyphrases for all the datasets
'''
ListOfDatasets = ['110-PT-BN-KP', '500N-KPCrowd-v1.1', 'citeulike180',
                  'fao30', 'fao780', 'Inspec', 'kdd', 'Krapivin2009',
                  'Nguyen2007', 'pak2018', 'PubMed', 'Schutz2008', 'SemEval2010',
                  'SemEval2017', 'theses100', 'wiki20', 'www', 'cacic', 'wicc', 'WikiNews']
'''


ListOfDatasets = ['500N-KPCrowd-v1.1']

#Uncomment if you want to extract keyphrases for all the algorithms

'''
ListOfAlgorithms = ['RAKE', 'YAKE', 'MultiPartiteRank', 'TopicalPageRank', 'TopicRank', 'PositionRank', 'SingleRank', 'TextRank',
                    'KPMiner', 'TFIDF', 'KEA']
'''

ListOfAlgorithms = ['YAKE']

#Please replace the following path by your own path, where data is the folder where the datasets may be found.
pathData = 'SPECIFY PATH FOR DATA FOLDER'

for algorithm in ListOfAlgorithms:
    print("\n")
    print("----------------------------------------------------------------------------------------")
    print(f"Algorithm being executed = \033[1m{algorithm}\033[0m")

    for i in range(len(ListOfDatasets)):
        dataset_name = ListOfDatasets[i]
        print("\n----------------------------------")
        print(f" dataset_name = {dataset_name}")
        print("----------------------------------")

        if algorithm == 'RAKE':
            Rake_object = Rake(numOfKeyphrases, pathData, dataset_name)
            listOfDocs = Rake_object.LoadDatasetFiles()
            Rake_object.runMultipleDocs(listOfDocs)
        elif algorithm == 'YAKE':
            YAKE_object = YAKE(numOfKeyphrases, pathData, dataset_name)
            listOfDocs = YAKE_object.LoadDatasetFiles()
            YAKE_object.runMultipleDocs(listOfDocs)
        elif algorithm == 'MultiPartiteRank':
            MultiPartiteRank_object = MultiPartiteRank(numOfKeyphrases, pathData, dataset_name)
            listOfDocs = MultiPartiteRank_object.LoadDatasetFiles()
            MultiPartiteRank_object.runMultipleDocs(listOfDocs)
        elif algorithm == 'TopicalPageRank':
            TopicalPageRank_object = TopicalPageRank(numOfKeyphrases, pathData, dataset_name, normalization)
            TopicalPageRank_object.CreateLDAModel()
            listOfDocs = TopicalPageRank_object.LoadDatasetFiles()
            TopicalPageRank_object.runMultipleDocs(listOfDocs)
        elif algorithm == 'TopicRank':
            TopicRank_object = TopicRank(numOfKeyphrases, pathData, dataset_name)
            listOfDocs = TopicRank_object.LoadDatasetFiles()
            TopicRank_object.runMultipleDocs(listOfDocs)
        elif algorithm == 'PositionRank':
            PositionRank_object = PositionRank(numOfKeyphrases, pathData, dataset_name, normalization)
            listOfDocs = PositionRank_object.LoadDatasetFiles()
            PositionRank_object.runMultipleDocs(listOfDocs)
        elif algorithm == 'SingleRank':
            SingleRank_object = SingleRank(numOfKeyphrases, pathData, dataset_name, normalization)
            listOfDocs = SingleRank_object.LoadDatasetFiles()
            SingleRank_object.runMultipleDocs(listOfDocs)
        elif algorithm == 'TextRank':
            TextRank_object = TextRank(numOfKeyphrases, pathData, dataset_name, normalization)
            listOfDocs = TextRank_object.LoadDatasetFiles()
            TextRank_object.runMultipleDocs(listOfDocs)
        elif algorithm == 'KPMiner':
            KPMiner_object = KPMiner(numOfKeyphrases, pathData, dataset_name, normalization)
            KPMiner_object.ComputeDocumentFrequency()
            listOfDocs = KPMiner_object.LoadDatasetFiles()
            KPMiner_object.runMultipleDocs(listOfDocs)
        elif algorithm == 'TFIDF':
            TFIDF_object = TFIDF(numOfKeyphrases, pathData, dataset_name, normalization)
            TFIDF_object.ComputeDocumentFrequency()
            listOfDocs = TFIDF_object.LoadDatasetFiles()
            TFIDF_object.runMultipleDocs(listOfDocs)
        elif algorithm == 'KEA':
            KEA_object = KEA(numOfKeyphrases, pathData, dataset_name, normalization)
            KEA_object.TrainingModel()
            listOfDocs = KEA_object.LoadDatasetFiles()
            KEA_object.runMultipleDocs(listOfDocs)


