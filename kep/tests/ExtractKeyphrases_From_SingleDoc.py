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

# Num of Keywords to Retrieve
numOfKeyphrases = 20

# Some algorithms need to know the path of the dataset in order to create some models
dataset_name = '500N-KPCrowd-v1.1'
doc = 'art_and_culture-20893614.txt'

#dataset_name = 'cacic'
#doc = '18572.txt'

#dataset_name = 'pak2018'
#doc = 'pak_0.txt'

#dataset_name = 'WikiNews'
#doc = '44543.txt'

#dataset_name = '110-PT-BN-KP'
#doc = '2000_10_09-13_00_00-JornaldaTarde-8-topic-seg.txt-Nr1.txt'

#Please replace the following path by your own path, where data is the folder where the datasets may be found.
pathData = 'H:/Backup/Research/Programming/JupyterNotebooks/Python/APIs_Packages/kep/data'

docPath = pathData + f'/Datasets/{dataset_name}/docsutf8/{doc}'

# For training KEA
nFolds = 5 #Num of folds

#ListOfAlgorithms = ['RAKE', 'YAKE', 'MultiPartiteRank', 'TopicalPageRank', 'TopicRank', 'SingleRank', 'TextRank', 'KPMiner', 'TFIDF', 'KEA']
ListOfAlgorithms = ['YAKE']

for algorithm in ListOfAlgorithms:
    print(f"Algorithm being executed = {algorithm}")

    if algorithm == 'RAKE':
        rake_object = Rake(numOfKeyphrases, pathData, dataset_name)
        sample_file = open(docPath, 'r', encoding="utf-8")
        text = sample_file.read()
        keyphrases = rake_object.runSingleDoc(text)
    elif algorithm == 'YAKE':
        yake_object = YAKE(numOfKeyphrases, pathData, dataset_name)
        sample_file = open(docPath, 'r', encoding="utf-8")
        text = sample_file.read()
        keyphrases = yake_object.runSingleDoc(text)
    elif algorithm == 'MultiPartiteRank':
        multiPartiteRank_object = MultiPartiteRank(numOfKeyphrases, pathData, dataset_name)
        keyphrases = multiPartiteRank_object.runSingleDoc(docPath)
    elif algorithm == 'TopicalPageRank':
        TopicalPageRank_object = TopicalPageRank(numOfKeyphrases, pathData, dataset_name, normalization)
        TopicalPageRank_object.CreateLDAModel()
        keyphrases = TopicalPageRank_object.runSingleDoc(docPath)
    elif algorithm == 'TopicRank':
        TopicRank_object = TopicRank(numOfKeyphrases, pathData, dataset_name)
        keyphrases = TopicRank_object.runSingleDoc(docPath)
    elif algorithm == 'PositionRank':
        PositionRank_object = PositionRank(numOfKeyphrases, pathData, dataset_name, normalization)
        keyphrases = PositionRank_object.runSingleDoc(docPath)
    elif algorithm == 'SingleRank':
        SingleRank_object = SingleRank(numOfKeyphrases, pathData, dataset_name, normalization)
        keyphrases = SingleRank_object.runSingleDoc(docPath)
    elif algorithm == 'TextRank':
        TextRank_object = TextRank(numOfKeyphrases, pathData, dataset_name, normalization)
        keyphrases = TextRank_object.runSingleDoc(docPath)
    elif algorithm == 'KPMiner':
        KPMiner_object = KPMiner(numOfKeyphrases, pathData, dataset_name, normalization)
        KPMiner_object.ComputeDocumentFrequency()
        keyphrases = KPMiner_object.runSingleDoc(docPath)
    elif algorithm == 'TFIDF':
        TFIDF_object = TFIDF(numOfKeyphrases, pathData, dataset_name, normalization)
        TFIDF_object.ComputeDocumentFrequency()
        keyphrases = TFIDF_object.runSingleDoc(docPath)
    elif algorithm == 'KEA':
        KEA_object = KEA(numOfKeyphrases, pathData, dataset_name, normalization)
        KEA_object.TrainingModel()  # Traing KEA Model on top of the entire labeled dataset.
        keyphrases = KEA_object.runSingleDoc(docPath)

    print(f"\n\nKeyphrases:{keyphrases}\n")
