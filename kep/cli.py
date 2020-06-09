import argparse
from kep import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pathData', nargs=1, type=str, default=[])
    parser.add_argument('-datasetsName', nargs='*', default=[])
    parser.add_argument('-algorithm', nargs=1, type=str, default=[])
    parser.add_argument('-nKeyphrases', nargs=1, type=int, default=[])
    parser.add_argument('-normalization', nargs=1, type=str, default=[])

    args = parser.parse_args()

    # Get variable number of parameters (specific to each algorithm)
    if args.pathData is not None:
        pathData = args.pathData[0]
    if args.nKeyphrases is not None:
        numOfKeyphrases = args.nKeyphrases[0]
    if args.normalization is not None:
        normalization = args.normalization[0]
    if args.datasetsName is not None:
        listOfDatasets = args.datasetsName
    if args.algorithm is not None:
        algorithm = args.algorithm[0]

    for i in range(len(listOfDatasets)):
    	dataset_name = listOfDatasets[i]

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

if __name__ == '__main__':
	# The entry point for program execution
	main()