import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import nltk
if __name__=='__main__'
 train = pd.read_csv(os.path.join(os.path.dirname(),'data','labeledTrainData.tsv'),header=0,\ delimiter="\t",quoting=3)
 test = pd.read_csv(os.path.join(os.path.dirname(),'data',testData.tsv'),header=0,delimiter="\t",\ quoting=3)
print 'The Frirst review is:'
print train["review"][0]
raw_input("press enter to continue..")
#cleaning data 
print 'Download text data'
nltk.download()
clean_train_reviews = []
print 'Cleaning and parsing the traing set movie.....\n"
 for i in xrange(0,len(train["review"])):
   clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_worldlist(train["review"][i],True)
print "creating the bag of words..\n"
vectorizer = CountVectorizer(analyzer = "word", \ tokenizer = None, \ preprosser = None, \ stop_words = None, \ max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# train classifier
print "traing the random forest ...."
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( train_data_features , train["sentiment"])
clean_test_reviews =[]

# fromat the testing data 
print "cleaning and parsing the test set movie reviews ...."
for i in xrange(0,len(test["review"])):
  clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i],True))
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# predict reviews
print "predicting test lables ..."
result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"],"sentiment":result} )
output.to_csv(os.path.join(os.path.dirname(__file__), 'data' , 'Bag_of_words_model.csv'), index = False , qouting = 3 )
print "wrote result to bag_of_words_model.csv"

