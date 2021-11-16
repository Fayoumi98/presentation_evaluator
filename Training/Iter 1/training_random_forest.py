# pandas imported for csv file manipulation
import pandas as pd
# sklearn train_test_split
from sklearn.model_selection import train_test_split
# Construct a Pipeline from the given estimators
from sklearn.pipeline import make_pipeline 
# Normalise the data so no feature overshadows the other features Standardize features by removing the mean and scaling to unit varianceby subtracting the mean and devide it by the standard deviation
from sklearn.preprocessing import StandardScaler 
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset
from sklearn.ensemble import RandomForestClassifier
# Accuracy metrics 
from sklearn.metrics import accuracy_score
# to save the model for further usage
import pickle





######################## Data Conditioning ########################


# import csv file of the data (put the data file in the same folder)
df = pd.read_csv('data_happy_nervous.csv')
#print(df.head())        # display first five raws
#print(def.tail())       # display last five raws





# remove the class column of "Nervous"
X = df.drop('Nervous', axis=1)          # features
# create a class column with the same length as the feature length
y = df['Nervous']                       # target value
#print(len(y),len(X))






# split the data => to training set and test set (test = 30% / training = 70%)
# random_state => Controls the shuffling applied to the data before applying the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
#print(len(y_test))






######################## Create Training pipeline ########################



# step 1 : Standard Scaler => Random Forest Classifier          ### rf


pipelines = {
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
}



# dictionary key 'rf'
# print(pipelines.keys())
# list each function in the pipeline sequencialy
# print(pipelines.items())







########################## Start Training ##############################

# create a blank dictionanry for the model
fit_models = {}

# loop inside each pipeline with the trainning data
for algorithm, pipeline in pipelines.items():
    # algorithm : algorithm dictionary key 'rf'
    # pipeline : => Random Forest Classifier

    model = pipeline.fit(X_train, y_train)
    fit_models[algorithm] = model
    
    
    
#print('done')
#print(fit_models)



# print test set predictions
#print(fit_models['rf'].predict(X_test))


for algorithm, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algorithm, accuracy_score(y_test, yhat))
    

with open('body_language_nervous_happy.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)



















