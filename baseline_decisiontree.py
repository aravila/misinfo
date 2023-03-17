import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from util import plot_confusion_matrix, print_metrices, cleantext
from sklearn import tree

data_path = './data_covid19_fake_news'
train = pd.read_excel(f'{data_path}/Constraint_English_Train.xlsx')
val = pd.read_excel(f'{data_path}/Constraint_English_Val.xlsx')
test = pd.read_excel(f'{data_path}/Constraint_English_Test.xlsx')

train['tweet'] = train['tweet'].map(lambda x: cleantext(x))
val['tweet'] = val['tweet'].map(lambda x: cleantext(x))
test['tweet'] = test['tweet'].map(lambda x: cleantext(x))

#minor variations in finla results due to randomness
pipeline = Pipeline([
        ('bow', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('c', tree.DecisionTreeClassifier())
    ])
fit = pipeline.fit(train['tweet'],train['label'])
print('Decision Tree')
print ('val:')
pred=pipeline.predict(val['tweet'])

print_metrices(pred,val['label'])
plot_confusion_matrix(confusion_matrix(val['label'],pred),target_names=['fake','real'], model_name='decisiontree', normalize = False, \
                      title = 'Confusion matix of DT on val data')
