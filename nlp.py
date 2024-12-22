import nltk
from tabulate import tabulate
#nltk.download_shell()

messages = [line.rstrip() for line in open('SMSSpamCollection')]

# for mess_no,message in enumerate(messages):
#     print(mess_no,message)
#     print('\n')

import pandas as pd

messages = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])
#print(tabulate(messages))

#print(messages.describe())
#print(messages.groupby('label').describe())

messages['length'] = messages['message'].apply(len)
print(messages.head())

import matplotlib.pyplot as plt
import seaborn as sns

# messages['length'].plot.hist(bins=150) #for one column or feature otherwise for multiple columns
# plt.show()

print(messages['length'].describe())

# print(messages[messages['length']==910]['message'].iloc[0])

messages.hist(column="length",by="label",bins=60,figsize=(12,4))
#plt.show()

import string
from nltk.corpus import stopwords


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words("English")]

list=messages['message'][:2].apply(text_process)
#print(list)

#Vectroisation

from sklearn.feature_extraction.text import CountVectorizer



# bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
# print(len(bow_transformer.vocabulary_))
#mess = messages['message']
#bow = bow_transformer.transform(mess)
# print(bow.shape)
# print(bow.getnnz)



from sklearn.feature_extraction.text import TfidfTransformer


# Tfidf_Transformer = TfidfTransformer().fit(bow)
# tfid = Tfidf_Transformer.transform(bow)
# print(tfid)
# print(Tfidf_Transformer.idf_[bow_transformer.vocabulary_['university']])


from sklearn.naive_bayes import MultinomialNB


# spam_detect_model = MultinomialNB().fit(tfid,messages['label'])
# print(spam_detect_model.predict(tfid))


from sklearn.model_selection import train_test_split

msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)

from sklearn.pipeline import Pipeline

pip = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])

pip.fit(msg_train,label_train)

#predictions=pip.predict(msg_test)

from sklearn.metrics import classification_report

#print(classification_report(predictions,label_test))

custom_messages = [
    "Congratulations! You are a Winner of kbc",    
    "Can we meet tomorrow at 10 am?",   
    "URGENT! You need to claim your prize immediately.",    
]

manual_testing = pip.predict(custom_messages)
for msg,pred in zip(custom_messages,manual_testing):
    print(msg)
    print("\n")
    print(pred)

#print(bow_transformer.get_feature_names_out()[58])


