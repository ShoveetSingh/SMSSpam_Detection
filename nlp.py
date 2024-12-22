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
    nopunc="".join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words("English")]

list=messages['message'][:2].apply(text_process)
print(list)

