# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:14:21 2020

@author: shiva.sharma
"""

import textract
import os
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

#nltk.download("punkt")
#nltk.download("wordnet")
#nltk.download('stopwords')


parent_directory = "C:\\Users/Shiva.Sharma/Desktop/Resume-Filter/all_resumes/"
ls=[]
for skill in os.listdir(parent_directory):  
        skill_dir=os.path.join(parent_directory, skill)
        #print(skill)
        #print(skill_dir)
        for resume in os.listdir(skill_dir):  
                     
            file=os.path.join(skill_dir, resume)
            #if(os.path.splitext(file)[1] == '.doc'):
            #    os.rename(file, os.path.splitext(file)[0]+'.docx')
            #    print("Changed Extension of File :" +file)
            
            #print(resume)
            text = textract.process(file)
            data=[]
            data.append(os.path.splitext(resume)[0])
            data.append(str(text))
            data.append(skill)
            
            ls.append(data)
            

     
        
#print(ls)

df = pd.DataFrame(ls, columns = ['Resume', 'Content','Skill'])

#print(df['Resume'])

#   #   #   #   #   ##### Cleaning

#Special Char
df['Content_Parsed_1'] = df['Content'].str.replace("\\\\n", " ")
df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\\\\t", " ")

#Lower
df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()

#Punctuations
punctuation_signs = list("?:!.,;")
df['Content_Parsed_3'] = df['Content_Parsed_2']

for punct_sign in punctuation_signs:
    df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')


#Stop Words

stop_words = list(stopwords.words('english'))

df['Content_Parsed_4'] = df['Content_Parsed_3']

for stop_word in stop_words:

    regex_stopword = r"\b" + stop_word + r"\b"
    df['Content_Parsed_4'] = df['Content_Parsed_4'].str.replace(regex_stopword, '')


#lemetize and Stemming
wordnet_lemmatizer = WordNetLemmatizer()
nrows = len(df)
lemmatized_text_list = []

for row in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text = df.loc[row]['Content_Parsed_4']
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)
    
df['Content_Parsed_5'] = lemmatized_text_list

#print(df[['Resume','Skill']])


#   #   #   #   #   ##### Labelling

skill_codes = {
    'Big Data': 0,
    'MSBI': 1,
    'Project Management Office': 2,
    'Tableau': 3
}

df['Skill_Code'] = df['Skill']
df = df.replace({'Skill_Code':skill_codes})

print(df)

#   #   #   #   #   ##### Test Split and TFIDF


X_train, X_test, y_train, y_test = train_test_split(df['Content_Parsed_5'], 
                                                    df['Skill_Code'], 
                                                    test_size=0.25, 
                                                    random_state=10)


tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=(1,2),
                        stop_words=None,
                        lowercase=False,
                        max_df=1.,
                        min_df=10,
                        max_features=300,
                        norm='l2',
                        sublinear_tf=True)

features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(type(features_train))
#print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
#print(features_test.shape)




for skill, skill_code in sorted(skill_codes.items()):
    features_chi2 = chi2(features_train, labels_train == skill_code)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' skill:".format(skill))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")



######RANDOM FOREST
    
#import rdmfrst
#rdmfrst.rf(df,features_train,labels_train,features_test,labels_test)
    
    
    
