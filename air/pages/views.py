from ast import IsNot
from cmath import isnan
from operator import is_not
from django.shortcuts import render

from data.models import Data
import pandas
# Create your views here.
from multiprocessing import context
from tkinter.tix import COLUMN
from django.shortcuts import render

import nltk
import pandas
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize , word_tokenize
import glob
import re
import os
import numpy as np
import sys
Stopwords = set(stopwords.words('english') + stopwords.words('arabic'))
#Stopwords = set(stopwords.words('arabic'))
import math


all_words = []
dict_global = {}


def finding_all_unique_words_and_freq(words):
    words_unique = []
    word_freq = {}
    for word in words:
        if word not in words_unique:
            words_unique.append(word)
    for word in words_unique:
        word_freq[word] = words.count(word)
    return word_freq
def finding_freq_of_word_in_doc(word,words):
    freq = words.count(word)

def remove_special_characters(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = re.sub(regex,'',text)
    return text_returned
 #df = pd.read_csv("e:\en.csv", sep='delimiter')
 #df = pd.read_csv("e:\data.csv", sep='delimiter')
#documents = pandas.read_csv(r'data.csv')
#documents = pandas.read_csv("e:\data.csv", sep='delimiter')

#doc= pandas.read_excel("e:\en.xls")
doc= pandas.read_excel("en.xls")

allwordlist=[]
wordbyquastiondict={}
for ind in doc.index:
    allwordlist.append(doc['quastion'][ind])
    allwordlist.append(doc['answer'][ind])
    wordbyquastiondict[doc['quastion'][ind]]=doc['quastion'][ind]+" " +doc['answer'][ind]

idx = 1
files_with_index = {}
for row in allwordlist:
    fname = row
    text = row
    text = remove_special_characters(text)
    text = re.sub(re.compile('\d'),'',text)
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word for word in words if len(words)>1]
    words = [word.lower() for word in words]
    words = [word for word in words if word not in Stopwords]
    dict_global.update(finding_all_unique_words_and_freq(words))
    files_with_index[idx] = os.path.basename(fname)
    idx = idx + 1

unique_words_all = set(dict_global.keys())

class Node:
    def __init__(self ,docId, freq = None):
        self.freq = freq
        self.doc = docId
        self.nextval = None
    
class SlinkedList:
    def __init__(self ,head = None):
        self.head = head

linked_list_data = {}
for word in unique_words_all:
    linked_list_data[word] = SlinkedList()
    linked_list_data[word].head = Node(1,Node)

word_freq_in_doc = {}
idx = 1
unique_word_foreach_doc={}
for quastion_key in wordbyquastiondict.keys():
    text = wordbyquastiondict[quastion_key]
    text = remove_special_characters(text)
    text = re.sub(re.compile('\d'),'',text)
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word for word in words if len(words)>1]
    words = [word.lower() for word in words]
    words = [word for word in words if word not in Stopwords]
    word_freq_in_doc = finding_all_unique_words_and_freq(words)
    for word in word_freq_in_doc.keys():
        linked_list = linked_list_data[word].head
        while linked_list.nextval is not None:
            linked_list = linked_list.nextval
        linked_list.nextval = Node(idx ,word_freq_in_doc[word])
    w = set(word_freq_in_doc.keys())
    unique_word_foreach_doc[quastion_key]=w
    idx = idx + 1

########################################################################
from contextlib import redirect_stdout
# module to redirect the output to a text file

terms = []
# list to store the terms present in the documents

keys = []
# list to store the names of the documents

vec_Dic = {}
# dictionary to store the name of the document and the boolean vector as list

dicti = {}
# dictionary to store the name of the document and the terms present in it as a
# vector

dummy_list = []
# list for performing some operations and clearing them
#########for vector
term_Freq = {}
idf = {}
weight = {}


def filter(documents, rows, cols):

    for i in range(rows):
      for j in range(cols):
        # traversal through the data frame
        if(j == 0):
            # first column has the name of the document in the csv file
            keys.append(documents.loc[i].iat[j])
        else:
            dummy_list.append(documents.loc[i].iat[j])
            # dummy list to update the terms in the dictionary
            for k in documents.loc[i].iat[j]:
                if k not in terms:
                   # add the terms to the list if it is not present else continue
                   terms.append(k)
    
      copy = dummy_list.copy()
      # copying the dummy list to a different list

      dicti.update({documents.loc[i].iat[0]: copy})
      
      # adding the key value pair to a dictionary

      dummy_list.clear()




def bool_Representation(dicti, rows, cols):

    terms.sort()
    for i in (dicti):
        for j in terms:
            for k in list(dicti[i]):
                if j in list(k):   
                    dummy_list.append(1)
                else:  
                    dummy_list.append(0)
        copy = dummy_list.copy()
        vec_Dic.update({i: copy})
        dummy_list.clear()   


def query_Vector(query):
	'''In this function we represent the query in the form of boolean vector'''

	qvect = []
	# query vector which is returned at the end of the function

	for i in terms:
		# if the word present in the list of terms is also present in the query
		# then append 1 else append 0

		if i in query:
			qvect.append(1)
		else:
			qvect.append(0)

	return qvect
	# return the query vector which is obtained in the boolean form


def prediction(q_Vect):
 dictionary = {}
 listi = []
 count = 0
 term_Len = len(terms)
 #print(vec_Dic)
 for i in vec_Dic:
    for t in range(term_Len):
        if(q_Vect[t] == vec_Dic[i][t]):
            if(q_Vect[t]==1):
               count += 1

    if count == 1 :    
       dictionary.update({i: count})
    
    count = 0
   
 for i in dictionary:
      listi.append(dictionary[i])
 listi = sorted(listi, reverse=True)

 ans = ' '

 result=[]

 with open('output.txt', 'w') as f:
    with redirect_stdout(f):
        print("ranking of the documents")
        for count, i in enumerate(listi):
            key = check(dictionary, i)
            if count == 0:
                ans = key
            print(key, "rank is", count+1)
            result.append(key)
            dictionary.pop(key)
        print(ans, "is the most relevant document for the given query")

 return result

	
def check(dictionary, val):
	'''Function to return the key when the value is known'''

	for key, value in dictionary.items():
		if(val == value):
			# if the given value is same as the value present in the dictionary
			# return the key

			return key


def main():
	documents = pandas.read_csv(r'documents.csv')
	# to read the data from the csv file as a dataframe

	rows = len(documents)
	# to get the number of rows

	cols = len(documents.columns)
	# to get the number of columns

	filter(documents, rows, cols)
	# function call to read and separate the name of the documents and the terms
	# present in it to a separate list from the data frame and also create a
	# dictionary which has the name of the document as key and the terms present in
	# it as the list of strings which is the value of the key

	bool_Representation(dicti, rows, cols)
	# In this function we get a boolean representation of the terms present in the
	# documents in the form of lists, later we create a dictionary which contains
	# the name of the documents as key and value as the list of boolean values
	#representing the terms present in the document

	print("Enter query")
	query = input()
	# to get the query input from the user, the below input is given for obtaining
	# the output as in output.txt file
	# hockey is a national sport

	query = query.split(' ')
	# spliting the query as a list of strings

	q_Vect = query_Vector(query)
	# function call to represent the query in the form of boolean vector

	prediction(q_Vect)
	# Function call to make the prediction regarding which document is related to
	# the given query by performing the boolean operations



   #############################################################################vector
  

def compute_Weight(doc_Count, cols):
    for i in terms:
        if i not in term_Freq:
            term_Freq.update({i: 0})
           

    for key, value in dicti.items():
        for k in list(value):
            for j in list(k):
               if j in term_Freq:
                 term_Freq[j] += 1
    idf = term_Freq.copy()

    for i in term_Freq:
        term_Freq[i] = term_Freq[i]/cols

    for i in idf:
        if idf[i] != doc_Count:
            idf[i] = math.log2(cols / idf[i])
        else:
            idf[i] = 0

    for i in idf:
        weight.update({i: idf[i]*term_Freq[i]})

    dummy_List = []
    for i in dicti:
        for j in dicti[i]:
            for k in list(j):
              dummy_List.append(weight[k])
 
        copy = dummy_List.copy()
        vec_Dic.update({i: copy})
        dummy_List.clear()


def get_Weight_For_Query(query):
     query_Freq = {}
     for i in terms:
        if i not in query_Freq:
            query_Freq.update({i: 0})
     for val in query:
        if val in query_Freq:
            query_Freq[val] += 1

     for i in query_Freq:
        query_Freq[i] = query_Freq[i] / len(query)
     return query_Freq


def similarity_Computation(query_Weight):
      numerator = 0
      denomi1 = 0
      denomi2 = 0

      similarity = {}
      for document in dicti:
        for k in list(dicti[document]):
          for terms in list(k):
            numerator += weight[terms] * query_Weight[terms]
            denomi1 += weight[terms] * weight[terms]
            denomi2 += query_Weight[terms] * query_Weight[terms]
          if denomi1 != 0 and denomi2 != 0:
            simi = numerator / (math.sqrt(denomi1) * math.sqrt(denomi2))
            similarity.update({document: simi})

            numerator = 0   
            denomi2 = 0
            denomi1 = 0
      return (similarity)
            


def prediction1(similarity, doc_count):
    result=[]
    if similarity:
      with open('output.txt', 'w') as f:
         with redirect_stdout(f):
            ans = max(similarity, key=similarity.get)
            print(ans, "is the most relevant document")
            print("ranking of the documents")
            for i in range(doc_count):
                  ans = max(similarity, key=lambda x: similarity[x],default=0)
                  print(ans, "rank is", i+1)
                  result.append(ans)

                  similarity.pop(ans)
    return result

##########################################################  <input type="text" id="myInput" onkeyup="myFunction()" placeholder="Search" title="Type in a name">
 

def index(request):
    print('iiiiiiiiiiiiiii')
    return render(request, 'pages/index.html', {'name':'ahmad'})

def search_and(data,result,words):
    ch=1
    for item in data:
        for word in words:
           if word in item.quastion or word in item.answer:
              #data1.append(item.answer,item.quastion)
             c=1
           else:
              ch=0
        if ch == 1:
            result.append(item.quastion) 
        ch=1
    return result



def search_or(data,result,words):
    for item in data:
        for word in words:
           if word in item.quastion or word in item.answer:
              result.append(item.quastion) 

    return result



def search_bol(data ,query):
    result=[]
    str_or=query.split('|')
    if str_or:
       for i in str_or:
         str_and=i.split('&')
         if str_and:
            result=search_and(data,result,str_and)

       result=search_or(data,result,str_or)
     
    elif query.split('&'):
        str_and=query.split('&')
        result=search_and(data,result,str_and)
    else:
        result=search_or(data,result,query)       
       
    return result



def search(request):
    data = Data.objects.all()
    search=''
    df=pandas.DataFrame(unique_word_foreach_doc.items(), columns=['quastion', 'answer'])

    #print(type(data))
    query=''

    result=[]
    if request.GET.get('Vector') != None:
        rows = len(df)
        cols = len(df.columns)
        filter(df, rows, cols)
        compute_Weight(rows, cols)
        query1 = query=request.GET.get('Vector')
        query = query=request.GET.get('Vector')
        query = query.split(' ')
        query_Weight = get_Weight_For_Query(query)

        similarity = similarity_Computation(query_Weight)

        row=len(similarity)
        result = prediction1(similarity, row)

        # data= search_or(data,'dubai','dd')
        #print(request.GET.get('Vector'))


    if request.GET.get('Extended') != None:
           
            rows = len(df)
            cols = len(df.columns)
            filter(df, rows, cols)
            bool_Representation(dicti, rows, cols)
            query=request.GET.get('Extended')
            query1 = query=request.GET.get('Extended')
            query = query.split(' ')
            q_Vect = query_Vector(query)
            result=prediction(q_Vect)
            print(result)

    if request.GET.get('Boolean') != None:
        result=[]
        print('booool')
        query=request.GET.get('Boolean')
        query1 = query=request.GET.get('Boolean')

        result=search_bol(data,query)
        print(result)

  

           

    #data=data.to_string()
   
   
    result = list(set(result))
    print(type(data))
    return render(request, 'pages/data.html', {'data':data,'result':result,'query':query1})
    #return render(request, 'pages/search.html', {'name':'ahmad'})


def data(request): 
    #if request.GET.get('Vector') != None:
    print(request.GET.get('Vector'))
    print('------------------')

    data = Data.objects.all()
    return render(request , 'pages/data.html',{'data': data })


