#!/usr/bin/env python
# coding: utf-8

# In[115]:


# Byte Pair Encoding


# In[116]:


V = [] #All unique charcters in C


# In[117]:


corpus = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"


# In[118]:


from nltk.tokenize import word_tokenize


# In[119]:


words = [w+'_' for w in word_tokenize(corpus)]
words


# In[120]:


counts = {}

for w in words:
    if w in counts:
        counts[w] +=1
    else:
        counts[w] = 1
counts


# In[121]:


corpus ={}

for word in counts:
    corpus[word]= [c for c in word]
corpus


# In[ ]:





# In[122]:


def make_pairs(v):
    pairs = set()
    for i in v:
        if i !='_':
            for j in v:
                if i != j:
                    pairs.add(i+','+j)
    return pairs  


# In[123]:


def count_pair_freq(all_pairs,corpus):
    pair_freq ={}
    
    for pair in all_pairs:
        for word in corpus:
            char_list = corpus[word]
            for i in range(len(char_list)-1):
                curr_pair = char_list[i]+ ',' + char_list[i+1]
                if curr_pair == pair:
                    if pair in pair_freq:
                        pair_freq[pair]+=counts[word]
                    else:
                        pair_freq[pair]=counts[word]
    return pair_freq
    


# In[124]:


def get_max_pair(pair_freq):
    Tl=''
    Tr=''
    max_freq =0
    for p in pair_freq:
        if pair_freq[p]>max_freq:
            Tl,Tr = p.split(',')
            max_freq = pair_freq[p]
    
    return [Tl,Tr]


# In[125]:


def new_corpus(corpus,Tl,Tr):
    
    for word in corpus:
        new_list = []
        i =0
        while (i < (len(corpus[word]))):
            if corpus[word][i]==Tl and corpus[word][i+1]==Tr:
                new_list.append(Tl+Tr)
                i+=2
            else:
                new_list.append(corpus[word][i])
                i+=1
        corpus[word] = new_list
    return corpus
                


# In[126]:


def get_vocab(corpus):
    V = set()

    for word in corpus:
        for w in corpus[word]:
            V.add(w)
    return V


# In[127]:


V = get_vocab(corpus)
def BPE(corpus,k):
    for i in range(k):
        
        print("corpus: ",corpus)
        
        # get new vocab:
        vocab = get_vocab(corpus)
        print("vocab: ",vocab)
        
        
        # Most frequent pair of adjacent tokens in C
        all_pairs = make_pairs(vocab)
#         print("all_pairs: ",all_pairs)
        
        count_pairs = count_pair_freq(all_pairs,corpus)
        print("count_pairs: ",count_pairs)
        
        Tl,Tr = get_max_pair(count_pairs)
        print("max_pairs: ",Tl," ",Tr,"\n")

        Tnew = Tl + Tr
        V.add(Tnew)


        # Replace each occurrence of tL, tR in C with tNEW
        corpus = new_corpus(corpus,Tl,Tr)
    
    return V

    


# In[128]:


BPE(corpus,9)


# In[ ]:




