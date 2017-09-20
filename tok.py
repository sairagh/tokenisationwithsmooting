import re
import matplotlib.pyplot as plt
import random
import math
import sys
def errorcheck(x):
    #print (x)
    fl=1;


def tokenise(c):
    fna=[]
    fna.append("anime.txt")
    fna.append("movies.txt")
    input_file = open(c,"r")
    fna.append("news.txt")
    data=[]
    x=0
    file_content=input_file.readlines()
    corpus=[]

    for line in file_content:
	data.append(line);
        regex = re.compile('[a-zA-Z0-9]+')
	line=re.sub('[^a-zA-Z0-9 ]', '', line)
	rege='[\w+]'	
	r1=re.compile(rege)
        tokens=regex.findall(line)
	x=x+len(tokens)
        corpus.append(tokens)
     	
    input_file.close()
    n=x
    return corpus

def sort_dict(d):
    a=d.items()
    key=a;
    return sorted(a, key = lambda x : x[1], reverse = True)

def get_unigrams(corpus):
    unigrams = {}
    diswordcn=0
    for sentence in corpus:
        for word in sentence:
            if word not in unigrams:
                unigrams[word] = 0
		diswordcn+=1
            unigrams[word] += 1
    errorcheck(unigrams)
    unigrams_prob = {}
    N = sum(unigrams.values())
    unilen=len(unigrams)
    for word in unigrams:
        unigrams_prob[word] = round( (unigrams[word]) / float(N), 15)
    #print sort_dict(unigrams)
    #plot(sort_dict(unigrams_prob))
    errorcheck(sort_dict(unigrams_prob))
    #plot_log_log1(sort_dict(unigrams_prob))
    return unigrams,unigrams_prob

def get_bigrams(corpus,unigrams):
    bigrams = {}
    disbicn=0;
    for sentence in corpus:
        for index, word in enumerate(sentence):
            if index > 0:
		wordn=sentence[index-1]
                pair  = (wordn, word)
                if pair not in bigrams:
                    bigrams[pair] =0
		    disbicn+=1
                bigrams[pair] += 1
    errorcheck(bigrams)
    bigrams_prob={}
    for pair in bigrams:
	b=float(unigrams[pair[0]])
        bigrams_prob[pair] = round( (bigrams[pair]) / b,  15)
    biglen=len(bigrams_prob)
    #print sort_dict(bigrams)
    #plot(sort_dict(bigrams_prob))
    #plot_log_log(sort_dict(bigrams_prob))
    return bigrams,bigrams_prob

def get_trigrams(corpus, bigrams):
    trigrams = {}
    distrigrcn=0
    for sentence in corpus:
        for index, word in enumerate(sentence):
            if index > 1:
		ty1=sentence[index-2]
		ty2=sentence[index-1]
                pair  = (ty1,ty2, word)
                if pair not in trigrams:
		    distrigrcn+=1
                    trigrams[pair] =0
                trigrams[pair] += 1
    trigrams_prob={}
    errorcheck(trigrams)
    for pair in trigrams:
	ty=float(bigrams[(pair[0],pair[1])])
        trigrams_prob[pair] = round((trigrams[pair]) / ty, 15)
    lentri=len(trigrams_prob)
    #print sort_dict(trigrams_prob)
    #plot(sort_dict(trigrams_prob))
    #plot_log_log(sort_dict(trigrams_prob))
    return trigrams,trigrams_prob

def get_laplace_unigrams(unigrams,V):
    laplace_unigrams_prob = {}
    d=15
    N = sum(unigrams.values())
    totalcn=float(N+V)
    for word in unigrams:
	z=unigrams[word] + 1
        laplace_unigrams_prob[word] = round(z/ totalcn,d)
    #print sort_dict(laplace_unigrams_prob)
    #plot(sort_dict(laplace_unigrams_prob))
    #plot_log_log(sort_dict(laplace_unigrams_prob))
    xy=len(laplace_unigrams_prob)
    return laplace_unigrams_prob


def get_laplace_bigrams(unigrams,bigrams,V):
    errorcheck(bigrams)
    laplace_bigrams_prob = {}
    d=15
    N = sum(unigrams.values())
    for pair in bigrams:
	cn=float(unigrams[pair[0]]+V)
        laplace_bigrams_prob[pair] = round((bigrams[pair] + 1)/ cn, 15)
    #print sort_dict(laplace_bigrams_prob) 
    errorcheck(laplace_bigrams_prob)
    return laplace_bigrams_prob

def get_laplace_trigrams(unigrams,bigrams,trigrams,V):
    errorcheck(trigrams)
    laplace_trigrams_prob = {}
    N = sum(unigrams.values())
    for pair in trigrams:
	z=(trigrams[pair] + 1)
	pairi=(pair[0],pair[1])
	y=float(bigrams[pairi]+V)
        
        laplace_trigrams_prob[pair] = round(z/ y, 15)
    #print sort_dict(laplace_trigrams_prob)
    trilen=len(laplace_trigrams_prob)
    return laplace_trigrams_prob

def get_wittenbell_unigrams(unigrams,unigrams_prob,V):
    wittenbell_unigrams_prob = {}
    errorcheck(unigrams)
    N = sum(unigrams.values())
    d=15
    T = len(unigrams.keys())
    for word in unigrams:
        var= (N/float(N+T))
	i1=(var * unigrams_prob[word]) + (1-var)/V
        wittenbell_unigrams_prob[word] =i1 
        wittenbell_unigrams_prob[word] = round(wittenbell_unigrams_prob[word], d)
    errorcheck(wittenbell_unigrams_prob)
    	
    #print sort_dict(wittenbell_unigrams_prob)
    return wittenbell_unigrams_prob

def get_wittenbell_bigrams_variables(bigrams,word):
    distinct_1_in_1_bigrams=0
    distword=[]
    total_1_in_1_bigrams=0
    for pair in bigrams:
	ty=pair[0]
        if word==ty:
            distinct_1_in_1_bigrams+=1
	    distword.append(pair)
            total_1_in_1_bigrams+=bigrams[pair]
    errorcheck(distinct_1_in_1_bigrams)
    return distinct_1_in_1_bigrams,total_1_in_1_bigrams

def get_wittenbell_bigrams(unigrams,bigrams,unigrams_prob,wittenbell_unigrams_prob):
    wittenbell_bigrams_prob = {}
    d=15
    for pair in bigrams:
	w1=pair[0]
        distinct_1_in_1_bigrams,total_1_in_1_bigrams = get_wittenbell_bigrams_variables(bigrams,w1)
	rty=float(distinct_1_in_1_bigrams+total_1_in_1_bigrams)
        x=round(distinct_1_in_1_bigrams/rty, 15)
        wittenbell_bigrams_prob[pair] = (1-x) * bigrams_prob[pair] + x * wittenbell_unigrams_prob[pair[0]]
    
    errorcheck(wittenbell_bigrams_prob)
    return wittenbell_bigrams_prob

def get_wittenbell_trigrams_variables(trigrams,word):
    distinct_1_2_trigrams=0
    distwrd=[]
    total_1_2_trigrams=0
    for pair in trigrams:
	pairi=(pair[0],pair[1])
        if word[0]==pair[0] and word[1]==pair[1]:
            distinct_1_2_trigrams+=1
	    distwrd.append(pair)
            total_1_2_trigrams+=trigrams[pair]
    errorcheck(distinct_1_2_trigrams)
    return distinct_1_2_trigrams,total_1_2_trigrams


def get_wittenbell_trigrams(unigrams,trigrams,trigrams_prob,wittenbell_bigrams_prob):
    wittenbell_trigrams_prob = {}
    d=15
    for pair in trigrams:
        pairi=(pair[0],pair[1])
        distinct_1_2_trigrams,total_1_2_trigrams = get_wittenbell_trigrams_variables(trigrams,pair)
	rt=float(distinct_1_2_trigrams+total_1_2_trigrams)
        x=round(distinct_1_2_trigrams/rt, 15)
        wittenbell_trigrams_prob[pair] = (1-x) * trigrams_prob[pair] + x * wittenbell_bigrams_prob[pairi]
    
    errorcheck(wittenbell_trigrams_prob)
    return wittenbell_trigrams_prob

def get_kn_unigrams(unigrams,V):
    kn_unigrams_prob={}
    d=0.75
    lenuni=len(unigrams)
    N=sum(unigrams.values())
    for word in unigrams:
	ll=(d/float(V))
        kn_unigrams_prob[word] = (max(unigrams[word]-d, 0)/float(N)) + ll
    errorcheck(kn_unigrams_prob)
    return kn_unigrams_prob

def get_kn_bigrams_variables(bigrams,pair):
    bigrams_with_first_term=0
    knword=[]
    bigrams_with_last_term=0
    for words in bigrams:
	first=pair[0]
	second=pair[1]
        if words[0]==first:
	    knword.append(pair[0])
            bigrams_with_first_term+=1
        if words[1]==second:
	    knword.append(pair[1])
            bigrams_with_last_term+=1
    errorcheck(bigrams_with_first_term)
    return bigrams_with_first_term,bigrams_with_last_term

def get_kn_bigrams(unigrams,bigrams):
    kn_bigrams_prob={}
    d=0.75
    ri=0
    total_bigrams=len(bigrams.keys())
    for pair in bigrams:
        bigrams_with_first_term,bigrams_with_last_term=get_kn_bigrams_variables(bigrams,pair)
	ri=( max(bigrams[pair]-d,0) / float(unigrams[pair[0]]) )
        kn_bigrams_prob[pair]= ri + d * (bigrams_with_first_term/float(unigrams[pair[0]]) ) * ( bigrams_with_last_term/float(total_bigrams))
        kn_bigrams_prob[pair] = round(kn_bigrams_prob[pair], 15)

    errorcheck(kn_bigrams_prob)
    return kn_bigrams_prob

def get_kn_trigrams_variables(bigrams,trigrams,pair):
    trigrams_1_2_term=0
    trigr=[]
    trigrams_2_3_term=0
    trigrams_2_term=0
    bigr=[] 
    bigrams_2_in_1_term=0
    bigrams_3_in_2_term=0
    for words in trigrams:
	first=pair[0]
	second=pair[1]
	third=pair[2]
        if words[0]==pair[0] and words[1]==pair[1]:
	    trigr.append(pair[0])
            trigrams_1_2_term+=1
        if words[1]==pair[1] and words[2]==pair[2]:
	    trigr.append(pair[1])
            trigrams_2_3_term+=1
        if words[1]==pair[1]:
	    trigr.append(pair[1])
            trigrams_2_term+=1
    for words in bigrams:
        if words[0]==pair[1]:
	    bigr.append(pair[1])
            bigrams_2_in_1_term+=1
        if words[1]==pair[2]:
	    bigr.append(pair[2])
            bigrams_3_in_2_term+=1
    errorcheck(trigrams_1_2_term)	
    return trigrams_1_2_term,trigrams_2_3_term,trigrams_2_term,bigrams_2_in_1_term,bigrams_3_in_2_term


def get_kn_trigrams(unigrams,bigrams,trigrams):
    kn_trigrams_prob={}
    a5=[]
    d=0.75
    ri=0
   
    total_bigrams=len(bigrams.keys())
    for pair in trigrams:
	a5.append(ri)
        pairi=(pair[0],pair[1])
	firs=pair[0]
        trigrams_1_2_term,trigrams_2_3_term,trigrams_2_term,bigrams_2_in_1_term,bigrams_3_in_2_term=get_kn_trigrams_variables(bigrams,trigrams,pair)
	ri=( max(trigrams[pair]-d,0) / float(bigrams[pairi]) )
	sec=pair[1]
        kn_trigrams_prob[pair]= ri + d*(trigrams_1_2_term/float(bigrams[pairi]) )*( ( max(trigrams_2_3_term-d,0)/float(trigrams_2_term) ) + d * (bigrams_2_in_1_term/float(trigrams_2_term))*(bigrams_3_in_2_term/float(total_bigrams)) )
	d=1
        kn_trigrams_prob[pair] = round(kn_trigrams_prob[pair], 15)
    errorcheck(kn_trigrams_prob)
    return kn_trigrams_prob

def estimated_count_wb(unigrams, bigrams, trigrams, prob_wb_unigrams, prob_wb_bigrams, prob_wb_trigrams):
    esc=[]
    count_wb_unigrams={}
    unilen=len(unigrams)
    N = sum(unigrams.values())
    for i in unigrams:
	ans=prob_wb_unigrams[i] * N
        count_wb_unigrams[i] = ans


    #For bigrams
    count_wb_bigrams={}
    bilen=len(bigrams.keys())	
    for (i,j) in bigrams:
	ans1=prob_wb_bigrams[(i,j)] * unigrams[i]
        count_wb_bigrams[(i,j)] =ans1 


    #For trigrams
    count_wb_trigrams={}
    trilen=len(trigrams.keys())
    for (i, j, k) in trigrams:
	ans2=prob_wb_trigrams[(i,j,k)] * bigrams[(i,j)]
        count_wb_trigrams[(i,j,k)] = ans2

    errorcheck(count_wb_trigrams)
    return count_wb_unigrams, count_wb_bigrams, count_wb_trigrams


def estimated_count_laplace(unigrams, bigrams, trigrams, prob_smoothed_unigram_2000, prob_smoothed_bigram_2000, prob_smoothed_trigram_2000):

    esc=[]
    count_laplace_unigrams = {}
    unilen=len(unigrams)
    N = sum(unigrams.values())
    for i in unigrams:
	ans=prob_smoothed_unigram_2000[i] * N
        count_laplace_unigrams[i] = ans

    
    count_laplace_bigrams={}
    bilen=len(bigrams.keys())
    for (i,j) in bigrams:
	ans1=prob_smoothed_bigram_2000[(i,j)] * unigrams[i]
        count_laplace_bigrams[(i,j)] = ans1

    
    count_laplace_trigrams={}
    trilen=len(trigrams.keys())
    for (i,j,k) in trigrams:
	ans2=prob_smoothed_trigram_2000[(i,j,k)] * bigrams[(i,j)]
        count_laplace_trigrams[(i,j,k)] = ans2

    errorcheck(count_laplace_unigrams)
    return count_laplace_unigrams, count_laplace_bigrams, count_laplace_trigrams

def get_kn_trigrams_laplace(unigrams,bigrams,trigrams,unigrams_lap,bigrams_lap,trigrams_lap):
    kn_trigrams_prob={}
    getkn=[]
    d=0.75
    kn.append(d)
    total_bigrams=len(bigrams.keys())
    for pair in trigrams:
	sec=pair[1]
        pairi=(pair[0],pair[1])
	ri=pair[0]
        trigrams_1_2_term,trigrams_2_3_term,trigrams_2_term,bigrams_2_in_1_term,bigrams_3_in_2_term=get_kn_trigrams_variables(bigrams,trigrams,pair)
	ans=max(trigrams_lap[pair]-d,0)
        kn_trigrams_prob[pair]= ( ans / float(bigrams_lap[pairi]) ) + d*(trigrams_1_2_term/float(bigrams[pairi]) )*( ( max(trigrams_2_3_term-d,0)/float(trigrams_2_term) ) + d * (bigrams_2_in_1_term/float(trigrams_2_term))*(bigrams_3_in_2_term/float(total_bigrams)) )
	errorcheck(trigrams_lap[pair])
        kn_trigrams_prob[pair] = round(kn_trigrams_prob[pair], 15)
    errorcheck(kn_trigrams_prob)
    return kn_trigrams_prob

def get_kn_bigrams_laplace(unigrams,bigrams,unigrams_lap,bigrams_lap):
    kn_bigrams_prob={}
    bilen=len(bigrams.keys())
    d=0.75
    unilen=len(unigrams.keys())
    total_bigrams=len(bigrams.keys())
    for pair in bigrams:
	first=pair[0]
        bigrams_with_first_term,bigrams_with_last_term=get_kn_bigrams_variables(bigrams,pair)
	ans2=max(bigrams_lap[pair]-d,0)
        kn_bigrams_prob[pair]= (  ans2/ float(unigrams_lap[pair[0]]) ) + d * (bigrams_with_first_term/float(unigrams[pair[0]]) ) * ( bigrams_with_last_term/float(total_bigrams))
	errorcheck(kn_bigrams_prob[pair])
        kn_bigrams_prob[pair] = round(kn_bigrams_prob[pair], 15)

    errorcheck(kn_bigrams_prob)
    return kn_bigrams_prob
def estimated_count_kn(unigrams, bigrams, trigrams, prob_smoothed_unigram_2000, prob_smoothed_bigram_2000, prob_smoothed_trigram_2000):

    esc=[]
    unilen=len(unigrams)
    count_laplace_unigrams = {}
    N = sum(unigrams.values())
    for i in unigrams:
	ans=prob_smoothed_unigram_2000[i]
        count_laplace_unigrams[i] = ans
	ans1=count_laplace_unigrams[i] * N
        count_laplace_unigrams[i] =ans1 
    #printvalues(count_laplace_unigrams)
    #For bigrams
    bilen=len(bigrams.keys())
    count_laplace_bigrams={}
    for (i,j) in bigrams:
	ans3=prob_smoothed_bigram_2000[(i,j)]
        count_laplace_bigrams[(i,j)] =  ans3
	ans4=count_laplace_bigrams[(i,j)]* unigrams[i]
        count_laplace_bigrams[(i,j)] = ans4
    #printvalues(count_laplace_bigrams)
    #For trigrams
    count_laplace_trigrams={}
    trilen=len(trigrams)
    for (i,j,k) in trigrams:
	ans5= prob_smoothed_trigram_2000[(i,j,k)]
        count_laplace_trigrams[(i,j,k)] =ans5
	ans6= count_laplace_trigrams[(i,j,k)] * bigrams[(i,j)]
        count_laplace_trigrams[(i,j,k)] = ans6
    errorcheck(count_laplace_unigrams)
    	
    return count_laplace_unigrams, count_laplace_bigrams, count_laplace_trigrams


'''def plot1(dicti):
    errorcheck(dicti)
    x = []
    maxlen=len(dicti)
    y = []
    for i in range(len(dicti)):
	ycor=(dicti[i][1]))
        y.append(ycor)
        x.append((i))
    plt.plot(x,y)
    errorcheck(x)
    plt.show()'''

def plot_log_log1(dicti):
    errorcheck(dicti)
    x = []
    maxlen=len(dicti)
    y = []
    #new_ticks = []
    for i in range(1,len(dicti)):
	ycor=dicti[i][1]
        y.append(math.log(ycor))
        x.append(math.log(i))
	xcor=math.log(i)
        #new_ticks.append(dicti[i][0])
    errorcheck(x)
    plt.plot(x,y)
    errorcheck(y)
    #plt.xticks(x, new_ticks)
    plt.show()

def plot(dict1,dict2,dict3,dict4,l1,l2,l3,l4,name):
    errorcheck(dict1)
    x1 = []
    y1 = []
    maxlen=len(dict1)
    for i in range(maxlen):
	ycor=(dict1[i][1])
        y1.append(ycor)
        x1.append((i))

    plt.plot(x1,y1,label=l1)

    x2 = []
    maxlen1=len(dict2)
    y2 = []
    for i in range(len(dict2)):
	ycor=(dict2[i][1])
        y2.append(ycor)
        x2.append((i))
    plt.plot(x2,y2,label=l2)

    x3 = []
    maxlen2=len(dict3)
    y3 = []
    for i in range(len(dict3)):
	ycor=(dict3[i][1])
        y3.append(ycor)
        x3.append((i))
    plt.plot(x3,y3,label=l3)

    x4 = []
    maxlen3=len(dict4)
    y4 = []
    for i in range(len(dict4)):
	ycor=(dict4[i][1])
        y4.append(ycor)
        x4.append((i))
    plt.plot(x4,y4,label=l4)
    maxlen3=maxlen2+maxlen1+maxlen+maxlen3
    errorcheck(x4)
    plt.title(name)

    plt.legend()
    plt.show()



def plot_log_log(dict1,dict2,dict3,dict4,l1,l2,l3,l4,name):
    errorcheck(dict1)
    x1 = []
    y1 = []
    for i in range(len(dict1)):
    	if  i>=1:
		ycor=math.log(dict1[i][1])
        	y1.append(ycor)
		xcor=math.log(i)
        	x1.append(xcor)
    plt.plot(x1,y1,label=l1)

    x2 = []
    maxlen1=len(dict2)
    y2 = []
    for i in range(maxlen1):
    	if  i>=1:
		ycor=math.log(dict2[i][1])
        	y2.append(ycor)
		xcor=math.log(i)
        	x2.append(xcor)
    plt.plot(x2,y2,label=l2)

    x3 = []
    maxlen2=len(dict3)
    y3 = []
    for i in range(maxlen2):
    	if  i>=1:
		ycor=math.log(dict3[i][1])
        	y3.append(ycor)
		xcor=math.log(i)
        	x3.append(xcor)
		
    plt.plot(x3,y3,label=l3)

    x4 = []
    maxlen3=len(dict4)
    y4 = []
    for i in range(len(dict4)):
    	if  i>=1:
		ycor=math.log(dict4[i][1])
        	y4.append(ycor)
		xcor=math.log(i)
        	x4.append(xcor)
    plt.plot(x4,y4,label=l4) 
    errorcheck(x4)
    plt.title(name)

    plt.legend()
    plt.show()

def cond_bigrams(bigrams, key):
    keu1=bigrams.items()
    ar=[]
    joint = {k[1] : v for k, v in bigrams.items() if k[0] == key}
    x=0
    sum_count = sum(joint.values())
    sum1=sum1+sum_count
    arr.append(sum1)
    return {k : v / float(sum_count) for k, v in joint.items() }

def generate_bigrams(unigrams, bigrams, length=5, first_word = None):
    words = []
    biw=[]
    if first_word == None:
	y=random.randrange(0, len(unigrams))
        first_word = list(unigrams.keys())[y]
    words.append(first_word)
    biw.append(word[0])
    for i in range(length - 1):
        biw.append(words[i])
        prev_dict = cond_bigrams(bigrams, words[i])
        ab=len(prev_dict.keys())   
        next_word = sorted(prev_dict.items(), key = lambda x : x[1], reverse = True)[0]
	errorcheck(words)
        words.append(next_word[0])

    return words
def cond_trigrams(trigrams, key):
    pre=[]
    joint = {k[2] : v for k, v in trigrams.items() if (k[0] == key[0] and k[1] == key[1])}
    a=0
    pre.append(key[0])
    sum_count = sum(joint.values())
    sum1=sum_count+13
    return {k : v / float(sum_count) for k, v in joint.items() }

def generate_trigrams(unigrams, bigrams,trigrams, length=5, first_word = None):
    words = []
    word1=[]
    if first_word == None:
	word1.append(len(unigrams.keys()))
        first_word = list(bigrams.keys())[random.randrange(0, len(bigrams))]
    j=0
    words=(list(first_word))
    print words
    for i in range(length - 2):
	word1.append(words[j])
        prev_dict = cond_trigrams(trigrams, [words[i], words[i+1]])
        
        next_word = sorted(prev_dict.items(), key = lambda x : x[1], reverse = True)[0]
	f1=0
        words.append(next_word[0])
	j=j+1
    return words



corpus=tokenise("anime.txt")

unigrams,unigrams_prob=get_unigrams(corpus)
V=200
#laplace_unigrams_prob = get_laplace_unigrams(unigrams,V)
wittenbell_unigrams_prob = get_wittenbell_unigrams(unigrams,unigrams_prob,V)
plot_log_log1(sort_dict(wittenbell_unigrams_prob))
#plot(sort_dict(wittenbell_unigrams_prob1),sort_dict(wittenbell_unigrams_prob2),sort_dict(wittenbell_unigrams_prob3),sort_dict(wittenbell_unigrams_prob4),"200","2000",len(unigrams),10*len(unigrams),"zipf_unigrams_diff_voc_witten_anime")
#plot_log_log(sort_dict(laplace_unigrams_prob1),sort_dict(laplace_unigrams_prob2),sort_dict(laplace_unigrams_prob3),sort_dict(laplace_unigrams_prob4),"200","2000",len(unigrams),10*len(unigrams),"log_unigrams_diff_voc_news")
kn_unigrams_prob = get_kn_unigrams(unigrams,200)
plot_log_log1(sort_dict(kn_unigrams_prob))


bigrams,bigrams_prob = get_bigrams(corpus,unigrams)
V=200
laplace_bigrams_prob1 = get_laplace_bigrams(unigrams,bigrams,V)
laplace_bigrams_prob2 = get_laplace_bigrams(unigrams,bigrams,V*10)
V=len(unigrams)
laplace_bigrams_prob3 = get_laplace_bigrams(unigrams,bigrams,V)
laplace_bigrams_prob4 = get_laplace_bigrams(unigrams,bigrams,10*V)
#wittenbell_bigrams_prob = get_wittenbell_bigrams(unigrams,bigrams,unigrams_prob,wittenbell_unigrams_prob)
plot(sort_dict(laplace_bigrams_prob1),sort_dict(laplace_bigrams_prob2),sort_dict(laplace_bigrams_prob3),sort_dict(laplace_bigrams_prob4),"200","2000",V,10*V,"zipf_bigrams_diff_voc_anime")
plot_log_log(sort_dict(laplace_bigrams_prob1),sort_dict(laplace_bigrams_prob2),sort_dict(laplace_bigrams_prob3),sort_dict(laplace_bigrams_prob4),"200","2000",V,10*V,"log_bigrams_diff_voc_anime")
kn_bigrams_prob =  get_kn_bigrams(unigrams,bigrams)


trigrams,trigrams_prob = get_trigrams(corpus,bigrams)
V=200
laplace_trigrams_prob1 = get_laplace_trigrams(unigrams,bigrams,trigrams,V)
laplace_trigrams_prob2 = get_laplace_trigrams(unigrams,bigrams,trigrams,V*10)
V=len(unigrams)
laplace_trigrams_prob3 = get_laplace_trigrams(unigrams,bigrams,trigrams,V)
laplace_trigrams_prob4 = get_laplace_trigrams(unigrams,bigrams,trigrams,10*V)
#wittenbell_trigrams_prob = get_wittenbell_trigrams(unigrams,trigrams,trigrams_prob,wittenbell_bigrams_prob)
plot(sort_dict(laplace_trigrams_prob1),sort_dict(laplace_trigrams_prob2),sort_dict(laplace_trigrams_prob3),sort_dict(laplace_trigrams_prob4),"200","2000",V,10*V,"zipf_trigrams_diff_voc_anime")
plot_log_log(sort_dict(laplace_trigrams_prob1),sort_dict(laplace_trigrams_prob2),sort_dict(laplace_trigrams_prob3),sort_dict(laplace_trigrams_prob4),"200","2000",V,10*V,"log_trigrams_diff_voc_anime")
kn_trigrams_prob =  get_kn_trigrams(unigrams,bigrams,trigrams)

kn_unigrams,kn_bigrams,kn_trigrams=estimated_count_kn(unigrams, bigrams, trigrams, kn_unigrams_prob, kn_bigrams_prob, kn_trigrams_prob)
ww1=generate_trigrams(unigrams,bigrams,trigrams,5,None)
print (ww1)

corpus_anime = tokenise("anime.txt")
anilen=len(corpus_anime)
corpus_movies = tokenise("movies.txt")
movlen=len(corpus_anime)
corpus_news = tokenise("news.txt")
newlen=len(corpus_anime)

unigrams_anime,unigrams_prob_anime = get_unigrams(corpus_anime)
uniani=len(unigrams_anime)
bigrams_anime,bigrams_prob_anime = get_bigrams(corpus_anime,unigrams_anime)
biani=len(bigrams_anime)
trigrams_anime,trigrams_prob_anime = get_trigrams(corpus_anime,bigrams_anime)
triani=len(trigrams_anime)

unigrams_movies,unigrams_prob_movies = get_unigrams(corpus_movies)
unimov=len(unigrams_movies)
bigrams_movies,bigrams_prob_movies = get_bigrams(corpus_movies,unigrams_movies)
bimov=len(bigrams_movies)
trigrams_movies,trigrams_prob_movies = get_trigrams(corpus_movies,bigrams_movies)
trimov=len(trigrams_movies)

unigrams_news,unigrams_prob_news = get_unigrams(corpus_news)
uninew=len(unigrams_news)
bigrams_news,bigrams_prob_news = get_bigrams(corpus_news,unigrams_news)
binew=len(bigrams_news)
trigrams_news,trigrams_prob_news = get_trigrams(corpus_news,bigrams_news)
trinew=len(trigrams_news)

plot(sort_dict(unigrams_anime),sort_dict(unigrams_movies),sort_dict(unigrams_news),{},"anime_uni","movies_uni","news_uni","none","Unigrams Zipfs")
errorcheck(sort_dict(unigrams_anime))
plot(sort_dict(bigrams_anime),sort_dict(bigrams_movies),sort_dict(bigrams_news),{},"anime_bi","movies_bi","news_bi","none","Bigrams Zipfs")
errorcheck(sort_dict(bigrams_anime))
plot(sort_dict(trigrams_anime),sort_dict(trigrams_movies),sort_dict(trigrams_news),{},"anime_tri","movies_tri","news_tri","none","Trigrams Zipfs")
errorcheck(sort_dict(trigrams_anime))

