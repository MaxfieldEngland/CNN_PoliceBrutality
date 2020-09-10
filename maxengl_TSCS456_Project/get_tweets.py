# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:07:14 2020

Used to collect tweets across a week-long period in the first phase; in the second phase, 
the script is used to generate cleaned testing and training corpora.

@author: Maxfield England
"""

# import tweepy
# import json
# import time
from nltk.corpus import stopwords
import re
import string
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#Phase 1: Collecting tweets
#Approx. once a day tweets are collected (100 per search term, across 10 search terms = approx./day)
#Current counted days: 7 7000 tweets analyzed)
# consumer_key = "jyjfLM9rzwimmE0xw30Tm5Dqq"
# consumer_secret = "Fptyjcggr1e2nnMOOUWKBMbdgrqnYjyscRYePo6h3D2a9EVvhZ"
# access_key ="1027293645676769280-jO1P77gG9NsJ0KJCg9THwLBjyy6DNS" 
# access_secret = "5pvsK3oxrrXn8oeqUVYkyQQt0lLHCoxSA8yBlKa2ZhR6c"

# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_key, access_secret)

# api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

searchterms = ['#EXPOSEANTIFA', '#blacklifematters', '#cops', 'police brutality', '#uniformlife',
               '#thinblueline', '#blacklivesmatter', '#alllivesmatter', '#BLUEFALL', '#ProjectVeritas']

# searchqueries = [(term + ' -filter:retweets') for term in searchterms]
# tweetList = []
    
# for searchquery in searchqueries:

#     data = api.search(q = searchquery, count = 100, lang = 'en', result_type = 'mixed', tweet_mode = 'extended')
#     time.sleep(5)
#     for tweetData in data['statuses']:
#         tweetList.append(tweetData['full_text'])
#     print("Tweets loaded for search term", searchquery)
    
#     # data_all = data['statuses']
#     # while (len(data_all) <= 2000):
#     #     time.sleep(5)
#     #     last = data_all[-1]['id']
#     #     data = api.search(q = searchquery, count=100, lang='en', result_type='mixed', max_id = last)
#     #     data_all += data['statuses'][1:]

#     # for i in range(0, len(data_all)):
#     #     tweet.append(data_all[i]['text'])
        
# corpus = '|ENDOFTWEET|'.join(tweetList)

# outfile = "PoliceBrutalityCorpus_Twitter7.txt"
# file = open(outfile, 'w', encoding='utf8')
# file.write(corpus)
# file.close()

#Phase 2: Processing Corpus

def compile_corpus(files):
    
    training = []
    testing = []
    
    for filename in files:
        file = open(filename, 'r', encoding='utf8')
        #read all text in file
        text = file.read()
        #close file
        file.close()
        tweets = text.split("|ENDOFTWEET|")
        print("Getting tweet pool from ",filename, ":",len(tweets),"tweets")
        length = len(tweets)
        for i in range(length):
            if i % 10 == 0:
                testing.append(tweets[i])
            else:
                training.append(tweets[i])
    print("Training:",len(training))
    print("Testing:",len(testing))
        
    #Remove duplicate tweets from corpora (in case any got picked up twice):
            #Potential issue 1: tweets existing in both testing and training corpora could have some kind of skew?
            #potential issue 2: duplicate removal could significantly affect counts in some category over another; but as long as we have a significant
            #amount of tweets in training, I don't think the model efficacy should be 
    for i in range(len(testing)):
        for j in range(i+1, len(testing)):
            if j < len(testing) and testing[i] == testing[j]:
                testing.remove(testing[j])
        
        
    for i in range(len(training)):
        for j in range(i+1, len(training)):
            if j < len(training) and training[i] == training[j]:
                training.remove(training[j])
                
    return training, testing

#Use our standard corpus cleaning for tweets! TODO: FIX, VERY BROKEN.
    # Keep tweets separate while cleaning each one? Look into that boyee


#Replace a tweet with its essential tokens
def clean_tweet(tweet):

    tokens = tweet.split()
    
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    #tokens = [word.lower() for word in tokens]
    #Get rid of stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    #filter out short tokens (single-char)
    tokens = [word for word in tokens if len(word) > 1]

    return ' '.join(tokens)
    

#List the tweet files we made over the 7 days
filenames = ("PoliceBrutalityCorpus_Twitter.txt", "PoliceBrutalityCorpus_Twitter2.txt", "PoliceBrutalityCorpus_Twitter3.txt",
             "PoliceBrutalityCorpus_Twitter4.txt", "PoliceBrutalityCorpus_Twitter5.txt", "PoliceBrutalityCorpus_Twitter6.txt",
             "PoliceBrutalityCorpus_Twitter7.txt")

training, testing = compile_corpus(filenames)
cleanTraining = [clean_tweet(trainTweet) for trainTweet in training]
cleanTesting = [clean_tweet(testTweet) for testTweet in testing]
        

with open('training.raw', 'wb') as trainRawWriteFile:
    pickle.dump(training, trainRawWriteFile)
    
with open('testing.raw', 'wb') as testRawWriteFile:
    pickle.dump(testing, testRawWriteFile)


with open('training.corpus', 'wb') as trainWriteFile:
    pickle.dump(cleanTraining, trainWriteFile)
    
with open('testing.corpus', 'wb') as testWriteFile:
    pickle.dump(cleanTesting, testWriteFile)
    
allTweets = training + testing
sen = SentimentIntensityAnalyzer()   
 
for term in searchterms:
    currentTweets = []
    numPos = 0
    numNeg = 0
    numNeutral = 0
    scoreTotal = 0
    termCount = 0
    for tweet in allTweets:

        if tweet.count(term) > 0:
            currentTweets.append(tweet)
            termCount += 1
        
    for tw in currentTweets:
        score = sen.polarity_scores(tw)['compound']
        scoreTotal += score
        if term == '#blacklivesmatter':
            print()
            print("------------------------")
            print("Tweet:", tw)
            print("Score:", score)
            print("------------------------")
            print()
        if score < -.1:
            numNeg += 1
        elif score > 0.1:
            numPos += 1
        else:
            numNeutral += 1
            
    if termCount > 0:
        avgScore = scoreTotal / termCount
    else:
        avgScore = 0
    
    print("VADER results for term:", term)
    print("Number of tweets that are positive:",numPos)
    print("Number of tweets that are negative:",numNeg)
    print("Number of tweets that are neutral:", numNeutral)
    print("Number of tweets:", termCount)
    print("Average score:", avgScore)
    


