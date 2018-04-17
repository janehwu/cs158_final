from __future__ import division
import emoji

#get number of tweets that contain emojis:
def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI

def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False



"""Given file of tweets, returns list of # emojis per tweet"""
def num_emojis_per_tweet(file):
	countList = []

	with open(fname) as f:
		tweets = f.readlines()

	for tweet in tweets:
		emojiCount = 0
		words = tweet.split(" ")
		for word in words:
			if char_is_emoji(word):
				emojiCount += 1

		countList.append(emojiCount)

	return countList


"""Given file of tweets, """
def percentage_tweets_with_emojis(file):
	with open(fname) as f:
		tweets = f.readlines()

	emojiCount = 0
	for tweet in tweets:
		if text_has_emoji(tweet):
			emojiCount+=1
	numTweets = len(tweets)

	return emojicount/numTweets 




'''Counts of type of punctuation per tweet'''
def percentage_tweets_with_punctuation(file):
	with open(fname) as f:
		tweets = f.readlines()
	numTweets = len(tweets)

	commas, periods, semicolons, colons, exclamations = (0,)*5

	for tweet in tweets: 
		if ',' in tweet: commas += 1
		if '.' in tweet: periods += 1
		if ';' in tweet: semicolons += 1
		if ':' in tweet: colons += 1
		if '!' in tweet: exclamations += 1

	commacount = commas/numTweets
	periodcount = periods/numTweets
	semicoloncount = semicolons/numTweets
	coloncount = colons/numTweets
	exclamationcount = exclamations/numTweets

	return commacount, periodcount, semicoloncount, coloncount, exclamationcount