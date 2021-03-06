# Download/format all tweets

import tweepy, datetime


def setup():
	consumer_key = 'A5NW9UldCETVB6NNxTlecffAI'
	consumer_secret = 'tSHuZRpWS20sRnmLjg6T0ka0zf0WbKtjZrKpsZ0cZjmWBNXIhz'
	access_token = '980319213083533312-aQFRLVubxRz1C9mbKafjexJ7zTnXRbc'
	access_token_secret = 'wIgXtimDt13gZMHfhiTvMbPCKAIqyqTQogKSlnvFVqyar'

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)
	return api

'''
Get _username_'s tweets from _year_
'''
def get_tweets(api, username, year, filename, start=66, end=200):
	f = open(filename, 'w')
	for page in range(start, end):
		tweets = api.user_timeline(username, page=page)
		print page, "First tweet:", tweets[0].created_at
		for tweet in tweets:
			tweet_year = tweet.created_at.year
			if tweet_year == year:
				print "Adding"
				text = tweet.text.encode("utf-8")
				f.write(text + '\n')
			elif tweet.created_at.year < year:
				print "Done"
				print tweet.created_at.year
				f.close()
				return

'''
Clean tweets by removing URLs and re-tweets
'''
def clean_tweets(oldfile, newfile):
	f_new = open(newfile, 'w')

	with open(oldfile) as f_old:
		for line in f_old:
			words = line.split(" ")

			if words[0] != 'RT': # Exclude re-tweets
				new_line = ""
				for i in range(len(words)):
					w = words[i]
					if len(w) > 0 and w[0] == '@':
						print 'removing @', w
						continue
					elif (len(w) > 5) and (w[:5] == 'https'): # Remove URLs (always at end)
						print 'ignoring', w
						new_line += '\n'
						break
					new_line += w

					if i != len(words) - 1:
						new_line += ' '
				
				if new_line != '\n':
					f_new.write(new_line)
	f_new.close()

def combine_tweets(pos_file, neg_file):
	pos_f = open(pos_file, 'r')
	neg_f = open(neg_file, 'r')
	f = open('final_data/all_tweets.txt', 'w')
	labels = open('final_data/labels.txt', 'w')
	for line in pos_f.readlines():
		f.write(line)
		labels.write('1\n')
	for line in neg_f.readlines():
		f.write(line)
		labels.write('-1\n')
	pos_f.close()
	neg_f.close()
	f.close()
	labels.close()

def main():
	api = setup()
	combine_tweets('final_data/kim_tweets_clean.txt', 'final_data/khloe_tweets_clean.txt')

	# Download tweets
	#get_tweets(api, 'KimKardashian', 2017, 'data/kimk_tweets_2017.txt', start=30, end=150)
	#get_tweets(api, 'khloekardashian', 2017, 'data/khloe_tweets_2017.txt', start=66, end=163)

	# Re-format tweets
	#clean_tweets('raw_data/kim_tweets_2017.txt', 'final_data/kim_tweets_clean.txt')
	#clean_tweets('raw_data/khloe_tweets_2017.txt', 'final_data/khloe_tweets_clean.txt')

main()

