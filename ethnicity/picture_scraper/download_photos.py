import urllib2
import os

from urllib import FancyURLopener
from random import choice

user_agents = [
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
    'Opera/9.25 (Windows NT 5.1; U; en)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
    'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9'
]

class MyOpener(FancyURLopener, object):
	version = choice(user_agents)
	
filter_file = open('/home/james/data/demographics/id_gender_african.tsv')

filters = {}
for i in filter_file:
	uid = i.strip().split('\t')[0]
	filters[uid] = None

photo_urls = open('/home/james/data/demographics/photo_urls_ids.tsv')

counter = 0

for line in photo_urls:

	counter += 1
	
	if counter % 1000 == 0:
		print(counter)

	uid,url = line.strip().split('\t')
	fn,ext = os.path.splitext(url)
	
	if not uid in filters:
		continue
	
	if os.path.isfile('/home/james/data/demographics/african_photos/'+uid+'.'+ext):
		continue

	myopener = MyOpener()
	
	try:
		content = myopener.open(url).read()
	
		photo = open('/home/james/data/demographics/african_photos/' + uid + '.' + ext,'w')
	
		photo.write(content)
	
		photo.close()
	
	except IOError:
		print(url)
		continue
	
