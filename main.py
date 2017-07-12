import csv
import os

import datetime
from collections import defaultdict
import heapq
from operator import  itemgetter

def load_reviews(path,**kwargs):
    options={
        'fieldnames':('userid','movieid','rating','timestamp'),
        'delimiter':'\t',
    }

    options.update(kwargs)
    parse_date=lambda r,k: datetime.datetime.fromtimestamp(float(r[k]))
    parse_int = lambda r,k: int(r[k])

    with open(path,'r',encoding="ISO-8859-1") as reviews:
        reader=csv.DictReader(reviews,**options)
        for row in reader:
            row['movieid'] = parse_int(row,'movieid')
            row['userid'] = parse_int(row,'userid')
            row['rating'] = parse_int(row,'rating')
            row['timestamp'] = parse_date(row,'timestamp')
            yield row

def relative_path(path):
    """
    return a path relative from this code file
    """
    dirname = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(dirname,path)
    return os.path.normpath(path)


def load_movies(path,**kwargs):
    options={
    'fieldnames':('movieid','title','release','video','url'),
    'delimiter' :'|',
    'restkey': 'genre',
    }
    options.update(kwargs)
    parse_date=lambda r,k: datetime.datetime.strptime(r[k],'%d-%b-%Y') if r[k] else None
    parse_int = lambda r,k: int(r[k])

    with open(path,'r',encoding="ISO-8859-1") as movies:
        reader=csv.DictReader(movies,**options)
        for row in reader:
            row['movieid'] = parse_int(row,'movieid')
            row['release'] = parse_date(row,'release')
            row['video'] = parse_date(row,'video')
            yield row

class MovieLens(object):
    """
    data structure to build our recomend
    """
    def __init__(self,udata,uitem):
        """
        instantiate with a path to u.data and u.item
        """
        self.udata= udata
        self.uitem=uitem
        self.movies={}
        self.reviews=defaultdict(dict)
        self.load_dataset()



    def load_dataset(self):
        """
        loads two datasets into memory, indexed on id
        """
        for movie in load_movies(self.uitem):
            self.movies[movie['movieid']] = movie

        for review in load_reviews(self.udata):
            self.reviews[review['userid']][review['movieid']]=review


    def reviews_for_movie(self,movieid):
        """
        yields the reviews for a given movie
        """
        for review in self.reviews.values():
            if movieid in review:
                yield review[movieid]

    def average_reviews(self):
        """
        avg the star rating for all movies. yields a tuple of movieid,
        the avg rating,and num of reviews
        """

        for movieid in self.movies:
            reviews= list(r['rating'] for r in self.reviews_for_movie(movieid))
            average=sum(reviews)/float(len(reviews))
            yield (movieid,average,len(reviews))

    def bayesian_average(self,c=59,m=3):
        """
        reports the bayesian avg with parameter c and m
        """
        for movieid in self.movies:
            reviews= list(r['rating'] for r in self.reviews_for_movie(movieid))

            #only want reviews more than 300
            #if do not care
            #remove if condition
            #only want the rating is more or equal to 4 
            if len(reviews) >300:
                average=((c * m)+ sum(reviews)) / float(c + len(reviews))

                if average>=4.0:
                    yield (movieid,average,len(reviews))

    def top_rated(self,n=10):
        """
        yields the n top rated movies
        """
        return heapq.nlargest(n,self.bayesian_average(),key=itemgetter(1))


def main():
    data=relative_path('u.data')
    item=relative_path('u.item')
    model = MovieLens(data,item)

    for mid,avg,num in model.top_rated(10):
        title=model.movies[mid]['title']

        print ("[%0.3f average rating (%i reviews)] %s" % (avg,num,title))

if __name__=='__main__':
    main()
