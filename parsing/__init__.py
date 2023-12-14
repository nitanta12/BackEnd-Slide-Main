from newspaper import Article
from datetime import datetime
import docx2txt

def parse_url(url):
    article = Article(url)
    article.download()
    article.parse()
    
    if article.title  is None:
        article.title = 'News Article'
    # print (article.)
    
    if len(article.authors) == 0:
        article.authors.append('Anonymous Author')
    
    if article.text is None:
        article.text = 'No Text'

    # remove all the special characters not recognized by the utf-8
    translation_table = dict.fromkeys(map(ord, '’—“”…‘'), None)
    article.text = article.text.translate(translation_table)
    
    document = {
        'title': article.title,
        'author': article.authors,
        'date': article.publish_date,
        'text': article.text,
        'image': article.top_image, 
        'images': article.images,
        'html': article.html,
        'slides': {},
        'no_of_slides': 0,
    }

    return document

def parse_text(text):


    document = {
        'title': 'Article',
        'author': ['Anonymous Author'],
        'date': None,
        'text': text,
        'image': None, 
        'images': [],
        'html': None,
        'slides': {},
        'no_of_slides': 0,
    }
    return document

def parse_upload(upload):

    # combine all the text of the upload dict in one string
    upload = ' '.join(upload.values())

    document = {
        'title': 'Article',
        'author': ['Anonymous Author'],
        'date': None,
        'text': upload,
        'image': None, 
        'images': [],
        'html': None,
        'slides': {},
        'no_of_slides': 0,
    }
    return document