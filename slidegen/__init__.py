import os
from pdf2image import convert_from_path

image_loc= 'output/images'

def create_markdown(document):
    md = """"""
    f = open("slidegen/theme.md", "r")
    atext = f.read()
    f.close()
    md += atext+"\n"
    md += create_home_slide(document)

    for i, (topic, content) in enumerate(document['slides'].items()):
        image_link = ''
        for num, sentences in content.items():
            if num == -1:
                image_link = sentences
            elif num==0 and i!=0:
                md += create_first_slide(topic,sentences, image_link)
            else: 
                md += create_new_slide(topic,sentences)

    return md

def create_home_slide(document):
    # Create a home slide
    text = "\n# " + document['title'] + "\n" + document['author'][0] + "\n\n"
    if document['image'] != '':
        text += "---\n![bg]({})\n".format(document['image'])
    return text

def create_first_slide(topic,content,image_link): 

    text = "\n---\n![bg right]({})".format(image_link)
    text += "\n# " + topic + "\n"
    # adding the bullet points
    for sentence in content:
        text += "\n- " + sentence + "\n"
    return text

def create_new_slide(topic,content):
    text = "\n---\n# " + topic + "\n"
    # adding the bullet points
    for sentence in content:
        text += "\n- " + sentence + "\n"
    return text

def convert_to_frames():
    # Convert the pdf to frames
    images_from_path = convert_from_path('output.pdf')
    for i, image in enumerate(images_from_path):
        image_path = os.path.join(image_loc, 'frame_{}.jpg'.format(i))
        image.save(image_path)

def create_slides(document):
    os.mkdir(image_loc)
    md = create_markdown(document)
    f = open('output.md', 'w')
    f.write(md)
    f.close()
    #marp should have executable permissions
    os.system('marp output.md --pdf ')
    #other commit was
    #os.system('./marp output.md --pdf')
    convert_to_frames()
