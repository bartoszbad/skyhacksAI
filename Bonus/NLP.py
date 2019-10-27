############################################################################################################################################
###############################                       READ ME                         ######################################################
############################################################################################################################################
# Określona liczba słów w etykietach. - > Etykiet jest około 50. Wśród nich jest tylko kilka nazw pomieszczeń, mebli itp.
# Słowa nie zawarta w słownikach z etykiet np white lub interior design, można sprawdzać dzięki nltk jaka to część mowy i odpowiednio wrzucać w zdanie.
# #z nltk synonimy, i inne fajne funkcje do łączenia słów w zdania. Może jakiś nowy json?

# tutaj tutorial nltk
# https://likegeeks.com/nlp-tutorial-using-python-nltk/#Tokenize-Text-Using-NLTK

# tutaj model sieci na word2vec na nltk, wrzucić zamiast bibli książkę/reklamę/broszurę o domach/pokojach i fajrant?
# https://textminingonline.com/dive-into-nltk-part-x-play-with-word2vec-models-based-on-nltk-corpus
# lista gotowcó z nltk
# Narazie jest szkielet mini modelu który uloży banalne zdania z etykiet.

############################################################################################################################################
############################################################################################################################################
import json
import datetime
import glob
import os

from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
import nltk
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from reportlab.lib.utils import ImageReader
from PIL import Image
import textwrap
import random

nltk.download('wordnet')

# Listy slow z etykiet.
RoomsList = ['Bedroom', 'Bathroom', 'Dining room', 'Kitchen', 'Living room', 'Room', 'Toilet']
RoomObjects = ['Bathroom cabinet', 'Bathroom sink', 'Bathub', 'Bed', 'Bed frame', 'Bed sheet', 'Cabinetry',
               'Chair', 'Chandelier', 'Chest of drawers', 'Coffe table', 'Couch', 'Countertop', 'Cupboard', 'Curtain',
               'Drawer', 'Fireplace', 'Hardwood', 'Kitchen & dining room table',
               'Kitchen stove', 'Mattress', 'Nightstand', 'Plumbing fixture', 'Refrigerator', 'Shower', 'Sink', 'Table',
               'Tablecloth', 'Tap', 'Window']
OutsideObjects = ['Grass', 'Facade', 'House', 'Property', 'Real estate', 'Roof', 'Rural area', 'Sky', 'Tree',
                  'Urban area']

input_dir = "input/test_1"
c = canvas.Canvas(datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S.pdf"))
c.setLineWidth(.3)
c.setFont('Helvetica', 12)
page_size = (600, 1150)
img_size = 200
c.setPageSize(page_size)
c.drawImage("bg.jpg", 0, 0, 600, 1150)

def get_syn(name):
    return str(wn.synsets(name)[random.randint(0, len(wn.synsets(name)) - 1)].lemmas()[0].name())


coords = [[50, 50], [50, 260], [50, 470], [50, 680], [50, 890], [50, 1100]]
i = len(coords) - 1
for files in os.listdir(input_dir):
    if files[::-1][:4] == "gpj.":
        # c.drawImage("blur.png", coords[i][0]+5, coords[i][1]-5, img_size, img_size, mask='auto')
        image = ImageReader(input_dir + "/" + str(files))
        c.drawImage(image, coords[i][0], coords[i][1]-200, img_size, img_size)
    if ".json" not in files:
        continue

    with open(input_dir + "/" + files) as file:
        d = json.load(file)

        was_obj = 0
        was_room = 0
        Sentence = ''
        for entity in d:
            if entity['name'] in RoomsList and was_room == 0:
                Sentence = Sentence + entity['name']
                was_room = 1
            elif entity['name'] in RoomObjects:
                if was_obj == 0:
                    if entity['name'][0] in ('a', 'e', 'i', 'o', 'u'):
                        Sentence = Sentence + ' has an ' + get_syn(entity['name'].lower()) + ' and'
                        was_obj = 1
                    else:
                        Sentence = Sentence + ' has a ' + entity['name'].lower() + ' and'
                        was_obj = 1
                else:
                    if entity['name'][0] in ('a', 'e', 'i', 'o', 'u'):
                        Sentence = Sentence + ' an ' + get_syn(entity['name'].lower()) + ' and'
                        was_obj = 1
                    else:
                        Sentence = Sentence + ' a ' + entity['name'].lower() + ' and'
                        was_obj = 1

        Sentence = Sentence + ' above all is a great place'
        for entity in d:
            if entity['name'] in OutsideObjects:
                Sentence2 = 'On the photo you can see a ' + get_syn('wonderful') + " " + entity['name'].lower() + ' and '
        Sentence2 = Sentence2
        Sentence3 = 'it is available for ' + get_syn('purchase') + ' just now!'
        level = 0
        wrap_text = textwrap.wrap(Sentence + ". " + Sentence2 + Sentence3, width=50)
        for wrapped_text in wrap_text:
            c.drawString(coords[i][0] + img_size + 20, coords[len(coords) - 1][1] - coords[len(coords) - 1 - i][1] + img_size - 10 - level*14 -200, wrapped_text)
            level += 1

        i -= 1
c.save()
