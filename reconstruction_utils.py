import matplotlib.pyplot as plt
import keras_ocr
import math
import numpy as np
import cv2 
import tensorflow as tf

import os
import pathlib
import time
import datetime

import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter
from google.colab.patches import cv2_imshow
from string import ascii_lowercase
from IPython import display

pipeline = keras_ocr.pipeline.Pipeline()

def read_text(img_path):
  img = keras_ocr.tools.read(img_path)
  prediction_groups = pipeline.recognize([img])
  keras_ocr.tools.drawAnnotations(image=img, predictions=prediction_groups[0])
  return prediction_groups

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)


def get_text_coords(prediction_groups):
  box = prediction_groups[0][0][1].tolist()

  x0, y0 = box[0]
  x1, y1 = box[1] 
  x2, y2 = box[2]
  x3, y3 = box[3] 
          
  x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
  x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)
  return x_mid0, y_mid0, x_mid1, y_mid1
#thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))


def load_vocabulary(word_file_path):
  words = []
  with open(word_file_path, 'r') as f:
      file_name_data = f.read()
      file_name_data=file_name_data.lower()
      words = re.findall('[a-zA-Z]+',file_name_data)
  # This is our vocabulary
  V = set(words)
  #print("Top ten words in the text are:", words[0:10])
  #print("Total Unique words are ", len(V))
  return words

def get_prob(words):
  word_freq = {}  
  word_freq = Counter(words)
  probs = {}     
  Total = sum(word_freq.values())    
  for k in word_freq.keys():
      probs[k] = word_freq[k]/Total
  return word_freq, probs

def my_autocorrect(words, incomplete_word, flag, start="", end = ""):
  word_freq, probs = get_prob(words)      
  input_word = incomplete_word.lower()
  sim = [1-(textdistance.Jaccard(qval=2).distance(v,incomplete_word)) for v in word_freq.keys()]
  df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
  df = df.rename(columns={'index':'Word', 0:'Prob'})
  df['Similarity'] = sim

  filter = df['Word'].apply(lambda x: any([True if len(word)>=len(incomplete_word) else False for word in x.split(' ')]))
  output = df[filter] #output.loc[len(output["Word"]) >= len(incomplete_word)]

  if (flag == True):
      #filter = output['Word'].apply(lambda x: any([True if word[0]==start and  word[-1]==end else False for word in x.split(' ')]))
      #output = output[filter] #output.loc[len(output["Word"]) >= len(incomplete_word)]
      my_regex = r"^" + re.escape(start)+".*"+re.escape(end)+"$"
      #filter = output[output['Word'].str.match('^P.*')== True]
      filter = output['Word'].apply(lambda x: any([True if re.match(my_regex, word) else False for word in x.split(' ')]))
      output = (output[filter])


  output = output.sort_values(['Similarity', 'Prob'], ascending=False).head()
  output = output.loc[output["Word"] != incomplete_word]

  return(output)

def missing_letters(correct, wrong):
    return set(correct) - set(wrong)

def get_craft_coords(img_name):
  #f = open("/content/CRAFT-pytorch/result/res_"+img_name+".txt", "r")
  f = open("craft_outputs/image_text_detection.txt", "r")

  lines = f.readlines() 
  x1 = []
  y1 = []
  x2 = []
  y2 = []
  x3 = []
  y3 = []
  x4 = []
  y4 = []


  for l in lines:
    x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_ = l.split(',')
    #if(y1<y2): 
    #y1.append(y1_)
    #else:
    x1.append(int(x1_))
    y1.append(int(y1_))
    x2.append(int(x2_))
    y2.append(int(y2_))
    x3.append(int(x3_))
    y3.append(int(y3_))
    x4.append(int(x4_))
    y4.append(int(y4_))
  return x1, y1, x2, y2, x3, y3, x4, y4

def extract_craft_text_region(img_name, img_path):
  x1, y1, x2, y2, x3, y3, x4, y4 = get_craft_coords(img_name)
  img = cv2.imread(img_path)
  lol = img.shape[1]*10//1600        
  real_text = img[y1[0]-10:y3[0]+10, x1[0]-lol:x3[0]+lol]
  #real_text = img[ 260:506, 665:1106]
  
  #extract_im = np.zeros((real_text.shape[0], real_text.shape[1]), dtype=np.uint8)
  #extract_im = extract_im + 128
  #extract_im[y1[0]-10:y3[0]+10, x1[0]-lol:x3[0]+ol] = real_text

  col = np.mean(real_text)
  if (col<150):
    colour = (0, 0, 0)
  elif (col<180):
    colour = (128, 128, 128)
  else:
    colour = (255, 255, 255)
  real_text = cv2.copyMakeBorder(real_text,30,30,30,30,cv2.BORDER_CONSTANT,value=colour)


  cv2_imshow(real_text)
  cv2.imwrite('cur_test/i_s.png', real_text)
  return real_text

def create_letter_img(text, im_shape):
  i_t = np.zeros((142, 110), dtype=np.uint8)
  i_t = i_t + 128
  org = (10, 110)
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 4
  color = (0, 0, 0)
  thickness = 10 #line thickness
  i_t = cv2.putText(i_t, text, org, font, fontScale, 
                 color, thickness, cv2.LINE_AA, False)
  i_t = cv2.resize(i_t, im_shape)
  return i_t

def non_stylised_gen(img_name, img_path, letter_write):
  i_s = extract_craft_text_region(img_name, img_path)
  
  #text = word_list['Word'].iloc[0].upper()
  for i in letter_write:
    text = str(i).upper()

  i_t = create_letter_img(text, (i_s.shape[1], i_s.shape[0]))
  cv2_imshow(i_t)
  cv2.imwrite('cur_test/i_t.png', i_t)

def sharpen_image(translated_im):
  kernel = np.array([[0, -1, 0],
                    [-1, 5,-1],
                    [0, -1, 0]])
  image_sharp = cv2.filter2D(src=translated_im, ddepth=-1, kernel=kernel)
  cv2_imshow(image_sharp)
  return image_sharp

def final_integration(img, img_name, letter_write, word_list):
  translated_im = cv2.imread('results/result.png')
  cv2.imwrite('cur_test/sharp.png', translated_im)

  translated_im = translated_im[30:-30, 30:-30]
  cv2_imshow(translated_im)
  image_sharp = sharpen_image(translated_im)
  sharp_im  = keras_ocr.tools.read('cur_test/sharp.png')
  prediction_groups = pipeline.recognize([sharp_im])
  keras_ocr.tools.drawAnnotations(image=sharp_im, predictions=prediction_groups[0])
  if prediction_groups==[[]]:
    sharp_im  = keras_ocr.tools.read('results/result.png')
    prediction_groups = pipeline.recognize([sharp_im])
    keras_ocr.tools.drawAnnotations(image=sharp_im, predictions=prediction_groups[0])

  box = prediction_groups[0][0][1].tolist()

  x0_2, y0_2 = box[0]
  x1_2, y1_2 = box[1] 
  x2_2, y2_2 = box[2]
  x3_2, y3_2 = box[3] 

  i = int(x0_2)
  j = int(x1_2)
  a = image_sharp.shape[0]
  lol = img.shape[1]*20//1600        
  text_extraction = image_sharp[0:a, i-lol:j+lol]
  #text_extraction = image_sharp[0+30:a-30, i-lol:j+lol]
  cv2_imshow(text_extraction)

  for i in letter_write:
    text = str(i).upper()
  result = word_list['Word'].iloc[0].upper().find(text)

  x1, y1, x2, y2, x3, y3, x4, y4 = get_craft_coords(img_name)

  if (result==0):
    img[y1[0]-10:y3[0]+10, x1[0]+lol-text_extraction.shape[1]:x1[0]+lol] = text_extraction
  else:  
    img[y1[0]-10:y3[0]+10, x3[0]-2*lol:x3[0]-2*lol+text_extraction.shape[1]] = text_extraction

  return img
