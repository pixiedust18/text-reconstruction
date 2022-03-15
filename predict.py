"""
SRNet - Editing Text in the Wild
Data prediction.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Krisha Bhambani
"""

from model import SRNet
import numpy as np
import os
import cfg
from utils import *
from datagen import srnet_datagen, get_input_data
import argparse
from craft_text_detector import Craft
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt 

from reconstruction_utils import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

            
print_log('model compiling start.', content_color = PrintColor['yellow'])
model = SRNet(shape = cfg.data_shape, name = 'predict')
print_log('model compiled.', content_color = PrintColor['yellow'])
load_wt_flag = False

def SRNet_execute(i_s_path, i_t_path, checkpoint, save_dir, input_dir=None):
    with model.graph.as_default():
        with tf.Session() as sess:
            # load pretrained weights
            saver = tf.train.Saver(tf.global_variables())
            #if not load_wt_flag:
            print_log('weight loading start.', content_color = PrintColor['yellow'])
            saver.restore(sess, checkpoint)
            print_log('weight loaded.', content_color = PrintColor['yellow'])
            load_wt_flag = True

            # predict
            print_log('predicting start.', content_color = PrintColor['yellow'])
            if input_dir is None:
                i_s = cv2.imread(i_s_path)
                i_t = cv2.imread(i_t_path)
                
                s1, s2, _ = i_s.shape
                t1, t2, _ = i_t.shape
                h = max(s1, t1)
                w = max(s2, t2)
                if (i_s.shape != i_t.shape):
                    i_s = cv2.resize(i_s, (w, h))
                    i_t = cv2.resize(i_t, (w, h))

                o_sk, o_t, o_b, o_f = model.predict(sess, i_t, i_s)
                
                cv2.imwrite(os.path.join(save_dir, 'result.png'), o_f)
                save_mode = 1
                if save_mode == 1:
                    cv2.imwrite(os.path.join(save_dir, 'result_sk.png'), o_sk)
                    cv2.imwrite(os.path.join(save_dir, 'result_t.png'), o_t)
                    cv2.imwrite(os.path.join(save_dir, 'result_b.png'), o_b)
            else:
                predict_data_list(model, sess, save_dir, get_input_data(input_dir), mode = 1)
            print_log('predicting finished.', content_color = PrintColor['yellow'])

def main():
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu', help = 'gpu id', default = None)
    parser.add_argument('--img_name', help = 'input original text patch')
    parser.add_argument('--img_dir', help = 'input standard text patch')
    parser.add_argument('--words_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix',
                        default = '/content/gdrive/MyDrive/BE Project/words.txt')
    parser.add_argument('--save_dir', help = 'Directory to save result', default = cfg.predict_result_dir)
    parser.add_argument('--checkpoint', help = 'tensorflow ckpt', default = cfg.predict_ckpt_path)
    args = parser.parse_args()


    assert args.save_dir is not None
    assert args.checkpoint is not None
    
    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if not os.path.exists('results/'):
      print("SF")
      os.mkdir('results/')
    if not os.path.exists('cur_test/'):
      print("SF")
      os.mkdir('cur_test/')
    if not os.path.exists('craft_outputs/'):
      os.mkdir('craft_outputs/')

     
    img_path = args.img_dir
    img_name = args.img_name
    img= cv2.imread(img_path)
    #orig = cv2.imread(img_path)
    #orig_copy = cv2.imread(img_path)

    print(img_name)
    if abs(img.shape[0] - img.shape[1])>150:
      if (img.shape[0]<img.shape[1]):
        s = img.shape[0] 
      else:
        s = img.shape[1]
      img = cv2.resize(img, (s+150, s))
      
    img_path = 'cur_test/'+img_name+'.jpg'
    cv2.imwrite(img_path, img)

    #craft------------------------------------------------------
    # set image path and export folder directory
    image = img # can be filepath, PIL image or numpy array
    output_dir = 'craft_outputs/'

    # create a craft instance
    if args.gpu:
        cuda = True
    else:
        cuda = False
    craft = Craft(output_dir=output_dir, crop_type="poly", cuda=cuda)

    # apply craft text detection and export detected regions to output directory
    prediction_result = craft.detect_text(image)

    # unload models from ram/gpu
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    #---------------------------------------------------------------

    prediction_words = read_text(img_path)
    #prediction_words = read_text(img_path)
    words = load_vocabulary(args.words_dir)
    incomplete_word = prediction_words[0][0][0]
    flag = False
    start = prediction_words[0][0][0]
    end = prediction_words[0][-1][0]
    for i in range(len(prediction_words[0])-1):
      incomplete_word+=prediction_words[0][i+1][0]
      flag = True
    word_list = my_autocorrect(words, incomplete_word, flag, start, end)
    letter_write = missing_letters(word_list['Word'].iloc[0].lower(), incomplete_word)
    col = np.mean(img)
    pref_col = 0
    if col<150:
        pref_col = 1
    elif col<180:
        pref_col = 2
    else:
        pref_col = 3

    final_res = np.array([])
    for i in range(4):
        non_stylised_gen(img_name, img_path, letter_write, i)
        SRNet_execute('cur_test/i_s.png', 'cur_test/i_t.png', args.checkpoint, save_dir='results/') ####
        result_path = 'results/result.png'
        pred_letter, res = final_integration(img_name,img_path, letter_write, word_list, i)
        #plt.imshow(res)
        cv2.imwrite(str(i)+'.png', res)

        print(pred_letter)
        x = letter_write.pop()
        letter_write.add(x)
        if not final_res.any() and pref_col == i:
            final_res = res
        if i==3 and not final_res.any():
            final_res = res
        if pred_letter.lower()==x.lower():
            final_res = res
            break
        
    cv2_imshow(final_res)
    cv2.imwrite('results/reconstructed_text.png', res)
    
                
if __name__ == '__main__':
    main()
