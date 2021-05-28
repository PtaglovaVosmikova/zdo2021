import numpy as np
import scipy
import scipy.misc
from scipy import stats
from scipy import ndimage
import numpy as np
import urllib
import skimage
import skimage.color
import skimage.io
import skimage.exposure
import skimage.data as dt
import skimage.morphology
import matplotlib.pyplot as plt
from skimage import data
import pylab as pl
from skimage import transform as tf
from PIL import Image, ImageFilter
from skimage.feature import canny
import glob
from . import podpurne_funkce

class VarroaDetector():
    def __init__(self):
        pass

    def predict(self, data):
        """
        :param data: np.ndarray with shape [pocet_obrazku, vyska, sirka, barevne_kanaly]
        :return: shape [pocet_obrazku, vyska, sirka], 0 - nic, 1 - varroa destructor
        """
        obr = "d:\images\Original_1200_image (1).jpg"
        #obr_k = "d:\images\Original_1200_image (1) kopie.jpg"

        img = skimage.io.imread(obr)
        #img_k = skimage.io.imread(obr_k)
        umg_hsv = skimage.color.rgb2hsv(img)

        obr_v = umg_hsv[:,:,2]
        #obr_k_vyr = img_k

        img_vyrez_v = obr_v
        gaussian_img_v = ndimage.gaussian_filter(img_vyrez_v, sigma=2)
        img_tr_v = (gaussian_img_v < 0.32)
        imgQR_tr = (img_vyrez_v < 0.45)

        kernel_img3 = skimage.morphology.square(5).astype(np.uint8)
        imgQR_tr_op = skimage.morphology.binary_opening(imgQR_tr, kernel_img3)
        imgQR_tr_rso = skimage.morphology.remove_small_objects(imgQR_tr_op, 80)

        imgQR_label = skimage.morphology.label(imgQR_tr_rso > 0)
        propsQR = skimage.measure.regionprops(imgQR_label+1)
        
        # POKUS O NALEZENÍ ČTVERCŮ V QR ------------------------------------------------------
        ctverec_max1 = -1
        ctverec_max_id1 = -1
        ctverec_max2 = -1
        ctverec_max_id2 = -1
        ctverec_max3 = -1
        ctverec_max_id3 = -1

        for i in range(len(propsQR)):
            a = propsQR[i].major_axis_length
            b = propsQR[i].minor_axis_length
            area = propsQR[i].area
            pravouhlost = area/(a*b)
            
            if pravouhlost > 0.7:
                if b > a - 2:
                    if ctverec_max1 < a:
                        #print ("id: ", i)
                        ctverec_max_id3 = ctverec_max_id2
                        ctverec_max3 = ctverec_max2
                        ctverec_max2 = ctverec_max1
                        ctverec_max_id2 = ctverec_max_id1
                        ctverec_max1 = a
                        ctverec_max_id1 = i
                    elif ctverec_max2 < a:
                        ctverec_max_id3 = ctverec_max_id2
                        ctverec_max3 = ctverec_max2
                        ctverec_max2 = a
                        ctverec_max_id2 = i
                    elif ctverec_max3 < a:
                        ctverec_max3 = a
                        ctverec_max_id3 = i
                        
        a1 = propsQR[ctverec_max_id1].area
        a2 = propsQR[ctverec_max_id2].area
        a3 = propsQR[ctverec_max_id3].area
        
        if a2/a1 < 0.8:
            print("jen 1")
            prum_a = a1
        elif a2/a1 > 0.8 and a3/a2 <0.8:
            print("jen 2")
            prum_a = (a1 + a2) / 2
        else:
            print("všechny 3")
            prum_a = (a1 + a2 + a3) / 3
        
        print ("průměrná area: ", prum_a)
        
        # -----------------------------------------------------------------------------------------
        small = (prum_a/4)/2.5

        kernel_img2 = skimage.morphology.square(3).astype(np.uint8)
        img_tr_er = skimage.morphology.binary_erosion(img_tr_v, kernel_img2)
        img_tr_rso = skimage.morphology.remove_small_objects(img_tr_er, small)
        img_tr_di = skimage.morphology.binary_dilation(img_tr_rso, kernel_img2)
        
        img_label = skimage.morphology.label(img_tr_rso > 0)
        props = skimage.measure.regionprops(img_label+1)
        
        # POKUS O NALEZENÍ KLEŠTÍKŮ --------------------------------------------------
        max_v_kl = prum_a/4
        min_v_kl = max_v_kl/2
        min_v_kl_f = max_v_kl/2.2
        
        hezky_klestik = 0
        fuj_klestik = 0
        vysledek = np.zeros_like(img_vyrez_v)
        #vys = obr_k_vyr
        #vysledek = np.zeros_like(img2_obarveny)
        
        id_hezky = []
        id_fuj = []
        
        for i in range(len(props)):
            nekomp = props[i].perimeter**2 / props[i].area
            #a = props[i].major_axis_length
            #b = props[i].minor_axis_length
            #obvod = props[i].perimeter
            area = props[i].area
            #nekomp > 12 and nekomp < 14:
            
            if min_v_kl < area < max_v_kl: 
                if 12 < nekomp < 13.8:
                    hezky_klestik = hezky_klestik + 1
                    id_hezky.append(i)
                    vysledek[img_label==i] = 1
                    #vys[img_label==i] = [255,0,0]
                elif 11 < nekomp < 15:
                    fuj_klestik = fuj_klestik + 1
                    id_fuj.append(i)
                    #vysledek[img_label==i] = 0.5
                    #vys[img_label==i] = [0,255,0]
            elif min_v_kl_f < area < max_v_kl: 
                if 11 < nekomp < 15:
                    fuj_klestik = fuj_klestik + 1
                    id_fuj.append(i)
                    #vysledek[img_label==i] = 0.5
                    #vys[img_label==i] = [0,0,255]

            
        print ("Počet hezký: ", hezky_klestik) 
        print ("Počet fuj: ", fuj_klestik)
        
        print ("Max velikost: ", max_v_kl)
        print ("Min velikost: ", min_v_kl)

        print("ahoj")
        output = np.zeros_like(data)
        return output
