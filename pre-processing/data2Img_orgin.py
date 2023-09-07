#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:36:07 2019

@author: yaching
"""


import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy.misc
import os

from os import walk
from os.path import basename
from os.path import join

#################################
import numpy as np

from decimal import Decimal

from PIL import Image

from skimage import io
import skimage.color


















def bTod(n, pre):
  '''
  把一個帶小數的二進位制數n轉換成十進位制
  小數點後面保留pre位小數
  '''
  string_number1 = str(n) #number1 表示二進位制數，number2表示十進位制數
  decimal = 0 #小數部分化成二進位制後的值
  flag = False 
  for i in string_number1: #判斷是否含小數部分
    if i == '.':
      flag = True
      break
  if flag: #若二進位制數含有小數部分
    string_integer, string_decimal = string_number1.split('.') #分離整數部分和小數部分
#    print("string_decimal: ", string_decimal)
    for i in range(len(string_decimal)):
        decimal += 2**(-i-1)*int(string_decimal[i]) #小數部分化成二進位制
    number2 = int(str(int(string_integer, 2))) + decimal
    return round(number2, pre)
  else: #若二進位制數只有整數部分
    return int(string_number1, 2)#若只有整數部分 直接一行程式碼二進位制轉十進位制 python還是騷 

def dTob(n, pre):
  '''
  把一個帶小數的十進位制數n轉換成二進位制
  小數點後面保留pre位小數
  '''
  string_number1 = '%f'%(n)
  string_number1 = str(string_number1) #number1 表示十進位制數，number2表示二進位制數 -->str
  
#  if 'e' in string_number1:
#      string_number1 = float(string_number1)
      
  flag = False 
  for i in string_number1: #判斷是否含小數部分
    if i == '.':
      flag = True
      break
  if flag:
    string_integer, string_decimal = string_number1.split('.') #分離整數部分和小數部分
    integer = int(string_integer)
    decimal = Decimal(str(n)) - integer
    l1 = [0,1]
    l2 = []
    decimal_convert = ""
    while True:
      if integer == 0: break
      x,y = divmod(integer, 2) #x為商，y為餘數
      l2.append(y)
      integer = x
    string_integer = ''.join([str(j) for j in l2[::-1]]) #整數部分轉換成二進位制
    i = 0
    while decimal != 0 and i < pre:
      result = int(decimal * 2)
      decimal = decimal * 2 - result
      decimal_convert = decimal_convert + str(result)
      i = i + 1
#      string_number2 = string_integer + '.' + decimal_convert
#    return float(string_number2)
#    print(string_integer)
    return string_integer, decimal_convert
 
   
  else: #若十進位制只有整數部分
    l1 = [0,1]
    l2 = []
    while True:
      if n == 0: break
      x,y = divmod(n, 2) #x為商，y為餘數
      l2.append(y)
      n = x
    string_number = ''.join([str(j) for j in l2[::-1]])
    return int(string_number)

####################################
def inputNmm_dec2bin(inputNum, pre):
#    np.set_printotions(suppress=True)
#    print("inputNum: ", inputNum)
    
    string_integer, decimal_convert = dTob(inputNum, pre)
    if(string_integer == ''):               #0
        tmp_bin = '000.' + decimal_convert
#        print(bTod(tmp_bin, pre))
        tmp_dec = bTod(tmp_bin, pre)
        tmp_bin = '000' + decimal_convert 
        return tmp_bin, tmp_dec
    elif(string_integer == '1'):               #1
        tmp_bin = '001.' + decimal_convert
#        print(bTod(tmp_bin, pre))
        tmp_dec = bTod(tmp_bin, pre)
        tmp_bin = '001' + decimal_convert 
        return tmp_bin, tmp_dec
    elif(string_integer == '10'):               #2
        tmp_bin = '010.' + decimal_convert
#        print(bTod(tmp_bin, pre))
        tmp_dec = bTod(tmp_bin, pre)
        tmp_bin = '010' + decimal_convert 
        return tmp_bin, tmp_dec
    elif(string_integer == '11'):               #3
        tmp_bin = '011.' + decimal_convert
#        print(bTod(tmp_bin, pre))
        tmp_dec = bTod(tmp_bin, pre)
        tmp_bin = '011' + decimal_convert 
        return tmp_bin, tmp_dec
    elif(string_integer == '100'):               #4
        tmp_bin = '100.' + decimal_convert
        tmp_dec = bTod(tmp_bin, pre)
        tmp_bin = '100' + decimal_convert 
        return tmp_bin, tmp_dec
    elif(string_integer == '101'):               #5
        tmp_bin = '101.' + decimal_convert
#        print(bTod(tmp_bin, pre))
        tmp_dec = bTod(tmp_bin, pre)
        tmp_bin = '101' + decimal_convert 
        return tmp_bin, tmp_dec
    elif(string_integer == '110'):               #6
        tmp_bin = '110.' + decimal_convert
#        print(bTod(tmp_bin, pre))
        tmp_dec = bTod(tmp_bin, pre)
        tmp_bin = '110' + decimal_convert 
        return tmp_bin, tmp_dec
    elif(string_integer == '111'):               #7
        tmp_bin = '111.' + decimal_convert
#        print(bTod(tmp_bin, pre))
        tmp_dec = bTod(tmp_bin, pre)
        tmp_bin = '111' + decimal_convert 
        return tmp_bin, tmp_dec    
 


#####################################
pre = 3


#########matrix#########
def matrix1():
    data1_len = len(data1)/(64*64)
    matrix1 = [[0 for z1 in range(64)] for a1 in range(64)] 
    for z1 in range(0, int(data1_len), 1):
        for i1 in range(0, 64, 1):
            for j1 in range(0, 64, 1):
            
                tmp1 = data1[j1+(i1*64)+(z1*64*64)]    #data1 input matrix1
                tmp2 = tmp1[0]
                tmp2 = float(tmp2)
                if(tmp2 < 0):
                    tmp2 = -(tmp2)
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp2, pre)
                    matrix1[i1][j1] = -(tmp_dec)
#                    print(matrix1[i1][j1])
                else:
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp2, pre)
                    matrix1[i1][j1] = tmp_dec
#                    print(matrix1[i1][j1])
        
#        scipy.misc.imsave(str(save_path) + str(file)+'/'+str(list1_name)+str(z1+1)+'.jpg', matrix1)    #matrix1 to image(jpg)
        io.imsave(str(save_path) + str(file)+'/'+str(list1_name)+str(z1+1)+'.tif', np.float32(matrix1))
        

        
def matrix2():
    data2_len = len(data2)/(64*64)
    matrix2 = [[0 for z2 in range(64)] for a2 in range(64)]     
    for z2 in range(0, int(data2_len), 1):
        for i2 in range(0, 64, 1):
            for j2 in range(0, 64, 1):

                tmp3 = data2[j2+(i2*64)+(z2*64*64)]    #data2 input matrix2
                tmp4 = tmp3[0]
                tmp4 = float(tmp4)
                matrix2[i2][j2] = tmp4
                print("tmp4 here",tmp4)
                if(tmp4 < 0):
                    tmp4 = -(tmp4)
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp4, pre)
                    matrix2[i2][j2] = -(tmp_dec)
#                    print(matrix2[i2][j2])
                else:
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp4, pre)
                    matrix2[i2][j2] = tmp_dec
#                    print(matrix2[i2][j2])
                 
#        scipy.misc.imsave(str(save_path)+str(file)+'/'+str(list2_name)+str(z2+1)+'.jpg', matrix2)    #matrix2 to image(jpg)
        io.imsave(str(save_path) + str(file)+'/'+str(list2_name)+str(z2+1)+'.tif', np.float32(matrix2))
    
def matrix3(): 
    data3_len = len(data3)/(64*64)    
    matrix3 = [[0 for z3 in range(64)] for a3 in range(64)]     
    for z3 in range(0, int(data3_len), 1):
        for i3 in range(0, 64, 1):
            for j3 in range(0, 64, 1):
                
                tmp5 = data3[j3+(i3*64)+(z3*64*64)]    #data3 input matrix3
                tmp6 = tmp5[0]
                tmp6 = float(tmp6)
                matrix3[i3][j3] = tmp6
    
                if(tmp6 < 0):
                    tmp6 = -(tmp6)
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp6, pre)
                    matrix3[i3][j3] = -(tmp_dec)
#                    print(matrix3[i3][j3])
                else:
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp6, pre)
                    matrix3[i3][j3] = tmp_dec
#                    print(matrix3[i3][j3])
                    
#        scipy.misc.imsave(str(save_path)+str(file)+'/'+str(list3_name)+str(z3+1)+'.jpg', matrix3)    #matrix3 to image(jpg)
        io.imsave(str(save_path) + str(file)+'/'+str(list3_name)+str(z3+1)+'.tif', np.float32(matrix3))
        
def matrix4(): 
    data4_len = len(data4)/(64*64)  
    matrix4 = [[0 for z4 in range(64)] for a4 in range(64)]     
    for z4 in range(0, int(data4_len), 1):
        for i4 in range(0, 64, 1):
            for j4 in range(0, 64, 1):
                
                tmp7 = data4[j4+(i4*64)+(z4*64*64)]    #data3 input matrix4
                tmp8 = tmp7[0]
                tmp8 = float(tmp8)
                matrix4[i4][j4] = tmp8
    
                if(tmp8 < 0):
                    tmp8 = -(tmp8)
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp8, pre)
                    matrix4[i4][j4] = -(tmp_dec)
#                    print(matrix4[i4][j4])
                else:
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp8, pre)
                    matrix4[i4][j4] = tmp_dec
#                    print(matrix4[i4][j4])
                   
#        scipy.misc.imsave(str(save_path)+str(file)+'/'+str(list4_name)+str(z4+1)+'.jpg', matrix4)    #matrix4 to image(jpg)
        io.imsave(str(save_path) + str(file)+'/'+str(list4_name)+str(z4+1)+'.tif', np.float32(matrix4))
        
        
def matrix5(): 
    data5_len = len(data5)/(64*64)    
    matrix5 = [[0 for z5 in range(64)] for a5 in range(64)]     
    for z5 in range(0, int(data5_len), 1):
        for i5 in range(0, 64, 1):
            for j5 in range(0, 64, 1):
                
                tmp9 = data5[j5+(i5*64)+(z5*64*64)]    #data3 input matrix4
                tmp10 = tmp9[0]
                tmp10 = float(tmp10)
                matrix5[i5][j5] = tmp10
    
                if(tmp10 < 0):
                    tmp10 = -(tmp10)
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp10, pre)
                    matrix5[i5][j5] = -(tmp_dec)
#                    print(matrix5[i5][j5])
                else:
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp10, pre)
                    matrix5[i5][j5] = tmp_dec
#                    print(matrix5[i5][j5])
          
#        scipy.misc.imsave(str(save_path)+str(file)+'/'+str(list5_name)+str(z5+1)+'.jpg', matrix5)    #matrix4 to image(jpg)
        io.imsave(str(save_path) + str(file)+'/'+str(list5_name)+str(z5+1)+'.tif', np.float32(matrix5))

        
def matrix6(): 
    data6_len = len(data6)/(64*64)    
    matrix6 = [[0 for z6 in range(64)] for a6 in range(64)]     
    for z6 in range(0, int(data6_len), 1):
        for i6 in range(0, 64, 1):
            for j6 in range(0, 64, 1):
                
                tmp11 = data5[j6+(i6*64)+(z6*64*64)]    #data3 input matrix4
                tmp12 = tmp11[0]
                tmp12 = float(tmp12)
                matrix6[i6][j6] = tmp12
    
                if(tmp12 < 0):
                    tmp12 = -(tmp12)
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp12, pre)
                    matrix6[i6][j6] = -(tmp_dec)
#                    print(matrix6[i6][j6])
                else:
                    tmp_bin, tmp_dec = inputNmm_dec2bin(tmp12, pre)
                    matrix6[i6][j6] = tmp_dec
#                    print(matrix6[i6][j6])
         
#        scipy.misc.imsave(str(save_path)+str(file)+'/'+str(list6_name)+str(z6+1)+'.jpg', matrix6)    #matrix4 to image(jpg)
        io.imsave(str(save_path) + str(file)+'/'+str(list6_name)+str(z6+1)+'.tif', np.float32(matrix6))

if __name__=='__main__':
#    address = '/home/yaching/tensorflow/CNC_Data/48k_Drive_End_Bearing_Fault_Data/'
#    save_path = "CNC_Data_Image_orgin/48k_Drive_End_Bearing_Fault_Data/"
    files_path=["12k_DE", "12k_FE", "48k_DE", "Normal"]
    for i in files_path:
        address = "/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/CWRU_data_matlab/" + str(i) + "/"
        save_path = "/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/CWRU_data_image7bit/" + str(i) + "/"
        os.mkdir(save_path)
                    
        for root, dirs, files in walk(address):   
            for f in files:
            
                data_name = join(root, f)  
                base = os.path.basename(data_name)
                file = os.path.splitext(base)[0]
                os.mkdir(str(save_path)+file)
    
                file_name = str(address)+str(f)
                data = scio.loadmat(file_name)###matlab data


                mat_array = scipy.io.whosmat(file_name)
    #            print(mat_array)
    
                if len(mat_array) == 2:
                    if mat_array[0][1][0] > 1:
                        list1_name = mat_array[0][0]
                        data1 = data[list1_name]
                        print(list1_name)
                        matrix1()
    
                    if mat_array[1][1][0] > 1:
                        list2_name = mat_array[1][0]
                        data2 = data[list2_name]
                        print(list2_name)
                        matrix2()
    
    
                if len(mat_array) == 3:
                    if mat_array[0][1][0] > 1:
                        list1_name = mat_array[0][0]
                        data1 = data[list1_name]
                        print(list1_name)
                        matrix1()
    
                    if mat_array[1][1][0] > 1:
                        list2_name = mat_array[1][0]
                        data2 = data[list2_name]
                        print(list2_name)
                        matrix2()
                    
                    if mat_array[2][1][0] > 1:
                        list3_name = mat_array[2][0]
                        data3 = data[list3_name]
                        print(list3_name)
                        matrix3()  
    
    
                elif len(mat_array) == 4:
                    if mat_array[0][1][0] > 1:
                        list1_name = mat_array[0][0]
                        data1 = data[list1_name]
                        print(list1_name)
                        matrix1()
    
                    if mat_array[1][1][0] > 1:
                        list2_name = mat_array[1][0]
                        data2 = data[list2_name]
                        print(list2_name)
                        matrix2()
                    
                    if mat_array[2][1][0] > 1:
                        list3_name = mat_array[2][0]
                        data3 = data[list3_name]
                        print(list3_name)
                        matrix3()  
                    
                    if mat_array[3][1][0] > 1:
                        list4_name = mat_array[3][0]
                        data4 = data[list4_name]
                        print(list4_name)
                        matrix4()
                                    
    
                elif len(mat_array) == 5:
                    if mat_array[0][1][0] > 1:
                        list1_name = mat_array[0][0]
                        data1 = data[list1_name]
                        print(list1_name)
                        matrix1()
    
                    if mat_array[1][1][0] > 1:
                        list2_name = mat_array[1][0]
                        data2 = data[list2_name]
                        print(list2_name)
                        matrix2()
                    
                    if mat_array[2][1][0] > 1:
                        list3_name = mat_array[2][0]
                        data3 = data[list3_name]
                        print(list3_name)
                        matrix3()  
                    
                    if mat_array[3][1][0] > 1:
                        list4_name = mat_array[3][0]
                        data4 = data[list4_name]
                        print(list4_name)
                        matrix4()
                                    
                    if mat_array[4][1][0] > 1:
                        list5_name = mat_array[4][0]
                        data5 = data[list5_name]
                        print(list5_name)
                        matrix5()
                        
                        
                elif len(mat_array) == 6:
                    if mat_array[0][1][0] > 1:
                        list1_name = mat_array[0][0]
                        data1 = data[list1_name]
                        print(list1_name)
                        matrix1()
    
                    if mat_array[1][1][0] > 1:
                        list2_name = mat_array[1][0]
                        data2 = data[list2_name]
                        print(list2_name)
                        matrix2()
                    
                    if mat_array[2][1][0] > 1:
                        list3_name = mat_array[2][0]
                        data3 = data[list3_name]
                        print(list3_name)
                        matrix3()  
                    
                    if mat_array[3][1][0] > 1:
                        list4_name = mat_array[3][0]
                        data4 = data[list4_name]
                        print(list4_name)
                        matrix4()
                                    
                    if mat_array[4][1][0] > 1:
                        list5_name = mat_array[4][0]
                        data5 = data[list5_name]
                        print(list5_name)
                        matrix5()
                                     
                    if mat_array[5][1][0] > 1:
                        list6_name = mat_array[5][0]
                        data6 = data[list6_name]
                        print(list6_name)
                        matrix6()                   
                else: 
                    print('other')
                    print('file',len(mat_array))
  
   
print('end')
    





            






   
