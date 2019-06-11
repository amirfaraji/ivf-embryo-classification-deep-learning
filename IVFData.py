########## LIBRARIES ##########
import time
import sys
import os
from scipy.misc import imresize
import numpy as np
from PIL import Image
import xlwt
import xlrd
from keras import backend as K
import matplotlib.pyplot as plt

class IVFData: 
    def __init__(self, img_wid, img_hgt):
        self.nof2012 =  76 # 61
        self.nof2013 =	73 # 53
        self.nof2014 =	142 # 122
        self.nof2015 =	137
        self.nof2016 =	86 # 59
        self.total_data = self.nof2012 + self.nof2013 + self.nof2014 + self.nof2015 + self.nof2016 
        if K.image_data_format == 'channel_first':
            self.input_size  = (self.total_data, 3, img_wid, img_hgt)
        else:
            self.input_size  = (self.total_data, img_wid, img_hgt, 3)

    def setup_data(self):
        y_data = np.zeros((self.total_data,))
        x_data = np.zeros(self.input_size)
        y_data[0:self.nof2012], x_data[0:self.nof2012,:,:,:] = self.readxls(sheetname = '2012', bookname = 'Data (anonymized)new.xlsx')
        y_data[self.nof2012:(self.nof2012+self.nof2013)], x_data[self.nof2012:(self.nof2012+self.nof2013),:,:,:] = self.readxls(sheetname = '2013', bookname = 'Data (anonymized)new.xlsx')
        y_data[(self.nof2012+self.nof2013):(self.nof2012+self.nof2013+self.nof2014)], x_data[(self.nof2012+self.nof2013):(self.nof2012+self.nof2013+self.nof2014),:,:,:] = self.readxls(sheetname = '2014', bookname = 'Data (anonymized)new.xlsx')
        y_data[(self.nof2012+self.nof2013+self.nof2014):(self.nof2012+self.nof2013+self.nof2014+self.nof2015)], x_data[(self.nof2012+self.nof2013+self.nof2014):(self.nof2012+self.nof2013+self.nof2014+self.nof2015),:,:,:] = self.readxls(sheetname = '2015', bookname = 'Data (anonymized)new.xlsx')
        y_data[(self.nof2012+self.nof2013+self.nof2014+self.nof2015):], x_data[(self.nof2012+self.nof2013+self.nof2014+self.nof2015):,:,:,:]  = self.readxls(sheetname = '2016', bookname = 'Data (anonymized)new.xlsx')

        return y_data, x_data


    def readxls(self, bookname = 'output.xls', sheetname = 'train'):
        """Read the 1st and 3rd column of the Excel file to find the outcome of IVF"""
        book = xlrd.open_workbook(bookname)
        worksheet = book.sheet_by_name(sheetname)
        outcome = []
        names = []
        for value in worksheet.col_values(1):
            if isinstance(value, float):
                outcome.append(value)

        for name in worksheet.col_values(0):
            if isinstance(name, str):
                names.append(name)
        names.pop(0) # Remove the Table heading

        print("Loading Images for years 2012-2016")
        imagedata = self.load_image_data((len(names), self.input_size[1], self.input_size[2], self.input_size[3]), names, outcome, sheetname + " images")

        return outcome, imagedata

    def load_image_data(self, input_shape, filenames, labels, dir = ''):
        """Load Image data to"""
        toolbar_width = 10
        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

        data = np.zeros(input_shape)
        count = 0
        for file in filenames: 
            im = Image.open("./crop/" + dir + "/" + file + ".BMP")
            p = np.array(im)
            data[count, :, :, :] = self.rescaling(p, (input_shape[1],input_shape[2], input_shape[3]))
            #self.show_image(p, str(labels[count]) + ": " + file)
            count += 1
            if (count % (input_shape[0] // toolbar_width) == 0):
                sys.stdout.write("-")
                sys.stdout.flush()

        sys.stdout.write("\n")
        print("Done")
        
        return data

    def rescaling(self, img, input_shape):
        res = imresize(img, size=input_shape, interp='bilinear')
        return res

    def show_image(self, img, title):
        plt.title(title)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.show()
        pass

    def split_k_folds(self, x_data, y_data, fold):
        """ Splits Data into 70-15-15"""
        # Find Indices for class 0 or 1
        known_birth_indices = [i for i, x in enumerate(y_data) if x != 2]


        # Pop all unknown outcomes
        new_x_data = x_data[known_birth_indices,:,:,:]
        new_y_data = y_data[known_birth_indices]

        birth_indices = [i for i, x in enumerate(new_y_data) if x == 1]
        non_birth_indices = [i for i, x in enumerate(new_y_data) if x == 0]


        # Split data to 50%-50% into Train-Valid-Test 70-15-15
        from math import ceil, floor
        num_of_train = ceil(len(new_y_data)*0.70)
        num_of_valid = floor(len(new_y_data)*0.15)
        num_of_test = floor(len(new_y_data)*0.15)
        # print(num_of_train,num_of_valid,num_of_test)
        
        all_index = list(range(len(new_x_data)))

        # Random Seed and Shuffle
        np.random.seed(0)
        np.random.shuffle(all_index)

        all_index = np.roll(all_index, -fold * num_of_test)

        y_train_set = new_y_data[all_index[0:num_of_train]]
        x_train_set = new_x_data[all_index[0:num_of_train], :, :, :]

        y_valid_set = new_y_data[all_index[num_of_train:num_of_train + num_of_valid]]
        x_valid_set = new_x_data[all_index[num_of_train:num_of_train + num_of_valid], :, :, :]

        y_test_set = new_y_data[all_index[-num_of_test:]]
        x_test_set = new_x_data[all_index[-num_of_test:], :, :, :]
        #Make 50-50
        x_valid_set, y_valid_set = self.split_50_50(x_valid_set, y_valid_set)
        x_test_set, y_test_set = self.split_50_50(x_test_set, y_test_set)

        return y_train_set, y_valid_set, y_test_set, x_train_set, x_valid_set, x_test_set


    def split_50_50(self, x_data, y_data):
        x = x_data.tolist()
        y = y_data.tolist()
        test_0 = [i for i, x in enumerate(y_data) if x == 0]
        test_1 = [i for i, x in enumerate(y_data) if x == 1]

        if len(test_0) < len(test_1):
            indexes = test_1[-(len(test_1) - len(test_0)):]
            for idx in indexes[len(indexes)::-1]:
                y.pop(idx)
                x.pop(idx)
        elif len(test_0) > len(test_1):
            indexes = test_0[-(len(test_0)-len(test_1)):]
            for idx in indexes[len(indexes)::-1]:
                y.pop(idx)
                x.pop(idx)
        else:
            print("Already 50-50")
        return np.asarray(x), np.asarray(y)
