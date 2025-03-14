import os
import random
import tkinter
from tkinter import *
from tkinter import Tk
from tkinter.filedialog import askopenfile
import cv2
import pandas as pd
import numpy as np
# import openpyxl
from PIL import ImageTk
# from openpyxl.chart import Reference, Series, BarChart3D
# from prettytable import PrettyTable
from tkinter import messagebox
from tkinter import Tk, filedialog
from PIL import Image, ImageTk
import time
import csv
from matplotlib import pyplot as plt
from skimage import io, filters
from CLDC import config as cfg

from CLDC.Pre_processing.AdaptiveMedianFilter import AdaptiveMedianFilter
from CLDC.Pre_processing.CLAHE import CLAHE

from CLDC.Feature_Extraction.CNNSE_MIV3 import CNNSE_MIV3
from CLDC.Feature_Extraction.CNN import CNN
from CLDC.Feature_Extraction.Inception_V3 import Inception_V3
from CLDC.Feature_Extraction.Handcraft_Features import Handcraft_Features

from CLDC.CLD_Classification.Existing_LSTM import Existing_LSTM
from CLDC.CLD_Classification.Existing_ANN import Existing_ANN
from CLDC.CLD_Classification.Proposed_MLMLP import Proposed_MLMLP


class Main_GUI:
    boolDSNRAMF = False
    boolDSCECLAHE = False
    boolDSFE = False
    boolDSFF = False
    boolDSSplitting = False
    boolTraining = False
    boolTesting = False

    boolImageRead = False
    boolImageNoiseRemoval = False
    boolImageContrastEnhancement = False
    boolImageFeatureExtraction = False
    boolImageFeatureFusion = False
    boolImageClassification = False

    iptrdata = []
    iptsdata = []
    iptrcls = []
    iptscls = []

    iptr_feature = []
    ipts_feature = []

    training = 80
    testing = 20

    def __init__(self, root):
        self.file_path = StringVar()
        self.noofnodes = StringVar()

        self.LARGE_FONT = ("Algerian", 16)
        self.text_font = ("Constantia", 15)
        self.text_font1 = ("Constantia", 10)

        self.frame_font = ("", 9)
        self.frame_process_res_font = ("", 12)
        self.root = root
        self.feature_value = StringVar()

        label_heading = tkinter.Label(root, text="Cotton Leaf Disease Classification using Hybrid Deep Learning with Multi-Loss Function", fg="deep pink", bg="azure3", font=self.LARGE_FONT)
        label_heading.place(x=0, y=0)

        self.label_frame_cld_dataset = LabelFrame(root, text="Cotton Leaf Disease Dataset", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_frame_cld_dataset.place(x=10, y=25, width=180, height=50)
        self.entry_cld_dataset = Entry(root, font=self.frame_font)
        self.entry_cld_dataset.place(x=20, y=45, width=160, height=25)
        self.entry_cld_dataset.insert(INSERT, "..\\\\Dataset\\\\")
        self.entry_cld_dataset.configure(state="disabled")

        self.label_frame_dataset_preprocessing = LabelFrame(root, text="Pre-Processing", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_frame_dataset_preprocessing.place(x=200, y=25, width=150, height=50)
        self.btn_dataset_preprocessing_noise_removal = Button(root, text="NR AMF", bg="deep sky blue", fg="#fff", font=self.text_font1, width=7, command = self.dataset_preprocessing_noise_removal)
        self.btn_dataset_preprocessing_noise_removal.place(x=210, y=45)
        self.btn_dataset_preprocessing_contrast_enhancement = Button(root, text="CE CLAHE", bg="deep sky blue", fg="#fff", font=self.text_font1, width=7, command = self.dataset_preprocessing_contrast_enhancement)
        self.btn_dataset_preprocessing_contrast_enhancement.place(x=280, y=45)

        self.label_frame_dataset_feature_extraction = LabelFrame(root, text="Feature Extraction", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_frame_dataset_feature_extraction.place(x=360, y=25, width=110, height=50)
        self.btn_dataset_feature_extraction = Button(root, text="Proceed", bg="deep sky blue", fg="#fff", font=self.text_font1, width=10, command=self.dataset_feature_extraction)
        self.btn_dataset_feature_extraction.place(x=370, y=45)

        self.label_frame_dataset_feature_fusion = LabelFrame(root, text="Feature Fusion", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_frame_dataset_feature_fusion.place(x=480, y=25, width=110, height=50)
        self.btn_dataset_feature_fusion = Button(root, text="Proceed", bg="deep sky blue", fg="#fff", font=self.text_font1, width=10, command=self.dataset_feature_fusion)
        self.btn_dataset_feature_fusion.place(x=490, y=45)

        self.label_frame_dataset_splitting = LabelFrame(root, text="Dataset Splitting", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_frame_dataset_splitting.place(x=600, y=25, width=110, height=50)
        self.btn_dataset_splitting = Button(root, text="Proceed", bg="deep sky blue", fg="#fff", font=self.text_font1, width=10, command=self.dataset_splitting)
        self.btn_dataset_splitting.place(x=610, y=45)

        self.label_frame_cld_classification = LabelFrame(root, text="CLD Classification", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_frame_cld_classification.place(x=720, y=25, width=150, height=50)
        self.btn_cld_training = Button(root, text="Training", bg="deep sky blue", fg="#fff", font=self.text_font1, width=6, command=self.cld_training)
        self.btn_cld_training.place(x=730, y=45)
        self.btn_cld_testing = Button(root, text="Testing", bg="deep sky blue", fg="#fff", font=self.text_font1, width=6, command=self.cld_testing)
        self.btn_cld_testing.place(x=800, y=45)

        self.label_tables_graphs = LabelFrame(root, text="Tables & Graphs", bg="azure3", fg="#FF6A6A", font=self.frame_font)
        self.label_tables_graphs.place(x=880, y=25, width=120, height=50)
        self.btn_tables_graphs = Button(root, text="Generate", bg="deep sky blue", fg="#fff", width=12, command=self.tables_graphs)
        self.btn_tables_graphs.place(x=890, y=45)
######################################################################################
        # Horizontal (x) Scroll bar
        self.xscrollbar = Scrollbar(root, orient=HORIZONTAL)
        self.xscrollbar.pack(side=BOTTOM, fill=X)
        # Vertical (y) Scroll Bar
        self.yscrollbar = Scrollbar(root)
        self.yscrollbar.pack(side=RIGHT, fill=Y)

        self.label_output_frame = LabelFrame(root, text="Process Window", bg="azure3", fg="#0000FF", font=self.frame_process_res_font)
        self.label_output_frame.place(x=170, y=90, width=570, height=280)
        # Text Widget
        self.data_textarea_process = Text(root, wrap=WORD, xscrollcommand=self.xscrollbar.set, yscrollcommand=self.yscrollbar.set)
        self.data_textarea_process.pack()
        # Configure the scrollbars
        self.xscrollbar.config(command=self.data_textarea_process.xview)
        self.yscrollbar.config(command=self.data_textarea_process.yview)
        self.data_textarea_process.place(x=180, y=110, width=550, height=250)
        self.data_textarea_process.configure(state="disabled")

        self.label_output_frame = LabelFrame(root, text="Result Window", bg="azure3", fg="#0000FF", font=self.frame_process_res_font)
        self.label_output_frame.place(x=750, y=90, width=250, height=500)
        # Text Widget
        self.data_textarea_result = Text(root, wrap=WORD, xscrollcommand=self.xscrollbar.set, yscrollcommand=self.yscrollbar.set)
        self.data_textarea_result.pack()
        # Configure the scrollbars
        self.xscrollbar.config(command=self.data_textarea_result.xview)
        self.yscrollbar.config(command=self.data_textarea_result.yview)
        self.data_textarea_result.place(x=760, y=110, width=230, height=470)
        self.data_textarea_result.configure(state="disabled")

#################################################################################
        self.label_image_testing = LabelFrame(root, text="Select Image for Testing", bg="azure3", fg="#9C661F", font=self.frame_font)
        self.label_image_testing.place(x=10, y=90, width=150, height=100)
        self.label_image_browse = Label(root, text="Selected Image", bg="azure3", fg="#C04000", font=4)
        self.label_image_browse.place(x=15, y=110, width=140, height=20)
        self.txt_image_browse = Entry(root)
        self.txt_image_browse.configure(state="disabled")
        self.txt_image_browse.place(x=20, y=130, width=130, height=25)
        self.btn_image_browse = Button(root, text="Browse", bg="deep sky blue", fg="#fff", width=15, font=self.text_font1, command=self.image_browse)
        self.btn_image_browse.place(x=20, y=160)

        self.label_image_preprocessing = LabelFrame(root, text="Pre_processing", bg="azure3", fg="#9C661F", font=self.frame_font)
        self.label_image_preprocessing.place(x=10, y=200, width=150, height=50)
        self.btn_image_noise_removal = Button(root, text="NR", bg="deep sky blue", fg="#fff", width=7, font=self.text_font1, command=self.image_nramf)
        self.btn_image_noise_removal.place(x=20, y=220)
        self.btn_image_contrast_enhancement = Button(root, text="CE", bg="deep sky blue", fg="#fff", width=7, font=self.text_font1, command=self.image_ceclahe)
        self.btn_image_contrast_enhancement.place(x=90, y=220)

        self.label_image_feature_extraction = LabelFrame(root, text="Feature Processing", bg="azure3", fg="#9C661F", font=self.frame_font)
        self.label_image_feature_extraction.place(x=10, y=260, width=150, height=50)
        self.btn_image_feature_extraction = Button(root, text="Extraction", bg="deep sky blue", fg="#fff", width=7, font=self.text_font1, command=self.image_fe)
        self.btn_image_feature_extraction.place(x=20, y=280)
        self.btn_image_feature_fusion = Button(root, text="Fusion", bg="deep sky blue", fg="#fff", width=7, font=self.text_font1, command=self.image_ff)
        self.btn_image_feature_fusion.place(x=90, y=280)

        self.label_image_cld_classification = LabelFrame(root, text="CLD Classification", bg="azure3", fg="#9C661F", font=self.frame_font)
        self.label_image_cld_classification.place(x=10, y=320, width=150, height=50)
        self.btn_image_cld_classification = Button(root, text="Proceed", bg="deep sky blue", fg="#fff", width=15, font=self.text_font1, command=self.image_cld_classification)
        self.btn_image_cld_classification.place(x=20, y=340)

##################################################################################3
        self.label_output_frame2 = LabelFrame(root, text="Testing Process Window", bg="azure3", fg="#008B8B", font=self.frame_process_res_font)
        self.label_output_frame2.place(x=10, y=380, width=730, height=300)

        self.label_si = Label(root, text="Selected Image", bg="azure3", fg="#9932CC", font=self.frame_process_res_font)
        self.label_si.place(x=20, y=400, width=230, height=30)
        self.image_si = Label()
        self.image_si.place(x=20, y=430, width=230, height=240)

        self.label_nramf = Label(root, text="Noise Removed Image", bg="azure3", fg="#9932CC", font=self.frame_process_res_font)
        self.label_nramf.place(x=260, y=400, width=230, height=30)
        self.image_nramf = Label()
        self.image_nramf.place(x=260, y=430, width=230, height=240)

        self.label_ceclahe = Label(root, text="Contrast Enhanced Image", bg="azure3", fg="#9932CC", font=self.frame_process_res_font)
        self.label_ceclahe.place(x=500, y=400, width=230, height=30)
        self.image_ceclahe = Label()
        self.image_ceclahe.place(x=500, y=430, width=230, height=240)

        self.label_frame_cl = LabelFrame(root, text="Cotton Leaf Disease Classification", bg="azure3", fg="#009ACD", font=self.frame_font)
        self.label_frame_cl.place(x=750, y=600, width=210, height=50)
        self.entry_image_classification = Entry(root, font=self.frame_font)
        self.entry_image_classification.place(x=760, y=620, width=190, height=25)
        self.entry_image_classification.configure(state="disabled")

        self.btn_exit = Button(root, text="Exit", bg="deep sky blue", fg="#fff", width=3, command=self.exit)
        self.btn_exit.place(x=970, y=620)

    def dataset_preprocessing_noise_removal(self):
        self.boolDSNRAMF = True
        self.data_textarea_process.configure(state="normal")
        print("\nPre-processing")
        print("================")
        self.data_textarea_process.insert(INSERT, "\n\nPre-processing")
        self.data_textarea_process.insert(INSERT, "\n================")
        print("\nNoise Removal using Adaptive Median Filter")
        print("--------------------------------------------")
        self.data_textarea_process.insert(INSERT, "\n\nNoise Removal using Adaptive Median Filter")
        self.data_textarea_process.insert(INSERT, "\n--------------------------------------------")

        if not os.path.exists("..\\Output\\Pre_processing\\NRAMF\\"):
            gfsfile_path = getListOfFiles(str("..\\Dataset\\"))

            for x in range(len(gfsfile_path)):
                fl_val = str(gfsfile_path[x]).split("\\")

                if not os.path.exists("..\\Output\\Pre_processing\\NRAMF\\" + str(fl_val[len(fl_val) - 2])):
                    os.makedirs("..\\Output\\Pre_processing\\NRAMF\\" + str(fl_val[len(fl_val) - 2]))

                spath = gfsfile_path[x]
                dpath = "..\\Output\\Pre_processing\\NRAMF\\" + str(fl_val[len(fl_val) - 2]) + "\\" + str(
                    fl_val[len(fl_val) - 1])
                try:
                    AdaptiveMedianFilter.NR_AMF(self, spath, dpath)
                except:
                    pass

        print("\nNoise Removal using Adaptive Median Filter was done Successfully...")
        self.data_textarea_process.insert(INSERT, "\n\nNoise Removal using Adaptive Median Filter was done Successfully...")
        messagebox.showinfo("showinfo", "Noise Removal using Adaptive Median Filter was done Successfully...")

        self.btn_dataset_preprocessing_noise_removal.configure(state="disabled")
        self.data_textarea_process.configure(state="disabled")

    def dataset_preprocessing_contrast_enhancement(self):
        if self.boolDSNRAMF:
            self.boolDSCECLAHE = True
            self.data_textarea_process.configure(state="normal")
            print("\nContrast Enhancement using CLAHE")
            print("----------------------------------")
            self.data_textarea_process.insert(INSERT, "\n\nContrast Enhancement using CLAHE")
            self.data_textarea_process.insert(INSERT, "\n----------------------------------")

            if not os.path.exists("..\\Output\\Pre_processing\\CECL\\"):
                gfsfile_path = getListOfFiles(str("..\\Output\\Pre_processing\\NRAMF\\"))

                for x in range(len(gfsfile_path)):
                    fl_val = str(gfsfile_path[x]).split("\\")

                    if not os.path.exists("..\\Output\\Pre_processing\\CECL\\" + str(fl_val[len(fl_val) - 2])):
                        os.makedirs("..\\Output\\Pre_processing\\CECL\\" + str(fl_val[len(fl_val) - 2]))

                    spath = gfsfile_path[x]
                    dpath = "..\\Output\\Pre_processing\\CECL\\" + str(fl_val[len(fl_val) - 2]) + "\\" + str(
                        fl_val[len(fl_val) - 1])
                    try:
                        CLAHE.CE_CLAHE(self, spath, dpath)
                    except:
                        pass

            print("\nContrast Enhancement using CLAHE was done Successfully...")
            self.data_textarea_process.insert(INSERT,
                                              "\n\nContrast Enhancement using CLAHE was done Successfully...")
            messagebox.showinfo("showinfo", "Contrast Enhancement using CLAHE was done Successfully...")

            self.btn_dataset_preprocessing_contrast_enhancement.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")
        else:
            messagebox.showinfo("showinfo", "Please done the Noise Removal first...")

    def dataset_feature_extraction(self):
        if self.boolDSCECLAHE:
            self.boolDSFE=True
            self.data_textarea_process.configure(state="normal")
            print("\nFeature Extraction using Handcraft, CNN, Inception-V3 and CNNSE_MIV3")
            print("======================================================================")
            self.data_textarea_process.insert(INSERT, "\n\nFeature Extraction using Handcraft, CNN, Inception-V3 and CNNSE_MIV3")
            self.data_textarea_process.insert(INSERT, "\n======================================================================")

            gfsfile_path = getListOfFiles(str("..\\Output\\Pre_processing\\CECL\\"))

            if not os.path.exists("..\\Output\\Features\\"):
                os.makedirs("..\\Output\\Features\\")

                data1 = []
                data2 = []
                data3 = []
                data4 = []

                for y in range(len(gfsfile_path)):
                    try:
                        a = str(gfsfile_path[y]).split("\\")

                        temp = []
                        temp.append(a[len(a)-1])
                        temp.append(a[len(a)-2])
                        temp.append(Handcraft_Features.extract_features(self, gfsfile_path[y]))
                        data1.append(temp)

                        temp = []
                        temp.append(a[len(a)-1])
                        temp.append(a[len(a)-2])
                        temp.append(CNN.extract_features(self, gfsfile_path[y]))
                        data2.append(temp)

                        temp = []
                        temp.append(a[len(a)-1])
                        temp.append(a[len(a)-2])
                        temp.append(Inception_V3.extract_features(self, gfsfile_path[y]))
                        data3.append(temp)

                        temp = []
                        temp.append(a[len(a)-1])
                        temp.append(a[len(a)-2])
                        temp.append(CNNSE_MIV3.extract_features(self, gfsfile_path[y]))
                        data4.append(temp)
                    except:
                        pass

                with open("../Output/Features/Features_Handcraft.csv", 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerows(data1)

                with open("../Output/Features/Features_CNN.csv", 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerows(data2)

                with open("../Output/Features/Features_Inception_V3.csv", 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerows(data3)

                with open("../Output/Features/Features_CNNSE_MIV3.csv", 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerows(data4)

            print("\nFeature Extraction was done Successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nFeature Extraction was done Successfully...")
            messagebox.showinfo("showinfo", "Feature Extraction was done Successfully...")

            self.btn_dataset_feature_extraction.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")
        else:
            messagebox.showinfo("showinfo", "Please done the Contrast Enhancement first")

    def dataset_feature_fusion(self):
        if self.boolDSFE:
            self.boolDSFF = True
            self.data_textarea_process.configure(state="normal")
            print("\nFeature Fusion")
            print("================")
            self.data_textarea_process.insert(INSERT, "\n\nFeature Fusion")
            self.data_textarea_process.insert(INSERT, "\n================")

            # Load the CSV files
            file1 = '..\\Output\\Features_Handcraft.csv'
            file2 = '..\\Output\\Features_CNN.csv'
            file3 = '..\\Output\\Features_Inception_V3.csv'
            file4 = '..\\Output\\Features_CNNSE_MIV3.csv'

            data = []
            temp = []
            with open(file1, mode='r') as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    temp.append(lines[0])

            with open(file2, mode='r') as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    temp.append(lines[0])

            with open(file3, mode='r') as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    temp.append(lines[0])

            with open(file4, mode='r') as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    temp.append(lines[0])

            data.append(temp)
            with open("..\\Output\\Features\\Fused_Features.csv", 'w', newline='') as f:
                w = csv.writer(f)
                w.writerows(data)

            print("\nFeature Fusion was done Successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nFeature Fusion was done Successfully...")
            messagebox.showinfo("showinfo", "Feature Fusion was done Successfully...")

            self.btn_dataset_feature_fusion.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")
        else:
            messagebox.showinfo("showinfo", "Please done the Feature Extraction first...")

    def dataset_splitting(self):
        if os.path.exists("..\\Output\\Features"):
            self.boolDSSplitting = True
            self.data_textarea_process.configure(state="normal")
            print("\nDataset Splitting")
            print("===================")
            self.data_textarea_process.insert(INSERT, "\n\nDataset Splitting")
            self.data_textarea_process.insert(INSERT, "\n===================")

            gfsfile_path = getListOfFiles(str("..\\Output\\Pre_processing\\CECL\\"))

            trsize = int((len(gfsfile_path) * self.training) / 100)
            tssize = int((len(gfsfile_path) * self.testing) / 100)

            for x in range(round(trsize)):
                fl_val = str(gfsfile_path[x]).split("\\")
                self.iptrdata.append(gfsfile_path[x])
                self.iptrcls.append(str(fl_val[len(fl_val)-1]))

            i = trsize

            while i < len(gfsfile_path):
                fl_val = str(gfsfile_path[x]).split("\\")
                self.iptsdata.append(gfsfile_path[i])
                self.iptscls.append(str(fl_val[len(fl_val)-1]))
                if i == len(gfsfile_path):
                    break
                i = i + 1

            print("Total no. of Data : " + str(len(gfsfile_path)))
            print("Total no. of Training Data (80%) : " + str(len(self.iptrdata)))
            print("Total no. of Testing Data (20%) : " + str(len(self.iptscls)))

            self.data_textarea_process.insert(INSERT, "\nTotal no. of Data : " + str(len(gfsfile_path)))
            self.data_textarea_process.insert(INSERT, "\nTotal no. of Training Data (80%) : " + str(len(self.iptrdata)))
            self.data_textarea_process.insert(INSERT, "\nTotal no. of Testing Data (20%) : " + str(len(self.iptscls)))

            print("\nDataset Splitting was done Successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nDataset Splitting was done Successfully...")
            messagebox.showinfo("showinfo", "Dataset Splitting was done Successfully...")

            self.btn_dataset_splitting.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")
        else:
            messagebox.showinfo("showinfo", "Please done the feature fusion first...")

    def cld_training(self):
        if self.boolDSSplitting:
            self.boolTraining = True
            self.data_textarea_process.configure(state="normal")
            self.data_textarea_result.configure(state="normal")
            self.data_textarea_process.insert(INSERT, "\n\nCotton Leaf Disease Classification Training")
            self.data_textarea_process.insert(INSERT, "\n===============================================")
            self.data_textarea_result.insert(INSERT, "\n\nCLD Classification Training")
            self.data_textarea_result.insert(INSERT, "\n============================")
            print("\nCotton Leaf Disease Classification Training")
            print("================================================")

            print("\nTotal no. of Training Data : " + str(len(self.iptrdata)))
            self.data_textarea_process.insert(INSERT, "\nTotal no. of Training Data : " + str(len(self.iptrdata)))
            self.data_textarea_process.insert(INSERT, "\nTotal no. of Training Data : " + str(len(self.iptrdata)))

            print("\nExisting Artificial Neural Network (ANN)")
            print("-----------------------------------------")
            self.data_textarea_process.insert(INSERT, "\n\nExisting Artificial Neural Network (ANN)")
            self.data_textarea_process.insert(INSERT, "\n-----------------------------------------")
            self.data_textarea_result.insert(INSERT, "\n\nExisting ANN")
            self.data_textarea_result.insert(INSERT, "\n---------------")
            stime = int(time.time() * 1000)
            Existing_ANN.training(self, self.iptrdata, self.iptrcls)
            etime = int(time.time() * 1000)
            cfg.cnntrtime = etime - stime
            print("Training Time : " + str(etime - stime) + " in ms")
            self.data_textarea_result.insert(INSERT, "\nTraining Time : " + str(etime - stime) + " in ms")

            print("\nExisting Long Short-Term Memory (LSTM)")
            print("-------------------------------------")
            self.data_textarea_process.insert(INSERT, "\n\nExisting Long Short-Term Memory (LSTM)")
            self.data_textarea_process.insert(INSERT, "\n------------------------------------")
            self.data_textarea_result.insert(INSERT, "\n\nExisting LSTM")
            self.data_textarea_result.insert(INSERT, "\n--------------")
            stime = int(time.time() * 1000)
            Existing_LSTM.training(self, self.iptrdata, self.iptrcls)
            etime = int(time.time() * 1000)
            cfg.dbntrtime = etime - stime
            print("Training Time : " + str(etime - stime) + " in ms")
            self.data_textarea_result.insert(INSERT, "\nTraining Time : " + str(etime - stime) + " in ms")

            print("\nProposed Multi Loss based Multi Layer Perceptron (MLMLP)")
            print("----------------------------------------------------------")
            self.data_textarea_process.insert(INSERT, "\n\nProposed Multi Loss based Multi Layer Perceptron (MLMLP)")
            self.data_textarea_process.insert(INSERT, "\n----------------------------------------------------------")
            self.data_textarea_result.insert(INSERT, "\n\nProposed MLMLP")
            self.data_textarea_result.insert(INSERT, "\n------------------")
            stime = int(time.time() * 1000)
            Proposed_MLMLP.training(self, self.iptrdata, self.iptrcls)
            etime = int(time.time() * 1000)
            cfg.odbntrtime = etime - stime
            print("Training Time : " + str(etime - stime) + " in ms")
            self.data_textarea_result.insert(INSERT, "\nTraining Time : " + str(etime - stime) + " in ms")

            print("\nCotton Leaf Disease Classification Training was done successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nCotton Leaf Disease Classification Training was done successfully...")
            messagebox.showinfo("Info Message", "Cotton Leaf Disease Classification Training was done successfully...")

            self.btn_cld_training.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")
        else:
            messagebox.showinfo("showinfo", "Please done dataset splitting first...")

    def cld_testing(self):
        if self.boolDSSplitting:
            if os.path.exists("..\\Models\\"):
                self.boolTesting = True
                self.boolTesting = True
                if not os.path.exists("..\\CM\\"):
                    os.mkdir("..\\CM\\")
                self.boolTesting = True
                self.data_textarea_process.configure(state="normal")
                self.data_textarea_result.configure(state="normal")
                cm = []
                temp = []
                temp.append("TP")
                temp.append("FN")
                cm.append(temp)

                temp = []
                temp.append("FP")
                temp.append("TN")
                cm.append(temp)

                self.data_textarea_process.insert(INSERT, "\n\nCotton Leaf Disease Classification Testing")
                self.data_textarea_process.insert(INSERT, "\n===============================================")
                self.data_textarea_result.insert(INSERT, "\n\nCLD Classification Testing")
                self.data_textarea_result.insert(INSERT, "\n============================")
                print("\nCotton Leaf Disease Classification Testing")
                print("================================================")

                print("\nExisting Artificial Neural Network (ANN)")
                print("-----------------------------------------")
                self.data_textarea_process.insert(INSERT, "\n\nExisting Artificial Neural Network (ANN)")
                self.data_textarea_process.insert(INSERT, "\n-----------------------------------------")
                self.data_textarea_result.insert(INSERT, "\n\nExisting ANN")
                self.data_textarea_result.insert(INSERT, "\n---------------")
                Existing_ANN.testing(self, self.iptsdata, self.iptscls)
                print("Confusion Matrix : ")
                for x in range(len(cm)):
                    print(cm[x])
                for x in range(len(cm)):
                    print(str(cfg.ednncm[x]))

                print("\nPrecision : " + str(cfg.ednnpre))
                print("Recall : " + str(cfg.ednnrec))
                print("FMeasure : " + str(cfg.ednnfsc))
                print("Accuracy : " + str(cfg.ednnacc))
                print("Sensitivity : " + str(cfg.ednnsens))
                print("Specificity : " + str(cfg.ednnspec))
                print("TPR : " + str(cfg.ednntpr))
                print("TNR : " + str(cfg.ednntnr))
                print("PPV : " + str(cfg.ednnppv))
                print("NPV : " + str(cfg.ednnnpv))
                print("FNR : " + str(cfg.ednnfnr))
                print("FPR : " + str(cfg.ednnfpr))

                self.data_textarea_result.insert(INSERT, "\nConfusion Matrix : ")
                for x in range(len(cm)):
                    self.data_textarea_result.insert(INSERT, "\n" + str(cm[x]))
                for x in range(len(cm)):
                    self.data_textarea_result.insert(INSERT, "\n" + str(cfg.ednncm[x]))

                self.data_textarea_result.insert(INSERT, "\n\nPrecision : " + str(cfg.ednnpre))
                self.data_textarea_result.insert(INSERT, "\nRecall : " + str(cfg.ednnrec))
                self.data_textarea_result.insert(INSERT, "\nFMeasure : " + str(cfg.ednnfsc))
                self.data_textarea_result.insert(INSERT, "\nAccuracy : " + str(cfg.ednnacc))
                self.data_textarea_result.insert(INSERT, "\nSensitivity : " + str(cfg.ednnsens))
                self.data_textarea_result.insert(INSERT, "\nSpecificity : " + str(cfg.ednnspec))
                self.data_textarea_result.insert(INSERT, "\nTPR : " + str(cfg.ednntpr))
                self.data_textarea_result.insert(INSERT, "\nTNR : " + str(cfg.ednntnr))
                self.data_textarea_result.insert(INSERT, "\nPPV : " + str(cfg.ednnppv))
                self.data_textarea_result.insert(INSERT, "\nNPV : " + str(cfg.ednnnpv))
                self.data_textarea_result.insert(INSERT, "\nFNR : " + str(cfg.ednnfnr))
                self.data_textarea_result.insert(INSERT, "\nFPR : " + str(cfg.ednnfpr))

                print("\nExisting Long Short-Term Memory (LSTM)")
                print("-------------------------------------")
                self.data_textarea_process.insert(INSERT, "\n\nExisting Long Short-Term Memory (LSTM)")
                self.data_textarea_process.insert(INSERT, "\n------------------------------------")
                self.data_textarea_result.insert(INSERT, "\n\nExisting LSTM")
                self.data_textarea_result.insert(INSERT, "\n--------------")
                Existing_LSTM.testing(self, self.iptsdata, self.iptscls)
                print("Confusion Matrix : ")
                for x in range(len(cm)):
                    print(cm[x])
                for x in range(len(cm)):
                    print(str(cfg.edbncm[x]))

                print("\nPrecision : " + str(cfg.edbnpre))
                print("Recall : " + str(cfg.edbnrec))
                print("FMeasure : " + str(cfg.edbnfsc))
                print("Accuracy : " + str(cfg.edbnacc))
                print("Sensitivity : " + str(cfg.edbnsens))
                print("Specificity : " + str(cfg.edbnspec))
                print("TPR : " + str(cfg.edbntpr))
                print("TNR : " + str(cfg.edbntnr))
                print("PPV : " + str(cfg.edbnppv))
                print("NPV : " + str(cfg.edbnnpv))
                print("FNR : " + str(cfg.edbnfnr))
                print("FPR : " + str(cfg.edbnfpr))

                self.data_textarea_result.insert(INSERT, "\nConfusion Matrix : ")
                for x in range(len(cm)):
                    self.data_textarea_result.insert(INSERT, "\n" + str(cm[x]))
                for x in range(len(cm)):
                    self.data_textarea_result.insert(INSERT, "\n" + str(cfg.edbncm[x]))

                self.data_textarea_result.insert(INSERT, "\n\nPrecision : " + str(cfg.edbnpre))
                self.data_textarea_result.insert(INSERT, "\nRecall : " + str(cfg.edbnrec))
                self.data_textarea_result.insert(INSERT, "\nFMeasure : " + str(cfg.edbnfsc))
                self.data_textarea_result.insert(INSERT, "\nAccuracy : " + str(cfg.edbnacc))
                self.data_textarea_result.insert(INSERT, "\nSensitivity : " + str(cfg.edbnsens))
                self.data_textarea_result.insert(INSERT, "\nSpecificity : " + str(cfg.edbnspec))
                self.data_textarea_result.insert(INSERT, "\nTPR : " + str(cfg.edbntpr))
                self.data_textarea_result.insert(INSERT, "\nTNR : " + str(cfg.edbntnr))
                self.data_textarea_result.insert(INSERT, "\nPPV : " + str(cfg.edbnppv))
                self.data_textarea_result.insert(INSERT, "\nNPV : " + str(cfg.edbnnpv))
                self.data_textarea_result.insert(INSERT, "\nFNR : " + str(cfg.edbnfnr))
                self.data_textarea_result.insert(INSERT, "\nFPR : " + str(cfg.edbnfpr))

                print("\nProposed Multi Loss based Multi Layer Perceptron (MLMLP)")
                print("----------------------------------------------------------")
                self.data_textarea_process.insert(INSERT, "\n\nProposed Multi Loss based Multi Layer Perceptron (MLMLP)")
                self.data_textarea_process.insert(INSERT, "\n----------------------------------------------------------")
                self.data_textarea_result.insert(INSERT, "\n\nProposed MLMLP")
                self.data_textarea_result.insert(INSERT, "\n------------------")
                Proposed_MLMLP.testing(self, self.iptsdata, self.iptscls)
                print("Confusion Matrix : ")
                for x in range(len(cm)):
                    print(cm[x])
                for x in range(len(cm)):
                    print(str(cfg.pmlmlpcm[x]))

                print("\nPrecision : " + str(cfg.pmlmlppre))
                print("Recall : " + str(cfg.pmlmlprec))
                print("FMeasure : " + str(cfg.pmlmlpfsc))
                print("Accuracy : " + str(cfg.pmlmlpacc))
                print("Sensitivity : " + str(cfg.pmlmlpsens))
                print("Specificity : " + str(cfg.pmlmlpspec))
                print("TPR : " + str(cfg.pmlmlptpr))
                print("TNR : " + str(cfg.pmlmlptnr))
                print("PPV : " + str(cfg.pmlmlpppv))
                print("NPV : " + str(cfg.pmlmlpnpv))
                print("FNR : " + str(cfg.pmlmlpfnr))
                print("FPR : " + str(cfg.pmlmlpfpr))

                self.data_textarea_result.insert(INSERT, "\nConfusion Matrix : ")
                for x in range(len(cm)):
                    self.data_textarea_result.insert(INSERT, "\n" + str(cm[x]))
                for x in range(len(cm)):
                    self.data_textarea_result.insert(INSERT, "\n" + str(cfg.pmlmlpcm[x]))

                self.data_textarea_result.insert(INSERT, "\n\nPrecision : " + str(cfg.pmlmlppre))
                self.data_textarea_result.insert(INSERT, "\nRecall : " + str(cfg.pmlmlprec))
                self.data_textarea_result.insert(INSERT, "\nFMeasure : " + str(cfg.pmlmlpfsc))
                self.data_textarea_result.insert(INSERT, "\nAccuracy : " + str(cfg.pmlmlpacc))
                self.data_textarea_result.insert(INSERT, "\nSensitivity : " + str(cfg.pmlmlpsens))
                self.data_textarea_result.insert(INSERT, "\nSpecificity : " + str(cfg.pmlmlpspec))
                self.data_textarea_result.insert(INSERT, "\nTPR : " + str(cfg.pmlmlptpr))
                self.data_textarea_result.insert(INSERT, "\nTNR : " + str(cfg.pmlmlptnr))
                self.data_textarea_result.insert(INSERT, "\nPPV : " + str(cfg.pmlmlpppv))
                self.data_textarea_result.insert(INSERT, "\nNPV : " + str(cfg.pmlmlpnpv))
                self.data_textarea_result.insert(INSERT, "\nFNR : " + str(cfg.pmlmlpfnr))
                self.data_textarea_result.insert(INSERT, "\nFPR : " + str(cfg.pmlmlpfpr))

                print("\nCotton Leaf Disease Classification Testing was done successfully...")
                self.data_textarea_process.insert(INSERT, "\n\nCotton Leaf Disease Classification Testing was done successfully...")
                messagebox.showinfo("Info Message", "Cotton Leaf Disease Classification Testing was done successfully...")

                self.data_textarea_process.configure(state="disabled")
                self.data_textarea_result.configure(state="disabled")
                self.btn_cld_testing.configure(state="disabled")
            else:
                messagebox.showinfo("showinfo", "Please done Cotton Leaf Disease Classification Training first...")
        else:
            messagebox.showinfo("showinfo", "Please done dataset splitting first...")

    def image_browse(self):
        self.boolTraining = True
        if self.boolTraining:
            self.boolImageRead = True
            self.txt_image_browse.configure(state="normal")
            self.data_textarea_process.configure(state="normal")
            self.test_image = askopenfile(mode='r', filetypes=[('All Files', '*')])
            self.txt_image_browse.insert(INSERT, "" + str(self.test_image.name))

            print("\nSelected Image Name : " + str(self.test_image.name))
            self.data_textarea_process.insert(INSERT, "\nSelected Image Name : " + str(self.test_image.name))
            self.bool_image_file = True

            img = cv2.imread(self.test_image.name)
            dim = (230, 240)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            in_image = ImageTk.PhotoImage(image=Image.fromarray(resized))
            self.lbl_si = Label(root, image=in_image)
            self.lbl_si.image = in_image
            self.lbl_si.place(x=20, y=430, width=230, height=240)

            print("\nTesting image was selected successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nTesting image was selected successfully...")
            messagebox.showinfo("Info Message", "Testing image was selected successfully...")

            self.txt_image_browse.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")

            self.btn_image_noise_removal.configure(state="normal")
            self.btn_image_contrast_enhancement.configure(state="normal")
            self.btn_image_feature_extraction.configure(state="normal")
            self.btn_image_feature_fusion.configure(state="normal")
            self.btn_image_cld_classification.configure(state="normal")
            self.btn_image_browse.configure(state="normal")
        else:
            messagebox.showinfo("showinfo", "Please done CKD training first...")

    def image_nramf(self):
        if self.boolImageRead:
            self.boolImageNoiseRemoval = True
            self.data_textarea_process.configure(state="normal")
            print("\nPre_processing")
            print("================")
            self.data_textarea_process.insert(INSERT, "\n\nPre_processing")
            self.data_textarea_process.insert(INSERT, "\n================")

            print("Noise Removal using Adaptive Median Filter")
            print("------------------------------------------")
            self.data_textarea_process.insert(INSERT, "\nNoise Removal using Adaptive Median Filter")
            self.data_textarea_process.insert(INSERT, "\n------------------------------------------")

            # Load the image
            image = cv2.imread(self.test_image.name)
            # Convert the image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
            plt.imsave("..\\Output\\NRAMF.jpg", denoised_image)
            dim = (230, 240)
            resized = cv2.resize(denoised_image, dim, interpolation=cv2.INTER_AREA)
            in_image = ImageTk.PhotoImage(image=Image.fromarray(resized))
            img = cv2.imread("..\\Output\\NRAMF.jpg")
            dim = (230, 240)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            in_image = ImageTk.PhotoImage(image=Image.fromarray(resized))
            self.image_nramf = Label(root, image=in_image)
            self.image_nramf.image = in_image
            self.image_nramf.place(x=260, y=430, width=230, height=240)

            print("\nNoise Removal using Adaptive Median Filter was done Successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nNoise Removal using Adaptive Median Filter was done Successfully...")
            messagebox.showinfo("showinfo", "Noise Removal using Adaptive Median Filter was done Successfully...")

            self.btn_image_noise_removal.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")
        else:
            messagebox.showinfo("showinfo", "Please select the image first...")

    def image_ceclahe(self):
        if self.boolImageRead:
            self.boolImageContrastEnhancement = True
            self.data_textarea_process.configure(state="normal")
            print("\nContrast Enhancement using CLAHE")
            print("--------------------------------")
            self.data_textarea_process.insert(INSERT, "\n\nContrast Enhancement using CLAHE")
            self.data_textarea_process.insert(INSERT, "\n--------------------------------")

            CLAHE.CE_CLAHE(self, "..\\Output\\NRAMF.jpg", "..\\Output\\CECLAHE.jpg")
            img = cv2.imread("..\\Output\\CECLAHE.jpg")
            dim = (230, 240)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            in_image = ImageTk.PhotoImage(image=Image.fromarray(resized))
            self.image_ceclahe = Label(root, image=in_image)
            self.image_ceclahe.image = in_image
            self.image_ceclahe.place(x=500, y=430, width=230, height=240)

            print("\nContrast Enhancement using CLAHE was done Successfully...")
            self.data_textarea_process.insert(INSERT,
                                              "\n\nContrast Enhancement using CLAHE was done Successfully...")
            messagebox.showinfo("showinfo", "Contrast Enhancement using CLAHE was done Successfully...")

            self.btn_image_contrast_enhancement.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")
        else:
            messagebox.showinfo("showinfo", "Please select the image first...")

    def image_fe(self):
        if self.boolImageContrastEnhancement:
            self.boolImageFeatureExtraction = True
            self.data_textarea_process.configure(state="normal")
            print("\nFeature Extraction using Handcraft, CNN, Inception-V3 and CNNSE_MIV3")
            print("======================================================================")
            self.data_textarea_process.insert(INSERT, "\n\nFeature Extraction using Handcraft, CNN, Inception-V3 and CNNSE_MIV3")
            self.data_textarea_process.insert(INSERT, "\n======================================================================")


            print("\nFeature Extraction was done Successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nFeature Extraction was done Successfully...")
            messagebox.showinfo("showinfo", "Feature Extraction was done Successfully...")

            self.btn_image_feature_extraction.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")
        else:
            messagebox.showinfo("showinfo", "Please done the Contrast Enhancement first...")

    def image_ff(self):
        if self.boolImageContrastEnhancement:
            self.boolImageFeatureExtraction = True
            self.data_textarea_process.configure(state="normal")
            print("\nFeature Fusion")
            print("================")
            self.data_textarea_process.insert(INSERT, "\n\nFeature Fusion")
            self.data_textarea_process.insert(INSERT, "\n================")

            # Load the CSV files
            file1 = '..\\Output\\Features_Handcraft.csv'
            file2 = '..\\Output\\Features_CNN.csv'
            file3 = '..\\Output\\Features_Inception_V3.csv'
            file4 = '..\\Output\\Features_CNNSE_MIV3.csv'

            data = []
            temp = []
            with open(file1, mode='r') as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    temp.append(lines[0])

            with open(file2, mode='r') as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    temp.append(lines[0])

            with open(file3, mode='r') as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    temp.append(lines[0])

            with open(file4, mode='r') as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    temp.append(lines[0])

            data.append(temp)
            with open("../Output/Fused_Features.csv", 'w', newline='') as f:
                w = csv.writer(f)
                w.writerows(data)

            print("\nFeature Fusion was done Successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nFeature Fusion was done Successfully...")
            messagebox.showinfo("showinfo", "Feature Fusion was done Successfully...")

            self.btn_image_feature_fusion.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")
        else:
            messagebox.showinfo("showinfo", "Please done the Feature Extraction first...")

    def image_cld_classification(self):
        if self.boolImageFeatureExtraction:
            self.data_textarea_process.configure(state="normal")
            self.entry_image_classification.configure(state="normal")

            self.entry_image_classification.delete(0, len(self.entry_image_classification.get()))

            self.data_textarea_process.insert(INSERT, "\n\nCotton Leaf Disease Classification")
            self.data_textarea_process.insert(INSERT, "\n====================================")
            print("\nCotton Leaf Disease Classification")
            print("====================================")
            result = Proposed_MLMLP.test_image(self, self.test_image.name)
            self.data_textarea_process.insert(INSERT, "\nSelected Image Class is : "+str(result))
            print("\nSelected Image Class is : "+str(result))
            self.entry_image_classification.insert(INSERT, str(result))
            messagebox.showinfo("showinfo", "Selected Image Class is : "+str(result))

            self.btn_image_cld_classification.configure(state="disabled")
            self.entry_image_classification.configure(state="disabled")
            self.data_textarea_process.configure(state="disabled")
        else:
            messagebox.showinfo("showinfo", "Please done the Feature Fusion first...")

    def tables_graphs(self):
        if self.boolTesting:

            if not os.path.exists("..\\Result\\"):
                os.mkdir("..\\Result\\")

            from CLDC.Code import Graph


            messagebox.showinfo("Info Message","Generate Tables and Graphs was done successfully...")
        else:
            messagebox.showinfo("Info Message","Please done the Sarcasm Detection Testing first ...")

    def exit(self):
        self.root.destroy()

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if not fullPath.endswith(".csv"):
                allFiles.append(fullPath)

    return allFiles

root = Tk()
root.title("COTTON LEAF DISEASE CLASSIFICATION")
root.geometry("1050x700")
root.resizable(0, 0)
root.configure(bg="azure3")
od = Main_GUI(root)
root.mainloop()
