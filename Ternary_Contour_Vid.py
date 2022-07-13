# This function was originally written by Dan Hallatt, in-part using the ternary plotter made by Corentin Le Guillou. Both at the Université de Lille.

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.tri as tri
import scipy.stats as st
from scipy.interpolate import Rbf
from scipy import interpolate, optimize
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import math
from sigfig import round
from cycler import cycler
from statistics import mean


# A) "Ternary_ContourVid' : plots .gif video of stitched series of ternary contour plots, using a data in the form [[A1, B1, C1], [A2, B2, C2], ...]. Also plots a ternary plot of all of the raw data, for comparison/check of the contours.

#   A.1) "Tern_Base" : defines the base structure of the ternary plot (axis names, stoichiometric reference lines, ect..).

#   A.2) "init_TernVid" : makes a blank contour plot prior to looping through each 'window' of data. Required for video (.gif) making.

#   A.3) "animate_TernVid" : steps through each window of data plotting a ternary contour plot for each.

#   A.4) "triplot" : plots static ternary plot of raw data.


def Ternary_ContourVid(dataset, WindDispType, WindDispPositionData, type, WindowWidth, NumberOfSteps, fps, NumLevels, StartingIndex_FirstWindow, FigureSavePath, FileName, Colour):
    """ Uses matplotlib to plot a ternay diagram who's data are plotted as a contour plot of the density (in A,B,C-space) of datapoints. The Ternary diagram scans through the data according to a user-defined number of datapoints and plots consecutive contour plots of that data, wrapped into a .gif video file.
    
    ** dataset : should be an array containing the values to plot, in the form [Ai, Bi, Ci] where A is the top corner, B is the left corner and C the right corner of the triangle (A = Si + Al, B = Fe, C = Mg for example)
        - Several datapoints should be within this array, where the entire dataset is in the form [[A1, B1, C1], [A2, B2, C2], ...[An, Bn, Cn]] for n datapoints. This form is fundamentally required for contours (populations of datapoints, not single datapoints).
        
        !!! NOTE !!! : Data cannot contain 'NaN' values. Must be cleaned prior to input in this function (or this function modified to clean data by removing them).
        
    ** WindDispType : either 'index', 'distance' or 'none' to display the  position of the window for each frame of the video (continously plots the first and last index of the datapoints considered for videoframe's contour). User specifies the type of identifier of the position of the window (data index, or physical position, such as distance in 'nm' from a user-defined datum like the end of an EDS line-scan).
    ** WindDispPositionData : either a simple 1D array of position data [X1, X2, X3, ... Xn] where n = number of datapoints in 'dataset', or 'none' if WindDispType is set to either 'index' or 'none.
    ** type : Either 'silicate' [Si+Al, Mg, Fe], 'silicateHydration' [Si+Al, Mg, O], or 'sulfide' [S, Ni, Fe].
    ** WindowWidth : width (number of data points) to consider for each contour plot (each videoframe). Higher number means smoother contours but lower 'resolution' in window position (wider range of data considered to make each contour).
    ** NumberOfSteps : the number of data points ([Si+Al, Fe, Mg] triplicates) which to scan through. Can be the algebraic solution to the entire dataset (considering the user-defined value of the WindowWidth), or can be less if a subset of the data is only wanted to be considered.
    ** fps : frames per second of the video. Increase if it progresses too slow, decrease if it plays too fast.
    ** NumLevels: number of contour levels.
    ** StartingIndex_FirstWindow : index of the first data point. Do you want to start at the first data point (=0), or somewhere offset from it? (=100 for example). Thus: StartingIndex_FirstWindow + WindowWidth * NumberOfSteps = Very last index of data considered (in final video frame).
    ** FigureSavePath : Path to folder location where figures should be saved. Must be in single quotations, example : '/Volumes/Samsung_T5/Experiment categories/Laser/Figures/'
    ** FileName : General name of files to be saved. Must be in single quotations, example : 'TEST_DataSet01'
    ** Colour : In quotations (such as 'Blues') the colour of the contour plot. Options available according to 'cmap' of matplotlib (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
    """
    
    FinalIndex_LastWindow = StartingIndex_FirstWindow +  WindowWidth + NumberOfSteps
    
    # ------------------------

    def Tern_Base():
        """
        This is specifically the base style of the ternary plot.
        - defines axis positions, stoichiometric reference lines/locations.
        """
        ax.plot([-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0],  [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], color = 'black', marker="_")
        ax.plot([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],  [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0], color='black', marker="_")
        ax.plot([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5], [0,0,0,0,0,0,0,0,0,0,0], color='black', marker="|")

        if type == 'silicate':
            ax.plot([0.29, -0.29], [0.4, 0.4], linestyle=':', color='dimgrey')
            ax.annotate('Serp', xy=(0.3, 0.4), xytext = (0.32, 0.39), size = 10)
            
            ax.plot([0.21, -0.21], [0.57, 0.57], linestyle=':', color='dimgrey')
            ax.annotate('Sap', xy=(0.21, 0.57), xytext = (0.23, 0.56), size = 10)
            
            ax.plot([0.329, -0.329], [0.333, 0.333], linestyle='--', color='dimgrey')
            ax.annotate('Ol', xy=(0.333, 0.333), xytext = (0.35, 0.32), size = 10)
            
            ax.plot([0.24, -0.24], [0.5, 0.5], linestyle='--', color='dimgrey')
            ax.annotate('Py', xy=(0.25, 0.50), xytext = (0.27, 0.49), size = 10)
            
            ax.annotate('Si + Al', xy=(0.2, 1.), xytext = (-0.06, 1.03), size = 13)
            ax.annotate('Fe', xy=(-0.55,0.), xytext = (-0.57,-0.03), size = 13)
            ax.annotate('Mg', xy=(0.55,0.), xytext = (0.53, -0.03), size = 13)

        if type == 'sulfide':
            ax.plot(-0.25, 0.5, 'ro', markersize=5)
            ax.plot(0.0, 0.47, 'bo', markersize = 5)
            ax.plot([-0.25, -0.22], [0.5, 0.56], color='purple', linewidth = 7)
            ax.plot(-0.17,0.67, 'go',  markersize = 5)
            ax.annotate('Troilite', xy=(-0.25, 0.5), xytext = (-0.4, 0.5), size=10)
            ax.annotate('Pentlandite', xy=(0.0, 0.47), xytext = (0.03,0.47), size=10)
            ax.annotate('Pyrrhotite', xy=(-0.235, 0.55), xytext = (-0.42,0.55), size=10)
            ax.annotate('Pyrite', xy=(-0.17,0.67), xytext = (-0.32,0.67), size=10)
            ax.annotate('S', xy=(0., 1.), xytext = (-0.05, 1.03), size=13)
            ax.annotate('Fe', xy=(-0.55,0.), xytext = (-0.56,-0.03), size=13)
            ax.annotate('Ni', xy=(0.55,0.), xytext = (0.52, -0.03), size=13)
            
        if type == 'silicateHydration':
            ax.plot(-0.21, 0.142857, marker="3", color= 'dimgrey', markersize=8, alpha=0.5)
            ax.annotate('Serp', xy=(-0.2, 0.14), xytext = (-0.235,0.05), size=10)
            ax.annotate('Si + Al', xy=(0.01, 1.0), xytext = (-0.06, 1.05), size = 13)
            ax.annotate('O', xy=(-0.55,0.0), xytext = (-0.57,-0.03), size = 13)
            ax.annotate('Mg + Fe', xy=(0.55,0.0), xytext = (0.53, -0.03), size = 13)
            
        
    # ------------------------

    def init_TernVid():
        """ This is an initial ternary contour plot, using blank data to initialize the ternary diagram. Required to be run before real data is populated on the ternary diagram.
        """
        
        Tern_Base()
        
        xmin_TernVid_temp, xmax_TernVid_temp = -0.6, 0.6
        ymin_TernVid_temp, ymax_TernVid_temp = -0.1, 1.1

        # Peform the kernel density estimate
        xx_TernVid_temp, yy_TernVid_temp = np.mgrid[xmin_TernVid_temp:xmax_TernVid_temp:300j, ymin_TernVid_temp:ymax_TernVid_temp:300j]
        ax.contourf(xx_TernVid_temp, yy_TernVid_temp, np.zeros((300, 300)), NumLevels, cmap=Colour)
        ax.axis('off')
        
    # ------------------------

    def animate_TernVid(i): # i is the index of each video frame, which has a certain number of datapoints defined by 'WindowWidth'.
        """ Stitches together all data-frame's contour plots into a gif.
        """
        
        ax.clear() # Clearing axis with every new frame. Will allow re-writing of window position, ect..

        Tern_Base() # defines basic ternary diagram.
        
        StartingIndex = StartingIndex_FirstWindow + i # index of one end of the data window.
        EndingIndex = StartingIndex_FirstWindow +  WindowWidth + i # index of the second end of the data window.
        
        # Defining temporary lists of data and the contour's gris for the particular video frame (i).
        data_TernVid_temp = data_TernVid_list[i]
        xx_TernVid_temp = xx_TernVid_list[i]
        yy_TernVid_temp = yy_TernVid_list[i]
            
        # plotting contour for video frame i.
        ax.contourf(xx_TernVid_temp, yy_TernVid_temp, data_TernVid_temp, NumLevels, cmap=Colour)
        
        # displaying the position of the window in the units of data indexes.
        if WindDispType == 'index':
            ax.text(0.38, 0.9, '[' + str(StartingIndex) + ' - ' + str(EndingIndex) + ']')
        elif WindDispType == 'distance':
            ax.text(0.32, 0.9, '[' + str(int(WindDispPositionData[StartingIndex])) + ' - ' + str(int(WindDispPositionData[EndingIndex])) + 'nm]')
        ax.axis('off')
    
    def triplot(dataset, mask, legend, type, dataform):
        """ A general plot that uses matplotlib to plot a ternary diagram
        ** dataset should be an array containing the values to plot, in the form [A, B, C] where A is the top corner, B is the left corner and C the right corner of the triangle (A = Si + Al, B = Fe, C = Mg for example).
        ** dataform : 'single' or 'multiple': Several datasets can be plotted, in that case the dataset is in the form [[A1, B1, C1], [A2, B2, C2], ...]. 'multiple' is required for the contour plot use of this function, as is used here.
        ** mask : can be applied to the dataset
        ** legend : A list for legending different datasets
        ** type : Either 'silicate' [Si+Al, Mg, Fe],  'silicateHydration' [Si+Al, Mg, O], or 'sulfide' [S, Ni, Fe].
        !!! This function is used here in the contour video function to plot a static ternary diagram with all of the data as a reference to the contour plot for the user. !!!
        """
        
        # Converting the 3D ternary data of [Si+Al, Fe, Mg] into 2D coordinates of [x,y] which correspond to the area within the ternary bounds. New data is thus tri_y and tri_x.
        tri_y=[]
        tri_x=[]
        if dataform == 'multiple':
            for i in range (len(dataset)): # i is number of datapoints (not number of data total (groups of 3, for each axis)).
                tri_y.append(dataset[i][0]/(dataset[i][0]+dataset[i][1]+dataset[i][2]))
                tri_x.append(np.array(-0.5)*(np.array(1.)-tri_y[i])+(dataset[i][2]/(dataset[i][0]+dataset[i][1]+dataset[i][2])))
        elif dataform == 'single':
            tri_y.append(dataset[0]/(dataset[0]+dataset[1]+dataset[2]))
            tri_x.append(np.array(-0.5)*(np.array(1.)-tri_y)+(dataset[2]/(dataset[0]+dataset[1]+dataset[2])))
        else:
            print('Problem in user-definition of "dataform" variable within "Ternary_ContourVid" function.')
        
        Tern_Base()
        
        for i in range(StartingIndex_FirstWindow, FinalIndex_LastWindow,1):
            ax.plot(tri_x[i], tri_y[i], 'o', markersize = 0.6)

        ax.axis('off')
        fig.savefig(FigureSavePath + 'Ternary_AllDataPoints.pdf')
     
    fig, ax = plt.subplots()
    # Step 1: Making ternary (non-contour) of all datapoints, for user's reference to contour plot.
    triplot(dataset=dataset, mask=None, legend=None, type=type, dataform='multiple')
        
    # Making ternary diagram contour video.
    data_TernVid_list = []
    xx_TernVid_list = []
    yy_TernVid_list = []
    for j in range(NumberOfSteps):
        StartingIndex = StartingIndex_FirstWindow + j # index of one end of the current data window.
        EndingIndex = StartingIndex_FirstWindow +  WindowWidth + j # index of the second end of the current data window.

        #Bounds of 2D grid-system for plotting contours, will set how close the edge of the figure is to the edge of the ternary axis.
        xmin_TernVid, xmax_TernVid = -0.6, 0.6
        ymin_TernVid, ymax_TernVid = -0.1, 1.1

        # Making grid to access contours on, defined on bounds defined above
        xx_TernVid, yy_TernVid = np.mgrid[xmin_TernVid:xmax_TernVid:300j, ymin_TernVid:ymax_TernVid:300j]
        positions_TernVid = np.vstack([xx_TernVid.ravel(), yy_TernVid.ravel()])
        
        # Data to be contoured, translated into [x,y] format from [[Si+Al, Fe, Mg], [Si+Al,Fe,Mg], ..] format. This is a temporary variable, re-written for each new video frame (each new sliding window across the line-scan.)
        dataset_TernVid_temp = dataset[StartingIndex: EndingIndex]
        tri_y_TernVid=[]
        tri_x_TernVid=[]
        for k in range (len(dataset_TernVid_temp)):
            tri_y_TernVid.append(dataset_TernVid_temp[k][0]/(dataset_TernVid_temp[k][0]+dataset_TernVid_temp[k][1]+dataset_TernVid_temp[k][2]))
            tri_x_TernVid.append(np.array(-0.5)*(np.array(1.)-tri_y_TernVid[k])+(dataset_TernVid_temp[k][2]/(dataset_TernVid_temp[k][0]+dataset_TernVid_temp[k][1]+dataset_TernVid_temp[k][2])))
            
        y_TernVid = tri_y_TernVid
        x_TernVid = tri_x_TernVid
        
        # Formatting input data to be contoured
        values_TernVid = np.vstack([x_TernVid, y_TernVid])
        kernel_TernVid = st.gaussian_kde(values_TernVid)
        
        # Formatting the input data and the grid system to a single format.
        data_TernVid = np.reshape(kernel_TernVid(positions_TernVid).T, xx_TernVid.shape)
        data_TernVid_list.append(data_TernVid) # Save contour data to list, one for each video frame.
        
        xx_TernVid_list.append(xx_TernVid) # Save grid data to list, one for each video frame.
        yy_TernVid_list.append(yy_TernVid)
    
    fig, ax = plt.subplots()
    anim_TernVid = animation.FuncAnimation(fig, animate_TernVid, init_func=init_TernVid, frames=NumberOfSteps, repeat=False)
    ax.axis('off')
    pillowwriter_TernVid = animation.PillowWriter(fps)
    anim_TernVid.save(FigureSavePath + FileName +  '_Ternary_Contour_Video_' + type +'Axis.gif', writer=pillowwriter_TernVid)
