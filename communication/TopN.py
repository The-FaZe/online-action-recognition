#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 00:17:31 2019

@author: abdullah
"""

import numpy as np
import cv2


class Top_N(object):
    
    client_flag = True #if import_indecies_top_N_scores it will print using the actual 
                       #scores from the server
                        #if import_scores were called it will print its tops cores
                        #(slient scores)
    def __init__(self, classInd_textfile, N=5):
        """
        input:
            classInd_textfile: the text file that contains actions
            N: is an integer refrering to top N actions
        """

        self.N = N
        self.actions_list = self.load_actions_label(classInd_textfile)
        self.scores = None
        self.top_N_scores = None
        self.top_N_actions = None
        self.indecies = None
        
        
    def load_actions_label(self, classInd):
        """
        returns a list of the actions ordered by the list index
        input:
            classInd: of type string refering the input textfile
            output:
                action_label: of type list
        """
        action_label_list=[]
        with open(classInd) as f:
            content = f.readlines()
      
        f.close()
        for line in content:
            action_label_list.append(line.split(' ')[-1].rstrip()) #splitting the number from the action
                                                            #and removing the '/n' using 
                                                            #rstrip() method
    
          
        return action_label_list    
    
    
    def import_scores(self, scores):
        """
        the method impoetes the socres of the actions
        """
        Top_N.client_flag = False
        self.scores = scores
    
    def get_top_N_actions(self):
        """
        input:
            scores: of type numpay 1d array (row vector) that contains actions'
                    scores
        returns a tuple (top N actions' indicies of type numpy array,
                                                     list of type N action,
                                                     numpy array of top N cores)
        return False if the scores were not given
        
        """
        #Handelling no scores' input
        try:
            if self.scores == None:
                return False
        except:
            pass
                
        sorted_indcies = np.argsort(self.scores)[::-1] #soring the indicies from 
                                                       #the biggesr to the loewst
                                                       
        
        sorted_indcies = sorted_indcies[:self.N]     #taking top N actioms
        
        top_N_actions = []
        top_N_scores  = []
        for i in sorted_indcies:
            top_N_actions.append(self.actions_list[i])
            top_N_scores.append(self.scores[i])
        
        self.top_N_scores = top_N_scores
        self.top_N_actions = top_N_actions
        self.indecies = sorted_indcies
        
            
        return sorted_indcies, top_N_actions, top_N_scores
    
    
    def import_indecies_top_N_scores(self, tuple_input):
        """
        the method takes a input tuple (indecies, scores), and import it to the
        class
        
        """
        Top_N.client_flag = True
        self.indecies, self.top_N_scores = tuple_input
        


    def add_scores(self,frame_,x=480,y=380,j=20,font = cv2.FONT_HERSHEY_SIMPLEX
        ,fontScale = 0.4,fontcolor=(255,255,255),lineType=1):
        c = 0
        for i in self.indecies:
                l = j*c
                s = self.actions_list[i]+':'+"{0:.4f}".format(self.top_N_scores[c]*100)
                cv2.putText(frame_,s, (x,y+l) 
                    ,font, fontScale, fontcolor, lineType)
                c +=1

                
    def index_to_actionString(self):
        
        """
        returns the list of actions' string
        """
        #Handelling no scores' input
        try:
            if self.indecies == None:
                return False
        except:
            pass
        
        top_N_actions = []
        for i in self.indecies:
            top_N_actions.append(self.actions_list[i])
            
        self.top_N_actions = top_N_actions
        
        return top_N_actions
            
        
    def __str__(self):
        
        """
        this method will be activated when doing print(object)
        """
        
        open_statement = "Top " + str(self.N) + " Actions.\n"
        
        #Handelling no scores' input
        try:
            if self.scores == None and self.indecies == None:
                return open_statement + "\nThere is no scores were given."
        except:
            pass
        
        try:
            if self.top_scores == None:
                return open_statement + "\nThere is no scores were given."
        except:
            pass
            
        if Top_N.client_flag: #if import_indecies_top_N_scores is called
            self.index_to_actionString()
        else: #if import_scores were called
            self.get_top_N_actions()
            
        action_satement = ''
        
        
        
        for i in range(self.N):
            action_satement += self.top_N_actions[i] + " : " \
                            + "{0:.4f}".format(self.top_N_scores[i]*100) + '\n'
                            
        return open_statement + action_satement
    
    
    
if __name__ == '__main__':
    
    classInd_file = 'UCF_lists/classInd.txt' #text file name
    top5_actions = Top_N(classInd_file)
    scores = np.random.random(101)
    
    #at the server side
    #top5_actions.import_scores(scores)
    #print(top5_actions.get_top_N_actions())
    
    #at the client side
    #top5_actions.import_indecies_top_N_scores(([1, 2, 90, 4, 5], np.random.random)
    #frame = np.random.randint(0, 7, (640,480,3),dtype=np.uint8)
    #print(frame.shape)
    #cv2.imshow('frame', frame)
    #cv2.waitKey(0)
    #top5_actions.add_scores(frame)
    #cv2.imshow('frame', frame)
    #cv2.waitKey(0)