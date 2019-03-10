#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 00:17:31 2019

@author: abdullah
"""

import numpy as np


class top_N(object):
    
    def __init__(self, classInd_textfile, N=5):
        """
        input:
            classInd_textfile: the text file that contains actions
            N: is an integer refrering to top N actions
        """

        self.N = N
        self.actions_list = self.load_actions_label(classInd_textfile)
        self.scores = None
        
        
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
            
        return sorted_indcies, top_N_actions, top_N_scores
    
    def __str__(self):
        
        """
        this method will be activated when doing print(object)
        """
        
        open_statement = "Top " + str(self.N) + " Actions.\n"
        
        #Handelling no scores' input
        try:
            if self.scores == None:
                return open_statement + "\nThere is no scores were given."
        except:
            pass
            
        
        action_satement = ''
        
        _, top_actions, top_N_scores = self.get_top_N_actions()
        
        for i in range(self.N):
            action_satement += top_actions[i] + " : " \
                            + "{0:.4f}".format(top_N_scores[i]*100) + '\n'
                            
        return open_statement + action_satement
    
    
if __name__ == '__main__':
    
    classInd_file = 'UCF_lists/classInd.txt' #text file name
    top5_actions = top_N(classInd_file, 5)
    scores = np.random.random(101)
    top5_actions.import_scores(scores)
    print(top5_actions.get_top_N_actions())
    print(top5_actions)
    