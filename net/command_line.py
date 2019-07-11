#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 19:40:01 2018

@author: abdullah
"""
import getopt
import sys

check_word = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
def validWord(word):
    """
    the function check if the input word has valid english charcters
    returns True if it has 
    returns False if it hasn't
    """
    if word != None:
        for char in word:
            if char in check_word:
                return True
        
    return False
    



def get_arguments(argv, code_name):
    """
    the function receivs inputs from terminal and returns a tuple contains
    (<input file directory>, <output file directory>, <Text File Path>)
        
    -h: means help
    -i: input path to ucf101 rgp dataset
    -o: output path to the extrcted train, and test files
    -t: Text File path to the extrcted train, and test files
    --inputPath=: input path to ucf101 rgp dataset
    --outputPath=: output path to the extrcted train, and test files
    --TextFilePath= Text File path to the extrcted train, and test files
    """
    input_path = None
    output_path = None
    textFile_path = None
    
    try:
        options, arguments = getopt.getopt(argv, "hi:t:o:",["inputPath=", 
                                "outputPath=", "textFilePath=", "help"])
        #print(options)
    except getopt.GetoptError:
        raise ValueError("Invalid input argument try\n" + 
                         "Usage:" + '"python3 ' + code_name + ' -i <INPUT PATH>' +
                  ' -o <OUTPUT Path> -t <TEXT FILE PATH>".\n' +
                  "For more type 'python3 " + code_name + "-h'.")
        #sys.exit
        
    for (opt, arg) in options:
        #print(opt, arg)
        
        if (opt == '-h' or opt == "--help") :
            print(code_name,"Command Line Argument Help")
            print("Usage:", '"python3 ', code_name,' -i <INPUT PATH>',
                  ' -o <OUTPUT Path> -t <TEXT FILE PATH>".' )
            print("-i: Input path to ucf101 rgp dataset, or you can use '--inputPath='.")
            print("-o: Iutput path to the extrcted train, and test files,",
                  "or you can use '--outputPath='.")
            print("-t: Text File path to the extrcted train, and test files,",
                  "or you can use '--TextFilePath='.")
            
            print("-h: Means help, or you can use '--help'.")
            
            sys.exit() #exiting the code
            
        
        
        elif opt=='-i' or opt =='--inputPath':
            
            input_path = arg.split(" ")[-1] #taking string without spaces
    
                    
            
        elif opt=='-o' or opt =='--outputPath':
            
            output_path = arg.split(" ")[-1] #taking string without spaces
            
        elif opt == '-t' or opt == '--textFilePath':
            textFile_path = arg.split(" ")[-1] #taking string without spaces
            
                     
             
    #if there is no input arguments           
    if len(options) ==0:
        raise ValueError("Yod did not input any argument.\n" +
                        "Usage:" + '"python3 ' + code_name + ' -i <INPUT PATH>' +
                  ' -o <OUTPUT PATH> -t <TEXT FILE PATH>".\n' +
                  "For more type 'python3 " + code_name + "-h'.")
            
        
    #if the user did not input input or output path
    
    print("TEXT PATH: '", textFile_path,"'")
    flag = False
    if input_path == None or (not validWord(input_path)):
        flag = True
        print("You must specify the INPUT file PATH.")
        
    if output_path == None or (not validWord(output_path)):
        flag = True
        print("You must specify the OUTPUT file PATH.")
       
        
    if textFile_path == None or (not validWord(textFile_path)):
        flag = True
        print("You must specify the TEXT FILE PATH.") 
    
    if flag:
        print("Usage:", '"python3 ', code_name,'-i <INPUT PATH>',
                  ' -o <OUTPUT PATH> -t <TEXT FILE PATH>".' )
        
        print("For more info type 'python3 " + str(code_name) +" -h'.")
        sys.exit()
       
    return input_path, output_path, textFile_path
            
            





"""
Tset1
"""

#argv = ['--inputPath=/hamo/anaconda','--outputPath=hamo', '--textFilePath=nkf']
#code_name = 'code.py'
#print(get_arguments(argv, code_name))

"""
Test2
"""
#argv = ['-outputPath= hamo', '-i ']
#code_name = 'code.py'
#print(get_arguments(argv, code_name))

"""
Test3
"""
#argv = ['-inputPath= /gamo/anaconda']
#code_name = 'code.py'
#print(get_arguments(argv, code_name))
    

"""
Test4 no input
"""
#argv = ['']
#code_name = 'code.py'
#print(get_arguments(argv, code_name))

"""
Test 5 help
"""
#argv = ['-h']
#code_name = 'code.py'
#print(get_arguments(argv, code_name))

"""
Test 6 
"""
#argv = ['-i /hamo/anaconda','-o hamo']
#code_name = 'code.py'
#print(get_arguments(argv, code_name))

"""
Test 7
"""
#argv = ['-m']
#code_name = 'code.py'
#print(get_arguments(argv, code_name))


#if __name__=="__main__":
#    print(get_arguments(sys.argv[1:], code_name="hamo.py"))
           
            
        
        
    