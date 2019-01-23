#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 19:40:01 2018

@author: abdullah
"""
import getopt
import sys


    
    

def get_arguments(argv, code_name):
    """
    the function receivs inputs from terminal and returns a tuple contains
    (<input file directory>,<output file directory>)
        
    -h: means help
    -i: input path to ucf101 rgp dataset
    -o: output path to the extrcted train, and test files
    --inputPath=: input path to ucf101 rgp dataset
    --outputPath=: output path to the extrcted train, and test files
    """
    input_path = None
    output_path = None
    try:
        options, arguments = getopt.getopt(argv, "hi:o:",["inputPath=", "outputPath=",
                                                          "help"])
        #print(options)
    except getopt.GetoptError:
        raise ValueError("Invalid input argument try", code_name+" -h")
        #sys.exit
        
    for (opt, arg) in options:
        #print(opt, arg)
        
        if (opt == '-h' or opt == "--help") :
            print(code_name,"Command Line Argument Help")
            print("Usage:", '"python3 ', code_name,'-i <INPUT Path> -o <OUTPUT Path>".' )
            print("-i: input path to ucf101 rgp dataset, or you can use '--inputPath='.")
            print("-o: output path to the extrcted train, and test files,",
                  "or you can use '--outputPath='.")
            print("-h: means help, or you can use '--help'.")
            
            sys.exit() #exiting the code
            
        
        
        elif opt=='-i' or opt =='--inputPath':
            
            input_path = arg.split(" ")[-1] #taking string without spaces
    
                    
            
        elif opt=='-o' or opt =='--outputPath':
            
            output_path = arg.split(" ")[-1] #taking string without spaces
            
                     
             
    #if there is no input arguments           
    if len(options) ==0:
        raise ValueError("Yod did not input any argument, Please type " +
                        "'python3 " + str(code_name) +" -h'")
            
        
    #if the user did not input input or output path
    
    if input_path == None or input_path == '':
        raise ValueError("You must specify the INPUT file"+
            " path.\n" + "For more info type 'python3 " + str(code_name) +" -h'")
        
    elif output_path == None or output_path == '':
       raise ValueError("You must specify the OUTPUT file"+
            " path.\n" + "For more info type 'python3 " + str(code_name) +" -h'")
        
    return input_path, output_path
            
            





"""
Tset1
"""

#argv = ['--inputPath== /hamo/anaconda','--outputPath== hamo']
#code_name = 'code.py'
#print(get_arguments(argv, code_name))

"""
Test2
"""
#argv = ['-outputPath= hamo']
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


#if __name__=="__main__":
#    print(get_arguments(sys.argv[1:], code_name="hamo.py"))
           
            
        
        
    