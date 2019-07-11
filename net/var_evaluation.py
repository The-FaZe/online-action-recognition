from functools import reduce

def Evaluation(score_list,psi=8.5):
  """ This function aims to evaluate the feasibility of the obtained 
      scores using a statistical evaluation based on a threshold equal to the 
      variance of the top k scores and their intermediate average.
  """
  score_list.sort(reverse = True)
  
  A = Avg(score_list)
  B = VAR(score_list)
  
  if B > A*psi:  #psi = 8.5 ---> 9
    return True
  else:
    return False
  

def VAR(score_list):
  """ Calculate variance
  """
  X_2 = raise_2(score_list)
  return (sum(X_2)/len(score_list))-Avg(score_list)**2
  

def Avg(score_list): 
  """ This is the fastest way for calcuating average, 
      better than built in function called "mean"
  """
  return reduce(lambda a, b: a + b, score_list) / len(score_list) 


def raise_2(score_list):
  """To raise the elemnts of the list to power 2
  """
  return [x**2 for x in score_list]