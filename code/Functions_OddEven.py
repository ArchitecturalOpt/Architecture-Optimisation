import numpy as np
import math
import os

def LineToArrayM_OddEven(a): #Function to read the lines from the files. a is the line to take
  b=[]  #List to take the str characters
  c=[]  #List to take the values
  comaI = -1  #Index for the separation of the values
  #print(a)
  for i in range (0,len(a)):  #Take a look at all the characters of a line
    if (a[i]!=" " and a[i]!="\n" and a[i]!="]" and a[i]!="["):  #Don't take non desired characters
      b.append(a[i])  #Take the character
    if (a[i]=="]"): #Stop if arrived at the end
      break
  for j in range (0,len(b)):  #Separate the values using the coma positions
    if b[j]==",":
      test = ''.join(b[(comaI+1):(j)])
      c.append(test)
      comaI=j
    elif j==len(b)-1: #If we are at the last character, take the last value
      test = ''.join(b[(comaI+1):(j+1)])
      c.append(test)
  c2 = [float(z) for z in c]  #Make it all float
  return(c2)


def reading_OddEven(lines,L1,L2,L3):
  index1=[] #Start index of matrix for the (all) neuron
  index2=[] #End idex of the neuron
  Weights=[]

#This loop will get us all the starting and ending matrices index
  for k in range (0,len(lines)-1):
    temp = lines[k]
    if temp[0:6]=="tensor":
      index1.append(k)                          #get all the lines starting a matrix
    elif temp[0:9]=="        [":
      index1.append(k)
  for k in range (0,len(lines)-1):
    temp = lines[k]
    if temp[len(temp)-3]=="]":
      index2.append(k)                          #get all the lines ending a matrix
    elif len(temp)>23 and temp[len(temp)-23]=="]":
      index2.append(k)
    elif len(temp)>40 and temp[len(temp)-40]=="]":
      index2.append(k)
    elif len(temp)>20 and temp[len(temp)-20]=="]":
      index2.append(k)


#Some values provided by the code are not useful in our case
  index1.pop()

  index2.pop()
# Delete all the bias of each neuron
  del index1[L1]
  del index2[L1]
  del index1[L1]
  del index2[L1]
  del index1[L1+L2]
  del index2[L1+L2]
  del index1[L1+L2]
  del index2[L1+L2]
  del index1[L1+L2+L3]
  del index2[L1+L2+L3]
  del index1[L1+L2+L3]
  del index2[L1+L2+L3]

#Start the reading for each couple of layers

#Reading for Input/Hidden1
  input = []
  for m2 in range (0,L1):  #Read the L1th first matrices
    d=[]
    d.append(LineToArrayM_OddEven(lines[index1[m2]][9:len(lines[index1[m2]])])) #Read the first line of a matrix
    for m22 in range (1,index2[m2]-index1[m2]):                 #Read all the middle lines of the matrix
      d.append(LineToArrayM_OddEven(lines[index1[m2]+m22]))             #Read the last line of a matrix
    d.append(LineToArrayM_OddEven(lines[index2[m2]]))
    d = [item for sublist in d for item in sublist]
    input.append(d)
  Weights.append(input)

#Reading for Hidden1/Hidden2
  input = []
  for m2 in range (L1,L1+L2):
    d=[]
    d.append(LineToArrayM_OddEven(lines[index1[m2]][9:len(lines[index1[m2]])]))
    for m22 in range (1,index2[m2]-index1[m2]):
      d.append(LineToArrayM_OddEven(lines[index1[m2]+m22]))
    d.append(LineToArrayM_OddEven(lines[index2[m2]]))
    d = [item for sublist in d for item in sublist]
    input.append(d)
  Weights.append(input)

#Reading for Hidden2/Hidden3
  input = []
  for m2 in range (L1+L2,L1+L2+L3):
    d=[]
    d.append(LineToArrayM_OddEven(lines[index1[m2]][9:len(lines[index1[m2]])]))
    for m22 in range (1,index2[m2]-index1[m2]):
      d.append(LineToArrayM_OddEven(lines[index1[m2]+m22]))
    d.append(LineToArrayM_OddEven(lines[index2[m2]]))
    d = [item for sublist in d for item in sublist]
    input.append(d)
  Weights.append(input)

  input = []
  for m2 in range (L1+L2+L3,L1+L2+L3+2):
    d=[]
    d.append(LineToArrayM_OddEven(lines[index1[m2]][9:len(lines[index1[m2]])]))
    if index1[m2]!=index2[m2]:
      for m22 in range (1,index2[m2]-index1[m2]):
        d.append(LineToArrayM_OddEven(lines[index1[m2]+m22]))
      d.append(LineToArrayM_OddEven(lines[index2[m2]]))
    d = [item for sublist in d for item in sublist]
    input.append(d)
  Weights.append(input)

  for u in range (0,len(Weights)):
    Weights[u] = np.asarray(Weights[u])

  return Weights

def InvTemp_Spectral_Radius_OddEven (W, axis=0):
    L,M = W.shape

    var = W.var()
    somma =0
    beta3h = np.sqrt(var) * np.sqrt(64*3) * (1/np.sqrt(2))
    beta4l = np.sqrt(var) * np.sqrt(64*3+784) * (1/np.sqrt(2))
    beta5l = np.sqrt(var) * np.sqrt(64*3+784+10) * (1/np.sqrt(2))

    for i in range (0,len(W)):
      for j in range(0,len(W[i])):
        somma = somma + W[i][j]*W[i][j]
    return np.sqrt(var), L, M, beta3h, beta4l, beta5l, var

#Get the values for all numbers of the matrix
def percentages_OddEven(lines):

  temp=[]
  values=[]
  B=[]
  error=[]

  for i in range(0,len(lines)):
    temp.append(LineToArrayM_OddEven(lines[i][1:len(lines[i])]))
    temp[i]=[int(c) for c in temp[i]]
    mean = sum(temp[i])
    B.append(np.array(temp[i])*100/float(mean))
    p=temp[i][i]/mean
    error.append(math.sqrt(p*(1-p)/mean))   #The binary uncertainty is computed, it can be added to the outputs


  for i in range (0, len(B)):
    values.append(B[i][i])
  return (values)

def getData_OddEven(name='seed1'):
    dataset='MNIST_OddEven'

    #Select the directory
    directory = 'Sets/'+dataset+'/'+name+'/'
    
    MEAN, WEIGHTS0, WEIGHTS1, WEIGHTS2, WEIGHTS3, WEIGHTSTOT=([] for i in range(6))
    
    i=0
    for filename in sorted(os.listdir(directory)):
        print(filename)
        with open(directory +'/'+filename+'/test.txt') as f1:
            lines1= f1.readlines()
        with open(directory+ '/'+filename+'/matrix_noisy.txt') as f2:
            lines2= f2.readlines()


        #Change the settings regarding the moving neurons layer: Li is for the number of neurons on the i-th layer
        #This must be modified regarding the moving of neurons performed during the training
        L1=64+i
        L2=64+i
        L3=64-2*i
        Weight = reading_OddEven(lines1,L1,L2,L3)
        WEIGHTS0.append(InvTemp_Spectral_Radius_OddEven(Weight[0]))
        WEIGHTS1.append(InvTemp_Spectral_Radius_OddEven(Weight[1]))
        WEIGHTS2.append(InvTemp_Spectral_Radius_OddEven(Weight[2]))
        WEIGHTS3.append(InvTemp_Spectral_Radius_OddEven(Weight[3]))
        MEAN.append(percentages_OddEven(lines2))  #Gets mean of right-recognition for each number
        i+=1

    newmean=[]
    for i in range(0,len(MEAN)):
        newmean.append(sum(MEAN[i])/float(len(MEAN[i]))) #Get the mean of right-recognition for all the numbers for a test
    
    WEIGHTSTOT.append(WEIGHTS0)
    WEIGHTSTOT.append(WEIGHTS1)
    WEIGHTSTOT.append(WEIGHTS2)
    WEIGHTSTOT.append(WEIGHTS3)
    
    return (WEIGHTSTOT, newmean)

def find_extremes(vectors):
    if not vectors:
        return None, None

    flattened = np.concatenate(vectors)
    min_value = np.min(flattened)
    max_value = np.max(flattened)

    return min_value, max_value

def GetConf_OddEven(name='seed1'):
    data_dicts = []  # List to store dictionnaries for each file
    directory = 'Sets/MNIST_OddEven/'+name+'/'
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory +'/'+filename, 'table_noisy.txt')
        print(filename)
        real_numbers = []
        vectors = []
        guessed_numbers = []
        norm_vectors = []
        zero_cor = []
        zero_wro = []
        uno_cor = []
        uno_wro = []

    # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
    # Process the lines
        for i in range(0, len(lines)):
           line1 = lines[i].strip()
           sub1="["
           sub2="]"
           line1=line1.replace(sub1,"*")
           line1=line1.replace(sub2,"*")
           re=line1.split("*")
           vector_str=re[1]
           vector = [float(num_str) for num_str in vector_str.split()]
           vectors.append(vector)

        min_value, max_value = find_extremes(vectors)
        norm_vectors = [(np.array(vector) - min_value) / (max_value - min_value) for vector in vectors]
        norm_vectors = [norm_vector.tolist() for norm_vector in norm_vectors]  # Convert to lists
        i=0
        for i in range(0, len(lines)):
            line1 = lines[i].strip()
            try:
                if line1.split('/')[-1].split('_')[0][0]=='t':
                    real_number = (int(line1.split('/')[-1].split('_')[0][5])+1)%2
                else:
                  real_number = (int(line1.split('/')[3][0])+1)%2
                real_numbers.append(real_number)
                c = len(real_numbers)
            except IndexError:
                print("Error: Unable to extract the first number")

            guessed_number = int(line1.split()[-1])
            guessed_numbers.append(guessed_number)

            if norm_vectors:
                norm_vector = norm_vectors[c-1]  # Get the current normalized tensor of the iteration in the list
                if real_number == 0:
                    if guessed_number == 0:
                        sorted_vector = np.sort(norm_vector)
                        difference = sorted_vector[-1] - sorted_vector[-2]
                        zero_cor.append(difference)
                    else:
                        difference = norm_vector[0] - norm_vector[guessed_number]
                        zero_wro.append(difference)
                if real_number == 1:
                    if guessed_number == 1:
                        sorted_vector = np.sort(norm_vector)
                        difference = sorted_vector[-1] - sorted_vector[-2]
                        uno_cor.append(difference)
                    else:
                        difference = norm_vector[1] - norm_vector[guessed_number]
                        uno_wro.append(difference)
         
        media_zero_cor = np.mean(zero_cor)
        media_zero_wro = np.mean(zero_wro)
        media_uno_cor = np.mean(uno_cor)
        media_uno_wro = np.mean(uno_wro)
        media_zero_tot = np.mean(zero_cor + zero_wro)
        media_uno_tot = np.mean(uno_cor + uno_wro)
        media_tutti_cor = np.mean(uno_cor + zero_cor)
        media_tutti_wro = np.mean(uno_wro + zero_wro )
        media_tutti_tot = 0.5*media_zero_tot + 0.5*media_uno_tot

        data_dict = {
            'Real Number': real_numbers,
            'Guessed Number': guessed_numbers,
            'Tensor': vectors,
            'Normalized Tensors': norm_vectors,
            'Media Zero Coretto': media_zero_cor,
            'Media Zero Errato': media_zero_wro,
            'Media Zero Totale': media_zero_tot,
            'Media Uno Coretto': media_uno_cor,
            'Media Uno Errato': media_uno_wro,
            'Media Uno Totale': media_uno_tot,
            'Media Tutti Valori Corretti': media_tutti_cor,
            'Media Tutti Valori Errati': media_tutti_wro,
            'Media Tutti Valori Completi': media_tutti_tot
            }

    # Add the dictionary to the list
        data_dicts.append(data_dict)

    tutti_dict = {'Media Tutti Valori Corretti': [],
                  'Media Tutti Valori Errati': [],
                  'Media Tutti Valori Completi': []
                  }

    for data_dict in data_dicts:
        tutti_dict['Media Tutti Valori Corretti'].append(data_dict['Media Tutti Valori Corretti'])
        tutti_dict['Media Tutti Valori Errati'].append(data_dict['Media Tutti Valori Errati'])
        tutti_dict['Media Tutti Valori Completi'].append(data_dict['Media Tutti Valori Completi'])

    Conf=list(tutti_dict['Media Tutti Valori Completi'])
    return Conf