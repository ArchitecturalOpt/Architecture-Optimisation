import numpy as np
import os
import math

def LineToArrayM(a): #Function to read the lines from the files. a is the line to take
  b=[]  #List to take the str characters
  c=[]  #List to take the values
  comaI = -1  #Index for the separation of the values
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

def reading(lines,L1,L2,L3):
  index1=[] #Start index of matrix for the (all) neuron
  index2=[] #End index of the neuron
  Weights=[]

  #This loop will get us all the starting and ending matrices index
  for k in range (0,len(lines)-1):
    temp = lines[k]
    if temp[0:6]=="tensor":
      index1.append(k)#get all the lines starting a matrix
    elif temp[0:9]=="        [":
      index1.append(k)
  for k in range (0,len(lines)-1):
    temp = lines[k]
    if temp[len(temp)-3]=="]":
      index2.append(k)
  #get all the lines ending a matrix
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
    d.append(LineToArrayM(lines[index1[m2]][9:len(lines[index1[m2]])])) #Read the first line of a matrix
    for m22 in range (1,index2[m2]-index1[m2]):                 #Read all the middle lines of the matrix
      d.append(LineToArrayM(lines[index1[m2]+m22]))             #Read the last line of a matrix
    d.append(LineToArrayM(lines[index2[m2]]))
    d = [item for sublist in d for item in sublist]
    input.append(d)
  Weights.append(input)

  #Reading for Hidden1/Hidden2
  input = []
  for m2 in range (L1,L1+L2):
    d=[]
    d.append(LineToArrayM(lines[index1[m2]][9:len(lines[index1[m2]])]))
    for m22 in range (1,index2[m2]-index1[m2]):
      d.append(LineToArrayM(lines[index1[m2]+m22]))
    d.append(LineToArrayM(lines[index2[m2]]))
    d = [item for sublist in d for item in sublist]
    input.append(d)
  Weights.append(input)

  #Reading for Hidden2/Hidden3
  input = []
  for m2 in range (L1+L2,L1+L2+L3):
    d=[]
    d.append(LineToArrayM(lines[index1[m2]][9:len(lines[index1[m2]])]))
    if index1[m2]!=index2[m2]:
      for m22 in range (1,index2[m2]-index1[m2]):
        d.append(LineToArrayM(lines[index1[m2]+m22]))
      d.append(LineToArrayM(lines[index2[m2]]))
    d = [item for sublist in d for item in sublist]
    input.append(d)
  Weights.append(input)

  #Reading for Hidden3/Output
  input = []
  for m2 in range (L1+L2+L3,L1+L2+L3+10):
    d=[]
    d.append(LineToArrayM(lines[index1[m2]][9:len(lines[index1[m2]])]))
    if index1[m2]!=index2[m2]:
      for m22 in range (1,index2[m2]-index1[m2]):
        d.append(LineToArrayM(lines[index1[m2]+m22]))
      d.append(LineToArrayM(lines[index2[m2]]))
    d = [item for sublist in d for item in sublist]
    input.append(d)
  Weights.append(input)

  for u in range (0,len(Weights)):
    Weights[u] = np.asarray(Weights[u])

  return Weights, index2

def InvTemp (W, axis=0):
    L,M = W.shape

    var = W.var()
    mean = W.mean()
    somma =0
    beta1 = np.sqrt(var) * np.sqrt(64+64+64)
    betanew = np.sqrt(var) * np.sqrt(64+64+64) * (1/np.sqrt(2))
    for i in range (0,len(W)):
      for j in range(0,len(W[i])):
        somma = somma + W[i][j]*W[i][j]
        
    return beta1, somma, float(somma)/(L*M), float(somma)/(L), var, var**2, np.sqrt(var *192/2),np.sqrt(var), L, M, betanew, mean

def InvTemp_Spectral_Radius (W, axis=0):
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
def percentages(lines):

  temp=[]
  values=[]
  B=[]
  error=[]

  for i in range(0,len(lines)):
    temp.append(LineToArrayM(lines[i][1:len(lines[i])]))
    temp[i]=[int(c) for c in temp[i]]
    mean = sum(temp[i])
    B.append(np.array(temp[i])*100/float(mean))
    p=temp[i][i]/mean
    error.append(math.sqrt(p*(1-p)/mean))   #The binary uncertainty is computed, it can be added to the outputs


  for i in range (0, len(B)):
    values.append(B[i][i])
  return (values)

def getData(dataset='MNIST', name='NS2hot2cold'):
    if dataset=='MNIST_OddEven':
        print('Use function "GetConf_OddEven"')
        return 0
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
        if name == 'cold2hot':
            L1=64-2*i
            L2=64+i
            L3=64+i 
        elif name == 'in2out':
            L1=64+i
            L2=64-2*i
            L3=64+i 
        else:
            L1=64+i
            L2=64+i
            L3=64-2*i
        Weight, index2  = reading(lines1,L1,L2,L3)
        WEIGHTS0.append(InvTemp_Spectral_Radius(Weight[0]))
        WEIGHTS1.append(InvTemp_Spectral_Radius(Weight[1]))
        WEIGHTS2.append(InvTemp_Spectral_Radius(Weight[2]))
        WEIGHTS3.append(InvTemp_Spectral_Radius(Weight[3]))
        MEAN.append(percentages(lines2))  #Gets mean of right-recognition for each number
        i+=1

    newmean=[]
    for i in range(0,len(MEAN)):
        newmean.append(sum(MEAN[i])/float(len(MEAN[i]))) #Get the mean of right-recognition for all the numbers for a test
    
    WEIGHTSTOT.append(WEIGHTS0)
    WEIGHTSTOT.append(WEIGHTS1)
    WEIGHTSTOT.append(WEIGHTS2)
    WEIGHTSTOT.append(WEIGHTS3)
    
    return (WEIGHTSTOT, newmean)

def getSpectralRadii(WEIGHTSTOT):
    num_topol = len(WEIGHTSTOT[1]) #number of topologies in ine experiment i.e. 31 neuron movements
    num_hamb = 4 # 2 hamburger
    tot_neu_3h = 64*3
    beta = np.zeros((num_topol,num_hamb))
    alpha = np.zeros((num_topol,5)) #form factors 5 for each topology
    
    for i in range(num_topol):
        for j in range(num_hamb):
             
             beta[i][j] = WEIGHTSTOT[j][i][5]
             
    for i in range(num_topol):
        for j in range(num_hamb):
                     
             if j==0:
                 alpha[i][j] = WEIGHTSTOT[j][i][4]
                         
             elif j==1:
                 alpha[i][j] = WEIGHTSTOT[j][i][4]
                 alpha[i][j+1] = WEIGHTSTOT[j][i][3]

             elif j==3:

                 alpha[i][j] = WEIGHTSTOT[j][i][4]
                 alpha[i][j+1] = WEIGHTSTOT[j][i][3]
    
    alpha_3h=np.zeros((num_topol,3))

    for i in range(num_topol):
        l = np.arange(64,96)

        for j in range(3):

            if j<=1:
                alpha_3h[i][j] = l[i]/(tot_neu_3h)

            elif j>1:
                alpha_3h[i][j] = (192-2*l[i])/(tot_neu_3h)
    
    beta_3h = np.zeros((num_topol,2))
    for i in range(num_topol):
        for j in range(2):
            # Take the element 0 of each WEIGHTS vector because it is the sqrt(var)
            beta_3h[i][j] = WEIGHTSTOT[j+1][i][0]* np.sqrt(tot_neu_3h) * (1/np.sqrt(2))
            
    S_radius_3h_2 = []

    for i in range(len(beta_3h)):
        M = np.zeros((3,3))

        M[0][1] = ((beta_3h[i][0])**2)*(alpha_3h[i][1])
        M[1][0] = ((beta_3h[i][0])**2)*(alpha_3h[i][0])
        M[1][2] = ((beta_3h[i][1])**2)*(alpha_3h[i][2])
        M[2][1] = ((beta_3h[i][1])**2)*(alpha_3h[i][1])
        # The 2 in front of is due to the fact the actual matrix is with a scalar factor 2 in front of
        max_eigenv = 2*np.sqrt(M[0][1]*M[1][0]+M[1][2]*M[2][1])
        S_radius_3h_2.append(float(max_eigenv))
    return S_radius_3h_2

def find_extremes(vectors):
    if not vectors:
        return None, None

    flattened = np.concatenate(vectors)
    min_value = np.min(flattened)
    max_value = np.max(flattened)

    return min_value, max_value

def GetConf(dataset='MNIST', name='NS2hot2cold'):
    if dataset=='MNIST_OddEven':
        print('Use function "GetConf_OddEven"')
        return 0
    data_dicts=[]
    directory = 'Sets/'+dataset+'/'+name+'/'
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory +'/'+filename, 'table_noisy.txt')
        real_numbers = []
        vectors = []
        guessed_numbers = []
        norm_vectors = []
        zero_cor = []
        zero_wro = []
        uno_cor = []
        uno_wro = []
        due_cor = []
        due_wro = []
        tre_cor = []
        tre_wro = []
        quattro_cor = []
        quattro_wro = []
        cinque_cor = []
        cinque_wro = []
        sei_cor = []
        sei_wro = []
        sette_cor = []
        sette_wro = []
        otto_cor = []
        otto_wro = []
        nove_cor = []
        nove_wro = []


        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Process the lines
        i=0
        while i in range(0, len(lines)):
            line1 = lines[i].strip()
            line2 = lines[i + 1].strip()

            try:
                if line2[len(line2)-7]==']':
                    line1 = lines[i].strip()
                    line2 = lines[i + 1].strip()
                    vector_str1 = line1.split('[')[-1].strip()
                    vector_str2 = line2.split(']')[0].strip()
                    vector_str = vector_str1 + ' ' + vector_str2
                    i=i+2
                else:
                    line3 = lines[i + 2].strip()
                    line1 = lines[i].strip()
                    line2 = lines[i + 1].strip()
                    vector_str1 = line1.split('[')[-1].strip()
                    vector_str2 = line2.strip()
                    vector_str3 = line3.split(']')[0].strip()
                    vector_str = vector_str1 + ' ' + vector_str2 + ' ' + vector_str3
                    i=i+3
                vector = [float(num_str) for num_str in vector_str.split()]

                vectors.append(vector)
            except IndexError:
                print("Error: Unable to extract the vector")

        min_value, max_value = find_extremes(vectors)
        norm_vectors = [(np.array(vector) - min_value) / (max_value - min_value) for vector in vectors]
        norm_vectors = [norm_vector.tolist() for norm_vector in norm_vectors]  # Convert to lists
        i=0
        while i in range(0, len(lines)):
            line1 = lines[i].strip()
            line2 = lines[i + 1].strip()
            try:
                real_number = int(line1.split('/')[-2])
                real_numbers.append(real_number)
                c = len(real_numbers)
            except IndexError:
                print("Error: Unable to extract the first number")

            try:
                if line2[len(line2)-7]==']':
                    guessed_number = int(line2.split()[-1])
                    guessed_numbers.append(guessed_number)
                    i=i+2
                else:
                    line3 = lines[i + 2].strip()
                    guessed_number = int(line3.split()[-1])
                    guessed_numbers.append(guessed_number)
                    i=i+3
            except IndexError:
                print("Error: Unable to extract the last number")
            except ValueError:
                guessed_number = int(''.join(filter(str.isdigit, line2.split()[-1])))
                print("Guessed Number:", guessed_number)
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
                if real_number == 2:
                    if guessed_number == 2:
                        sorted_vector = np.sort(norm_vector)
                        difference = sorted_vector[-1] - sorted_vector[-2]
                        due_cor.append(difference)
                    else:
                        difference = norm_vector[2] - norm_vector[guessed_number]
                        due_wro.append(difference)
                if real_number == 3:
                    if guessed_number == 3:
                        sorted_vector = np.sort(norm_vector)
                        difference = sorted_vector[-1] - sorted_vector[-2]
                        tre_cor.append(difference)
                    else:
                        difference = norm_vector[3] - norm_vector[guessed_number]
                        tre_wro.append(difference)
                if real_number == 4:
                    if guessed_number == 4:
                        sorted_vector = np.sort(norm_vector)
                        difference = sorted_vector[-1] - sorted_vector[-2]
                        quattro_cor.append(difference)
                    else:
                        difference = norm_vector[4] - norm_vector[guessed_number]
                        quattro_wro.append(difference)
                if real_number == 5:
                    if guessed_number == 5:
                        sorted_vector = np.sort(norm_vector)
                        difference = sorted_vector[-1] - sorted_vector[-2]
                        cinque_cor.append(difference)
                    else:
                        difference = norm_vector[5] - norm_vector[guessed_number]
                        cinque_wro.append(difference)
                if real_number == 6:
                    if guessed_number == 6:
                        sorted_vector = np.sort(norm_vector)
                        difference = sorted_vector[-1] - sorted_vector[-2]
                        sei_cor.append(difference)
                    else:
                        difference = norm_vector[6] - norm_vector[guessed_number]
                        sei_wro.append(difference)
                if real_number == 7:
                    if guessed_number == 7:
                        sorted_vector = np.sort(norm_vector)
                        difference = sorted_vector[-1] - sorted_vector[-2]
                        sette_cor.append(difference)
                    else:
                        difference = norm_vector[7] - norm_vector[guessed_number]
                        sette_wro.append(difference)
                if real_number == 8:
                    if guessed_number == 8:
                        sorted_vector = np.sort(norm_vector)
                        difference = sorted_vector[-1] - sorted_vector[-2]
                        otto_cor.append(difference)
                    else:
                        difference = norm_vector[8] - norm_vector[guessed_number]
                        otto_wro.append(difference)
                if real_number == 9:
                    if guessed_number== 9:
                        sorted_vector = np.sort(norm_vector)
                        difference = sorted_vector[-1] - sorted_vector[-2]
                        nove_cor.append(difference)
                    else:
                        difference = norm_vector[9] - norm_vector[guessed_number]
                        nove_wro.append(difference)
        media_zero_cor = np.mean(zero_cor)
        media_zero_wro = np.mean(zero_wro)
        media_uno_cor = np.mean(uno_cor)
        media_uno_wro = np.mean(uno_wro)
        media_due_cor = np.mean(due_cor)
        media_due_wro = np.mean(due_wro)
        media_tre_cor = np.mean(tre_cor)
        media_tre_wro = np.mean(tre_wro)
        media_quattro_cor = np.mean(quattro_cor)
        media_quattro_wro = np.mean(quattro_wro)
        media_cinque_cor = np.mean(cinque_cor)
        media_cinque_wro = np.mean(cinque_wro)
        media_sei_cor = np.mean(sei_cor)
        media_sei_wro = np.mean(sei_wro)
        media_sette_cor = np.mean(sette_cor)
        media_sette_wro = np.mean(sette_wro)
        media_otto_cor = np.mean(otto_cor)
        media_otto_wro = np.mean(otto_wro)
        media_nove_cor = np.mean(nove_cor)
        media_nove_wro = np.mean(nove_wro)
        media_zero_tot = np.mean(zero_cor + zero_wro)
        media_uno_tot = np.mean(uno_cor + uno_wro)
        media_due_tot = np.mean(due_cor + due_wro)
        media_tre_tot = np.mean(tre_cor + tre_wro)
        media_quattro_tot = np.mean(quattro_cor + quattro_wro)
        media_cinque_tot = np.mean(cinque_cor + cinque_wro)
        media_sei_tot = np.mean(sei_cor + sei_wro)
        media_sette_tot = np.mean(sette_cor + sette_wro)
        media_otto_tot = np.mean(otto_cor + otto_wro)
        media_nove_tot = np.mean(nove_cor + nove_wro)
        media_tutti_cor = np.mean(uno_cor + zero_cor + due_cor + tre_cor + quattro_cor + cinque_cor + sei_cor+ sette_cor + otto_cor + nove_cor)
        media_tutti_wro = np.mean(uno_wro + zero_wro + due_wro + tre_wro + quattro_wro + cinque_wro + sei_wro+ sette_wro + otto_wro + nove_wro)
        media_tutti_tot = 0.0980*media_zero_tot + 0.1135*media_uno_tot + 0.1032*media_due_tot + 0.1010*media_tre_tot + 0.0982*media_quattro_tot + 0.0892*media_cinque_tot + 0.0958*media_sei_tot + 0.1028*media_sette_tot + 0.0974*media_otto_tot + 0.1009*media_nove_tot

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
        'Media Due Coretto': media_due_cor,
        'Media Due Errato': media_due_wro,
        'Media Due Totale': media_due_tot,
        'Media Tre Coretto': media_tre_cor,
        'Media Tre Errato': media_tre_wro,
        'Media Tre Totale': media_tre_tot,
        'Media Quattro Coretto': media_quattro_cor,
        'Media Quattro Errato': media_quattro_wro,
        'Media Quattro Totale': media_quattro_tot,
        'Media Cinque Coretto': media_cinque_cor,
        'Media Cinque Errato': media_cinque_wro,
        'Media Cinque Totale': media_cinque_tot,
        'Media Sei Coretto': media_sei_cor,
        'Media Sei Errato': media_sei_wro,
        'Media Sei Totale': media_sei_tot,
        'Media Sette Coretto': media_sette_cor,
        'Media Sette Errato': media_sette_wro,
        'Media Sette Totale': media_sette_tot,
        'Media Otto Coretto': media_otto_cor,
        'Media Otto Errato': media_otto_wro,
        'Media Otto Totale': media_otto_tot,
        'Media Nove Coretto': media_nove_cor,
        'Media Nove Errato': media_nove_wro,
        'Media Nove Totale': media_nove_tot,
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

    Conf=[]
    Conf=tutti_dict['Media Tutti Valori Completi']
    return Conf
