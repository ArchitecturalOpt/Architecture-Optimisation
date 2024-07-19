import numpy as np
import matplotlib.pyplot as plt
from Functions import getData, getSpectralRadii, GetConf
from Functions_OddEven import getData_OddEven, GetConf_OddEven

#'MNIST', 'MNIST_OddEven' or 'Fashion'
dataset_folder='MNIST'

name_folder='NS2hot2cold'

if dataset_folder=='MNIST_OddEven':
    WEIGHTSTOT, newmean = getData_OddEven(name=name_folder)
    Spec_Rad= getSpectralRadii(WEIGHTSTOT)
    Conf = GetConf_OddEven(name=name_folder)
    
else:
    WEIGHTSTOT, newmean = getData(dataset=dataset_folder, name=name_folder)
    Spec_Rad= getSpectralRadii(WEIGHTSTOT)
    Conf = GetConf(dataset=dataset_folder, name=name_folder)
    
np.savetxt('Data_saves/'+dataset_folder+'/'+name_folder+'/Acc'+'.out', newmean)

temp0=[]
temp1=[]
temp2=[]
temp3=[]
idx_var=6 #take var

for k in range (0,len(WEIGHTSTOT[1])):
  #temp0 and temp3 can be uncommented for visualising the variance of the weights between the input layer and the
  #firts hidden layer, or the weights between the third hidden layer and the output layer
  temp0.append(WEIGHTSTOT[0][k][idx_var])
  temp1.append(WEIGHTSTOT[1][k][idx_var])
  temp2.append(WEIGHTSTOT[2][k][idx_var])
  temp3.append(WEIGHTSTOT[3][k][idx_var])
  
np.savetxt('Data_saves/'+dataset_folder+'/'+name_folder+'/T0'+'.out', temp0)
np.savetxt('Data_saves/'+dataset_folder+'/'+name_folder+'/T1'+'.out', temp1)
np.savetxt('Data_saves/'+dataset_folder+'/'+name_folder+'/T2'+'.out', temp2)
np.savetxt('Data_saves/'+dataset_folder+'/'+name_folder+'/T3'+'.out', temp3)

np.savetxt('Data_saves/'+dataset_folder+'/'+name_folder+'/SR'+'.out', Spec_Rad)

np.savetxt('Data_saves/'+dataset_folder+'/'+name_folder+'/Conf'+'.out', Conf)


#These settings are fitting the plot for the training "NS2hot2cold"
fig,ax = plt.subplots()
x2=[]
for i in range (0,len(Spec_Rad)):
  temp="$\\alpha_{"+str(i)+"}$"
  x2.append(temp)

ax.plot(x2,Spec_Rad, label='Media Tutti Valori', marker='o',color='black')
ax.set_ylabel("$\\rho$($\\beta$,$\\alpha$)",fontsize=14,color='black')
plt.xticks(np.arange(4, 64, 4).tolist())
ax2=ax.twinx()
ax2.plot(x2, Conf,color="darkgoldenrod",marker="o", label='Mean of recognition for each topology', alpha=0.6)
ax2.set_ylabel("Rob",fontsize=14, color='darkgoldenrod')
ax.axvline(x=x2[8], ymin=0, ymax=1,color ='purple',ls='--')
ax2.axhline(y= Conf[8], xmin=0, xmax=1,color ='purple',ls='--')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
plt.show()
plt.savefig('images/SR_CONF.png')


fig,ax = plt.subplots()
T1=[]
T2=[]
for k in range (0,len(WEIGHTSTOT[0])):
  T1.append(WEIGHTSTOT[1][k][3])
  T2.append(WEIGHTSTOT[2][k][3])

ax.plot(x2,T1,color="blue", marker="o",alpha=0.6, label=r"$\beta^{(1)}$")
ax.plot(x2,T2,color="red", marker="o", alpha=0.6,label=r"$\beta^{(2)}$")
ax.set_ylabel("$\\beta$",fontsize=14,color='black')
ax.legend(loc='lower right', bbox_to_anchor=(0.2,0.01))
plt.xticks(np.arange(4, 64, 4).tolist())
ax3=ax.twinx()
ax3.plot(x2,newmean, label='Media Tutti Valori', marker='x',color='green')
ax3.set_ylabel("Acc",fontsize=14,color='green')
ax3.set_ylim(75,90)
ax3.axvline(x=x2[8], ymin=0, ymax=60,color ='purple',ls='--')
ax3.axhline(y= newmean[8], xmin=0, xmax=60,color ='purple',ls='--')
plt.show()
plt.savefig('images/TEMP_ACC.png')