import os
import numpy as np
import matplotlib.pyplot as plt

#'MNIST', 'MNIST_OddEven' or 'Fashion'
dataset_folder='MNIST'

Name_file_list = [name for name in os.listdir('Data_saves/'+dataset_folder+'/')]
N_files=len(Name_file_list)
len_study=(len(list(np.genfromtxt('Data_saves/'+dataset_folder+'/'+Name_file_list[0]+'/Acc.out'))))

T1, T2, Conf, Acc, SR = (np.zeros(len_study) for i in range(5))
T1_err, T2_err, Conf_err, Acc_err, SR_err= (np.zeros(len_study) for i in range(5))

Norm = 0
i=0

for k in Name_file_list:
    Norm+=1
    
    Acc_temp = np.genfromtxt('Data_saves/'+dataset_folder+'/'+k+'/Acc.out')
    SR_temp = np.genfromtxt('Data_saves/'+dataset_folder+'/'+k+'/SR.out')
    Conf_temp = np.genfromtxt('Data_saves/'+dataset_folder+'/'+k+'/Conf.out')
    T1_temp = np.genfromtxt('Data_saves/'+dataset_folder+'/'+k+'/T1.out')
    T2_temp = np.genfromtxt('Data_saves/'+dataset_folder+'/'+k+'/T2.out')
    
    Acc+=Acc_temp
    SR+=SR_temp
    Conf+=Conf_temp
    T1+=T1_temp
    T2+=T2_temp
    
T1 = (1/Norm)*T1
T2 = (1/Norm)*T2
Acc = (1/Norm)*Acc
Conf = (1/Norm)*Conf
SR = (1/Norm)*SR

Norm = 0
i=0
for k in Name_file_list:
    Norm+=1
    Acc_temp = np.genfromtxt('Data_saves/'+dataset_folder+'/'+k+'/Acc.out')
    SR_temp = np.genfromtxt('Data_saves/'+dataset_folder+'/'+k+'/SR.out')
    Conf_temp = np.genfromtxt('Data_saves/'+dataset_folder+'/'+k+'/Conf.out')
    T1_temp = np.genfromtxt('Data_saves/'+dataset_folder+'/'+k+'/T1.out')
    T2_temp = np.genfromtxt('Data_saves/'+dataset_folder+'/'+k+'/T2.out')
    

    Acc_err += (Acc - Acc_temp)**2
    SR_err += (SR - SR_temp)**2
    Conf_err += (Conf - Conf_temp)**2
    T1_err += (T1 - T1_temp)**2
    T2_err += (T2 - T2_temp)**2

T1_err = np.sqrt( (1/Norm)*T1_err)
T2_err = np.sqrt( (1/Norm)*T2_err)
Acc_err = np.sqrt( (1/Norm)*Acc_err)
SR_err = np.sqrt( (1/Norm)*SR_err)
Conf_err = np.sqrt( (1/Norm)*Conf_err)

fig,ax = plt.subplots()
fig.set_size_inches(7.5, 5.0)
fig.subplots_adjust(right=.8)

x2=[]
for i in range (0,len_study):
  temp="$\\alpha_{"+str(i)+"}$"
  x2.append(temp)
  
if dataset_folder == 'Fashion':
    ax.errorbar(x2,SR,yerr=SR_err,color="black", marker="o",alpha = 0.7)
    ax.set_ylabel("$\\rho$($\\beta$,$\\alpha$)",fontsize=14,color='black')

    ax3=ax.twinx()
    ax3.errorbar(x2,T1,yerr=T1_err,color="blue", marker="o",alpha=0.2, label="Beta for HL1-HL2")
    ax3.errorbar(x2,T2,yerr=T2_err,color="red", marker="o", alpha=0.2,label="Beta for HL2-HL3")
    ax3.get_yaxis().set_ticks([])
    ax3.legend(loc='lower left')

    ax4=ax.twinx()
    ax4.errorbar(x2,np.array(Conf),yerr=Conf_err, marker="o",alpha=0.7, color = "darkgoldenrod")
    ax4.spines['right'].set_position(("axes", 1.15))
    ax4.set_ylabel("Rob",fontsize=14, color = "darkgoldenrod")
    ax4.set_ylim(0.1,0.3)

    ax5=ax.twinx()
    ax5.errorbar(x2,Acc, yerr=Acc_err,marker="o",color = "green",alpha=0.7)
    ax5.set_ylabel("Acc",fontsize=14, color = "green")
    ax5.set_ylim(60.7,80.2)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    
    plt.show()
    fig.savefig('images/stats_'+dataset_folder+'.png',format='png',bbox_inches='tight',dpi=100)
    
elif dataset_folder == 'MNIST':
    ax.errorbar(x2,SR,yerr=SR_err,color="black", marker="o",alpha = 0.7)
    ax.set_ylabel("$\\rho$($\\beta$,$\\alpha$)",fontsize=14,color='black')
    ax.set_ylim(37,52)

    # Variation of the betas from their average
    ax3=ax.twinx()
    ax3.errorbar(x2,T1,yerr=T1_err,color="blue", marker="o",alpha=0.2, label="Beta for HL1-HL2")
    ax3.errorbar(x2,T2,yerr=T2_err,color="red", marker="o", alpha=0.2,label="Beta for HL2-HL3")
    ax3.get_yaxis().set_ticks([])

    # Conf
    ax4=ax.twinx()
    ax4.errorbar(x2,np.array(Conf),yerr=Conf_err, marker="o",alpha=0.7, color = "darkgoldenrod")
    ax4.spines.right.set_position(("axes", 1.15))
    ax4.set_ylabel("Rob",fontsize=14, color = "darkgoldenrod")
    ax4.set_ylim(0.1,0.50)

    #Accuracy
    ax5=ax.twinx()
    ax5.errorbar(x2,Acc, yerr=Acc_err,marker="o",color = "green",alpha=0.7)
    ax5.set_ylabel("Acc",fontsize=14, color = "green")
    ax5.set_ylim(75,92.2)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    
    plt.show()
    fig.savefig('images/stats_'+dataset_folder+'.png',format='png',bbox_inches='tight',dpi=100)

elif dataset_folder == 'MNIST_OddEven':
    ax.errorbar(x2,SR,yerr=SR_err,color="black", marker="o",alpha = 0.7)
    ax.set_ylabel("$\\rho$($\\beta$,$\\alpha$)",fontsize=14,color='black')
    ax.set_ylim(37,47)

    # Variation of the betas from their average
    ax3=ax.twinx()
    ax3.errorbar(x2,T1,yerr=T1_err,color="blue", marker="o",alpha=0.2, label="Beta for HL1-HL2")
    ax3.errorbar(x2,T2,yerr=T2_err,color="red", marker="o", alpha=0.2,label="Beta for HL2-HL3")
    ax3.get_yaxis().set_ticks([])

    # CL
    ax4=ax.twinx()
    ax4.errorbar(x2,np.array(Conf),yerr=Conf_err, marker="o",alpha=0.7, color = "darkgoldenrod")
    ax4.spines.right.set_position(("axes", 1.15))
    ax4.set_ylabel("Rob",fontsize=14, color = "darkgoldenrod")
    ax4.set_ylim(0.75,0.950)

    #Accuracy
    ax5=ax.twinx()
    ax5.errorbar(x2,Acc, yerr=Acc_err,marker="o",color = "green",alpha=0.7)
    ax5.set_ylabel("Acc",fontsize=14, color = "green")
    ax5.set_ylim(92,95.2)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    
    plt.show()
    fig.savefig('images/stats_'+dataset_folder+'.png',format='png',bbox_inches='tight',dpi=100)