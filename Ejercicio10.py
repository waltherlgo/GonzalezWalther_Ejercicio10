import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

plt.figure(figsize=(15,5))
plt.subplot(2,3,1)
plt.title("Matriz de Covarianza")
plt.imshow(cov)

plt.subplot(2,3,2)
plt.title("Varianza explicada")
plt.plot(np.cumsum(valores)/np.sum(valores))
plt.xlabel("Componentes")
plt.ylabel("Fraccion")
max_comps = (np.count_nonzero((np.cumsum(valores)/np.sum(valores))<0.6))
print(max_comps+1) # Necesito este numero de componentes para tener al menos el 60 de la varianza.

plt.subplot(2,3,4)
plt.imshow(vectores[:,0].reshape(8,8))
plt.title('Primer Eigenvector')
plt.subplot(2,3,5)
plt.title('Segundo Eigenvector')
plt.imshow(vectores[:,1].reshape(8,8))
plt.subplot(2,3,6)
plt.title('Tercer Eigenvector')
plt.imshow(vectores[:,2].reshape(8,8))
plt.subplots_adjust(hspace=0.5)

def prob(x_train,vector,m,sig):
    Ps=1
    for i in x_train:
        comp=np.dot(i,vector)
        Ps=Ps*1/(sig*np.sqrt(2*np.pi))*np.exp(-1/2*((comp-m)/sig)**2) 
    return Ps
#Segun eso, tomo la media sobre los valores train, y luego evaluo las probabilidades en cada sigma
ones=y_train==1
x_t=x_train[ones]
sigl=np.zeros(4)
m=np.zeros(4)
for i in range(4):
    m[i]=np.mean(x_t@vectores[:,i])
    Pm=-np.inf
    for sig in np.linspace(0.001,20,1000):
        Pp=prob(x_t,vectores[:,i],m[i],sig)
        if Pp>Pm:
            Pm=Pp
            sigl[i]=sig
            

ones=y_train==2
x_t=x_train[ones]
sigl2=np.zeros(4)
m2=np.zeros(4)
for i in range(4):
    m2[i]=np.mean(x_t@vectores[:,i])
    Pm=-np.inf
    for sig in np.linspace(0.001,20,1000):
        Pp=prob(x_t,vectores[:,i],m2[i],sig)
        if Pp>Pm:
            Pm=Pp
            sigl2[i]=sig

muestra=114
y_train[muestra]

xo=x_train[y_train==1]
LP=np.array([])
LP2=np.array([])
for xm in xo:
    Probl=np.zeros(4)
    Probl2=np.zeros(4)
    #muestra=100
    #xm=x_test[muestra]
    for i in range(4):
        comp=np.dot(xm,vectores[:,i])
        Probl[i]=1/(sigl[i]*np.sqrt(2*np.pi))*np.exp(-1/2*((comp-m[i])/sigl[i])**2)
    LP=np.append(LP,np.prod(Probl))
    for i in range(4):
        comp=np.dot(xm,vectores[:,i])
        Probl2[i]=1/(sigl2[i]*np.sqrt(2*np.pi))*np.exp(-1/2*((comp-m2[i])/sigl2[i])**2)
    LP2=np.append(LP2,np.prod(Probl2))