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

#Defino una probabilidad para encontrar la función de distribución en cada digito
def prob(x_train,vector,m,sig):
    Ps=1
    for i in x_train:
        comp=np.dot(i,vector)
        Ps=Ps*1/(sig*np.sqrt(2*np.pi))*np.exp(-1/2*((comp-m)/sig)**2) 
    return Ps
#Ahora encuentro los parametros m y sig de cada digito para las primeras 4 componentes principales
sigl=np.zeros((10,4))
m=np.zeros((10,4))
for i in range(10):
    x_t=x_train[y_train==i]
    for i2 in range(4):
        m[i,i2]=np.mean(x_t@vectores[:,i2])
        Pm=-np.inf
        for sig in np.linspace(0.001,20,1000):
            Pp=prob(x_t,vectores[:,i],m[i,i2],sig)
            if Pp>Pm:
                Pm=Pp
                sigl[i,i2]=sig
#Ahora defino una función  que me diga que tan probable es que un dato se encuentre en un digito
def probdig(x,dig):
    Probl=np.zeros(4)
    for i in range(4):
        comp=np.dot(x,vectores[:,i])
        Probl[i]=1/(sigl[dig,i]*np.sqrt(2*np.pi))*np.exp(-1/2*((comp-m[dig,i])/sigl[dig,i])**2)
    P=np.prod(Probl)
    return P
#Y definimos un predictor, que me dice cual es el digito mas probable
def predict(x):
    Probs=np.zeros(10)
    for i in range(10):
        Probs[i]=probdig(x,i)
    return np.argmax(Probs)
#Podemos evaluar la eficiencia tomando muestras y haciendo las predicciones
Predict=np.zeros(500)
etiqueta=np.zeros(500)
for muestra in range(500):
    xm=x_test[muestra]
    Predict[muestra]=predict(xm)
    etiqueta[muestra]=y_test[muestra]
sum(etiqueta==Predict)/500











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

#Defino una probabilidad para encontrar la función de distribución en cada digito
def prob(x_train,vector,m,sig):
    Ps=1
    for i in x_train:
        comp=np.dot(i,vector)
        Ps=Ps*1/(sig*np.sqrt(2*np.pi))*np.exp(-1/2*((comp-m)/sig)**2) 
    return Ps
NC=20
#Ahora encuentro los parametros m y sig de cada digito para las primeras NC componentes principales
sigl=np.zeros((10,NC))
m=np.zeros((10,NC))
for i in range(10):
    x_t=x_train[y_train==i]
    for i2 in range(NC):
        m[i,i2]=np.mean(x_t@vectores[:,i2])
        Pm=-np.inf
        for sig in np.linspace(0.001,20,1000):
            Pp=prob(x_t,vectores[:,i2],m[i,i2],sig)
            if Pp>Pm:
                Pm=Pp
                sigl[i,i2]=sig
                
def probdig(x,dig,NC):
    Probl=np.zeros(NC)
    for i in range(NC):
        comp=np.dot(x,vectores[:,i])
        Probl[i]=1/(sigl[dig,i]*np.sqrt(2*np.pi))*np.exp(-1/2*((comp-m[dig,i])/sigl[dig,i])**2)
    P=np.prod(Probl)
    return P
#Y definimos un predictor, que me dice cual es el digito mas probable
def predict(x,NC):
    Probs=np.zeros(10)
    for i in range(10):
        Probs[i]=probdig(x,i,NC)
    return np.argmax(Probs)
#Hacemos predicciones con NC parametros
NC=4
Predict=np.zeros(500)
etiqueta=np.zeros(500)
for muestra in range(500):
    xm=x_test[muestra]
    Predict[muestra]=predict(xm,NC)
    etiqueta[muestra]=y_test[muestra]
print(sum(etiqueta==Predict)/500)

#Podemos plotear la eficiencia en función del número de parametros
PN=np.zeros(10)
for i in range(1,11):
    print(i)
    Predict=np.zeros(500)
    etiqueta=np.zeros(500)
    for muestra in range(500):
        xm=x_test[muestra]
        Predict[muestra]=predict(xm,i)
        etiqueta[muestra]=y_test[muestra]
    PN[i-1]=sum(etiqueta==Predict)/500
plt.plot(range(1,11),PN)
