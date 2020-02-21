import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
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
NC=8
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

#Podemos plotear la eficiencia en función del número de parametros
PN=np.zeros(NC)
sx=x_test.shape[0]
for i in range(1,NC+1):
    #print(i)
    Predict=np.zeros(sx)
    etiqueta=np.zeros(sx)
    for muestra in range(sx):
        xm=x_test[muestra]
        Predict[muestra]=(predict(xm,i)==1)
        etiqueta[muestra]=(y_test[muestra]==1)
    PN[i-1]=sum(etiqueta==Predict)/sx
Predict=np.zeros(sx)
etiqueta=np.zeros(sx)
for muestra in range(sx):
    xm=x_test[muestra]
    Predict[muestra]=(predict(xm,np.argmax(PN))==1)
    etiqueta[muestra]=(y_test[muestra]==1)
sx=x_train.shape[0]
Predict2=np.zeros(sx)
etiqueta2=np.zeros(sx)
for muestra in range(sx):
    xm=x_train[muestra]
    Predict2[muestra]=(predict(xm,np.argmax(PN))==1)
    etiqueta2[muestra]=(y_train[muestra]==1)
plt.figure(figsize=(6,18))
plt.subplot(1,2,2)
MatConf=np.zeros((2,2))
for i in range(2):
    for i2 in range(2):
        MatConf[i-1,i2-1]=np.sum((etiqueta==i)*(Predict==i2))
P=MatConf[0,0]/(MatConf[0,0]+MatConf[0,1])
R=MatConf[0,0]/(MatConf[0,0]+MatConf[1,0])
F1=2*P*R/(P+R)
CNames=["1","0"]
plt.imshow(MatConf)
Names=[["TP","FN"],["FP","TN"]]
plt.xticks([0,1],CNames)
plt.yticks([0,1],CNames)
for i in range(2):
    for i2 in range(2):
        plt.text (i-0.3,i2,Names[i][i2]+"=%1.0f"%MatConf[i,i2] )
plt.title("Test: F1 = %4.3f"%F1)
plt.xlabel("Truth")
plt.ylabel("Predict")
plt.subplot(1,2,1)
MatConf2=np.zeros((2,2))
for i in range(2):
    for i2 in range(2):
        MatConf2[i-1,i2-1]=np.sum((etiqueta2==i)*(Predict2==i2))
CNames=["1","0"]
plt.imshow(MatConf2)
Names=[["TP","FN"],["FP","TN"]]
plt.xticks([0,1],CNames)
plt.yticks([0,1],CNames)
plt.xlabel("Truth")
plt.ylabel("Predict")
P=MatConf2[0,0]/(MatConf2[0,0]+MatConf2[0,1])
R=MatConf2[0,0]/(MatConf2[0,0]+MatConf2[1,0])
F1=2*P*R/(P+R)
for i in range(2):
    for i2 in range(2):
        plt.text (i-0.3,i2,Names[i][i2]+"= %1.0f"%MatConf2[i,i2])
plt.title("Train: F1 = %4.3f"%F1)
plt.savefig("matriz_de_confusión.png")
plt.show()