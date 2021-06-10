from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.tix import *
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import os.path

#declaracion de la ventana tkinter, tamaño y nombre
root = Tk()
root.geometry("600x580")
root.title("Perceptron Multicapa usado como red neuronal para predecir el cáncer de mama")
root.config(bg="green")

#variables de tkinter:
filen = StringVar(value="") #para guardar la ruta del archivo
trainp = IntVar() #para guardar el porcentaje de datos de entrenamiento
maxit = IntVar() #numero maximo de iteraciones
nnodos = StringVar() #cantidad de nodos por capa
activ = StringVar(value="relu") #para especificar función de activacion
solv = StringVar(value="adam") #para especificar el metodo para la optimizacion de pesos.
graphics = StringVar(value="Si")
preciB = IntVar() #precision para Benignas
preciM = IntVar() #precision para Malignas
#matriz de confusion:
cantBc = IntVar() 
cantBi = IntVar()
cantMc = IntVar()
cantMi = IntVar()

#funcion para saber si un numero es un entero
def es_entero(variable):
   try:
      int(variable)
      return True
   except:
      return False

#funcion para obtener la ruta de un archivo
def abrir():
   ruta=askopenfilename()
   filen.set(ruta)

#funcion para graficar los datos mostrados
#def graficar():
       


#funcion que se llama cuando se apreta el boton para entrenar la red neuronal y probarla
def entrenar_probar():
   extension = os.path.splitext(filen.get())[1] #se obtiene la extension del archivo
   if (extension == ".csv"): #se comprueba si la extension es csv
      mamd = pd.read_csv(filen.get())
   else: #se lanza mensaje de error sino es csv
      messagebox.showwarning(message="La extensión del archivo seleccionado debe ser igual a '.csv'.", title="Extensión incorrecta")
      return

   if (trainp.get() < 100 and trainp.get() > 2): #se comprueba si el porcentaje de entrenamiento es adecuado
      tsize = 1 - (trainp.get()/100)
   else: #se lanza mensaje de error sino es así
      messagebox.showwarning(message="El porcentaje de datos de entrenamiento debe ser menor a 100% y mayor a 2%.", title="Porcentaje incorrecto")
      return
   training_set, validation_set = train_test_split(mamd, test_size = tsize, random_state = 21) #se divide la data de entrenamiento y la de prueba
   X_train = training_set.iloc[:,0:-1].values #obtiene los valores (bi-rads, age, shape, margin, density)
   Y_train = training_set.iloc[:,-1].values #obtiene los valores severity
   X_val = validation_set.iloc[:,0:-1].values #obtiene los valores (bi-rads, age, shape, margin, density)
   y_val = validation_set.iloc[:,-1].values #obtiene los valores severity
   stnods = nnodos.get()
   hlsnodos = []
   for x in stnods.split(','): 
      if es_entero(x): #se comprueba que las cantidades de nodos por capa sean enteros
         hlsnodos.append(int(x))
      else: #se lanza mensaje de error sino es asi
         messagebox.showwarning(message="La cantidad de nodos por cada capa debe ser un numero entero.", title="Información incorrecta")
         return
   if (maxit.get() < 1000000 and maxit.get() > 10): #se comprueba si el numero maximo de iteraciones es adecuado
      #se crea y establece la red neuronal (MLP)
      classifier = MLPClassifier(hidden_layer_sizes=hlsnodos, max_iter=maxit.get(),activation = activ.get(),solver=solv.get(),random_state=1)
   else: #se lanza mensaje de error sino es asi
      messagebox.showwarning(message="El numero maximo de iteraciones debe ser menor a 1000000 y mayor a 10", title="Numero incorrecto")
      return
   classifier.fit(X_train, Y_train) #se entrena el MLP
   y_pred = classifier.predict(X_val) #se prueba el MLP
   cm = confusion_matrix(y_pred, y_val) #se crea la matriz de confusion
   #se actualizan datos para ser mostrados en pantalla
   preciB.set((cm[0][0]/(cm[0][0]+cm[1][0]))*100) #% precision diagnosticar tumor benigno
   preciM.set((cm[1][1]/(cm[0][1]+cm[1][1]))*100) #% precision diagnosticar tumor maligno
   cantBc.set(cm[0][0]) #cantidad de tumores benignos diagnosticados correctamente
   cantBi.set(cm[1][0]) #cantidad de tumores benignos diagnosticados incorrectamente
   cantMc.set(cm[1][1]) #cantidad de tumores malignos diagnosticados correctamente
   cantMi.set(cm[0][1]) #cantidad de tumores malignos diagnosticados incorrectamente
   #if (graphics.get() == "Si"): graficar()

#mensaje en tkinter de selección de archivos, boton y caja de texto respectiva
mensArchivo = Label(root, text="Selecciona el archivo con los datos ", background="orange")
mensArchivo.place(x=30, y=20)
entryArchivo = Entry(root, textvariable=filen, width=70)
entryArchivo.place(x=30, y=50)
botonAbrirArchivo =Button(root,text="Seleccionar archivo", command=abrir)
botonAbrirArchivo.place(x=470, y=48)

#mensajes en tkinter para ingreso de datos afines al entrenamiento y prueba de la red neuronal y cajas de texto respectivas
mensTrain = Label(root, text="Ingrese el porcentaje de datos de entrenamiento (%): ", background="yellow")
mensTrain.place(x=30, y=100)
entryTrain = Entry(root, textvariable=trainp, width=10)
entryTrain.place(x=500, y=100)
mensMaxit = Label(root, text="Ingrese el numero maximo de iteraciones: ", background="orange")
mensMaxit.place(x=30, y=130)
entryMaxit = Entry(root, textvariable=maxit, width=10)
entryMaxit.place(x=500, y=130)
mensNnodos = Label(root, text="Ingrese el numero de nodos por capa ordenadamente separados por comas: ", background="yellow")
mensNnodos.place(x=30, y=160)
entryNnodos = Entry(root, textvariable=nnodos, width=60)
entryNnodos.place(x=30, y=190)

#cajas de seleccion y labels para seleccionar funcion de activacion y metodo de optimizacion
mensActive = Label(root, text="Función de activación: ", background="orange")
mensActive.place(x=30, y=220)
combActive = ttk.Combobox(root, values=["relu", "logistic", "tanh", "identity"], state='readonly', textvariable=activ)
combActive.place(x=30, y=250)
mensSolver = Label(root, text="Optimización de pesos: ", background="yellow")
mensSolver.place(x=200, y=220)
combSolver = ttk.Combobox(root, values=["adam", "lbfgs", "sgd"], state='readonly', textvariable=solv)
combSolver.place(x=200, y=250)
mensSolver = Label(root, text="¿Graficos? ", background="orange")
mensSolver.place(x=370, y=220)
combSolver = ttk.Combobox(root, values=["Si", "No"], state='readonly', textvariable=graphics)
combSolver.place(x=370, y=250)

#boton que llama a la funcion entrenar_probar
botonEntrenar =Button(root,text="Entrenar y Probar", command=entrenar_probar)
botonEntrenar.place(x=30, y=300)

#labels y cajas de texto que dan informacion sobre la precision de diagnostico de la red neuronal
mporceBeni = Label(root, text="Porcentaje de precisión para diagnosticar un tumor benigno (%): ", background="orange")
mporceBeni.place(x=30, y=360)
entrympBeni = Entry(root, textvariable=preciB, width=10)
entrympBeni.place(x=400, y=360)
mporceMali = Label(root, text="Porcentaje de precisión para diagnosticar un tumor maligno (%): ", background="yellow")
mporceMali.place(x=30, y=390)
entrympMali = Entry(root, textvariable=preciM, width=10)
entrympMali.place(x=400, y=390)

#labels y cajas de texto que dan informacion sobre la cantidad de tumores benignos diagnosticados correcta e incorrectamente
mcBenic = Label(root, text="Cantidad de tumores benignos diagnosticados correctamente: ", background="orange")
mcBenic.place(x=30, y=430)
entryBenic = Entry(root, textvariable=cantBc, width=10)
entryBenic.place(x=400, y=430)
mcBenii = Label(root, text="Cantidad de tumores benignos diagnosticados como malignos: ", background="yellow")
mcBenii.place(x=30, y=460)
entryBenii = Entry(root, textvariable=cantBi, width=10)
entryBenii.place(x=400, y=460)

#labels y cajas de texto que dan informacion sobre la cantidad de tumores malignos diagnosticados correcta e incorrectamente
mcMalic = Label(root, text="Cantidad de tumores malignos diagnosticados correctamente: ", background="orange")
mcMalic.place(x=30, y=500)
entryMalic = Entry(root, textvariable=cantMc, width=10)
entryMalic.place(x=400, y=500)
mcMalii = Label(root, text="Cantidad de tumores malignos diagnosticados como benignos: ", background="yellow")
mcMalii.place(x=30, y=530)
entryMalii = Entry(root, textvariable=cantMi, width=10)
entryMalii.place(x=400, y=530)

#loop de Tkinter
root.mainloop()
