
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.basemap 
from mpl_toolkits.basemap import Basemap
import datetime
import matplotlib.colors
import pandas as pd
import scipy.stats as scp
from windrose import WindroseAxes
Waves=Dataset('DataEraInterim_Wavesdaily.nc','r')

print Waves.variables
print Waves.variables.keys()



altura= np.array(Waves.variables['swh'][:])
direccion= np.array(Waves.variables['mwd'][:])
periodo= np.array(Waves.variables['mwp'][:])
lat= np.array(Waves.variables['latitude'][:])
lon= np.array(Waves.variables['longitude'][:])
time= np.array(Waves.variables['time'][:]).astype(np.float)


altura[altura==-32767]=np.nan
direccion[direccion==-32767]=np.nan
periodo[periodo==-32767]=np.nan




fecha = np.array([datetime.datetime(1900,01,01)+\
datetime.timedelta(hours = time[i]) for i in range(len(time))])

#se recorta lat y lon para el punto 1

lat1=np.where(lat==12.125)[0][0]
lon1=np.where(lon==278.75)[0][0]

#se recorta lat y lon para el punto 2

lat2=np.where(lat==12.125)[0][0]
lon2=np.where(lon==278.625)[0][0]

#se recorta lat y lon para el punto 3

lat3=np.where(lat==12.125)[0][0]
lon3=np.where(lon==278.5)[0][0]

#se recorta lat y lon para el punto 4

lat4=np.where(lat==12.125)[0][0]
lon4=np.where(lon==278.375)[0][0]

#se recorta lat y lon para el punto 5

lat5=np.where(lat==12.125)[0][0]
lon5=np.where(lon==278.25)[0][0]

#se recorta lat y lon para el punto 6

lat6=np.where(lat==12.125)[0][0]
lon6=np.where(lon==278.125)[0][0]


#se recorta lat y lon para el punto 7

lat7=np.where(lat==12.125)[0][0]
lon7=np.where(lon==278.0)[0][0]


#se recorta lat y lon para el punto 8

lat8=np.where(lat==12.125)[0][0]
lon8=np.where(lon==277.875)[0][0]


#se recorta lat y lon para el punto 9

lat9=np.where(lat==13)[0][0]
lon9=np.where(lon==278.75)[0][0]

#se recorta lat y lon para el punto 10

lat10=np.where(lat==13)[0][0]
lon10=np.where(lon==278.625)[0][0]

#se recorta lat y lon para el punto 11

lat11=np.where(lat==13)[0][0]
lon11=np.where(lon==278.5)[0][0]

#se recorta lat y lon para el punto 12

lat12=np.where(lat==13)[0][0]
lon12=np.where(lon==278.375)[0][0]

#se recorta lat y lon para el punto 13

lat13=np.where(lat==13)[0][0]
lon13=np.where(lon==278.25)[0][0]

#se recorta lat y lon para el punto 14

lat14=np.where(lat==13)[0][0]
lon14=np.where(lon==278.125)[0][0]


#se recorta lat y lon para el punto 15

lat15=np.where(lat==13)[0][0]
lon15=np.where(lon==278.0)[0][0]


#se recorta lat y lon para el punto 16

lat16=np.where(lat==13)[0][0]
lon16=np.where(lon==277.875)[0][0]

#se recorta lat y lon para el punto 17

lat17=np.where(lat==12.25)[0][0]
lon17=np.where(lon==277.875)[0][0]

#se recorta lat y lon para el punto 18

lat18=np.where(lat==12.375)[0][0]
lon18=np.where(lon==277.875)[0][0]

#se recorta lat y lon para el punto 19

lat19=np.where(lat==12.5)[0][0]
lon19=np.where(lon==277.875)[0][0]

#se recorta lat y lon para el punto 20

lat20=np.where(lat==12.625)[0][0]
lon20=np.where(lon==277.875)[0][0]

#se recorta lat y lon para el punto 21

lat21=np.where(lat==12.75)[0][0]
lon21=np.where(lon==277.875)[0][0]

#se recorta lat y lon para el punto 22

lat22=np.where(lat==13)[0][0]
lon22=np.where(lon==277.875)[0][0]

#se recorta lat y lon para el punto 23

lat23=np.where(lat==12.25)[0][0]
lon23=np.where(lon==278.75)[0][0]

#se recorta lat y lon para el punto 24

lat24=np.where(lat==12.375)[0][0]
lon24=np.where(lon==278.75)[0][0]

#se recorta lat y lon para el punto 25

lat25=np.where(lat==12.5)[0][0]
lon25=np.where(lon==278.75)[0][0]

#se recorta lat y lon para el punto 26

lat26=np.where(lat==12.625)[0][0]
lon26=np.where(lon==278.75)[0][0]

#se recorta lat y lon para el punto 27

lat27=np.where(lat==12.75)[0][0]
lon27=np.where(lon==278.75)[0][0]

#se recorta lat y lon para el punto 28

lat28=np.where(lat==12.875)[0][0]
lon28=np.where(lon==278.75)[0][0]



#se recort la info (time, lat , lon)

alt1=altura[:,lat1,lon1]
dir1=direccion[:,lat1,lon1]
per1=periodo[:,lat1,lon1]


mis_lats= np.arange (12.125,13.125,0.125)
mis_lons=np.arange(277.875,278.875,0.125)

#####creo directorios para cada variable en cada uno de los puntos de la grilla San Andrés######

Swh={}
Dir={}
Per={}

x=0
for i in mis_lats:
	for j in mis_lons:
		x+=1
		if x <10:
			Swh['swh'+'0'+str(x)+'('+str(i)+'/'+str(j)+')'.format(x)]=altura[:,np.where(lat==i)[0][0],np.where(lon==j)[0][0]]
			Dir['dir'+'0'+str(x)+'('+str(i)+'/'+str(j)+')'.format(x)]=direccion[:,np.where(lat==i)[0][0],np.where(lon==j)[0][0]]
			Per['per'+'0'+str(x)+'('+str(i)+'/'+str(j)+')'.format(x)]=periodo[:,np.where(lat==i)[0][0],np.where(lon==j)[0][0]]
		else:
			Swh['swh'+str(x)+'('+str(i)+'/'+str(j)+')'.format(x)]=altura[:,np.where(lat==i)[0][0],np.where(lon==j)[0][0]]
			Dir['dir'+str(x)+'('+str(i)+'/'+str(j)+')'.format(x)]=direccion[:,np.where(lat==i)[0][0],np.where(lon==j)[0][0]]
			Per['per'+str(x)+'('+str(i)+'/'+str(j)+')'.format(x)]=periodo[:,np.where(lat==i)[0][0],np.where(lon==j)[0][0]]

Swh_s=sorted(Swh.items())
Dir_s=sorted(Dir.items())
Per_s=sorted(Per.items())


DataSwh=pd.DataFrame(Swh_s)
DataDir=pd.DataFrame(Dir_s)
DataPer=pd.DataFrame(Per_s)

columns=['SWH', 'DIRECCION', 'PERIODO']
Parametros= np.c_[DataSwh[1][37], DataDir[1][37],DataPer[1][37]]

Waves = pd.DataFrame(Parametros, index=fecha, columns=columns)

###  Per['per63(12.125/278.75)'][fechaRun1:fechaRun2+1]  >>>> para llamar un punto específico
12.125/278.75


#ciclo anual punto 1 SAI
altura1= alt1[np.isfinite(alt1)]

CicloAnual_altura1= np.zeros([12]) * np.NaN

Meses = np.array([fecha[i].month for i in range(len(fecha))])
for k in range(1,13):
    tmpp = np.where(Meses == k)[0]
   
    altura1_tmp= altura1[tmpp]
    CicloAnual_altura1[k-1]= np.mean(altura1_tmp)

Fig= plt.figure()
plt.rcParams.update({'font.size':14})
plt.plot(CicloAnual_altura1,'-', color='skyblue',lw=3,label='Hs')
x_label = ['Año']
plt.title('Ciclo Anual Altura de Ola Significante', fontsize=24)
plt.xlabel('Mes',fontsize=18)
plt.ylabel('Hs(metros)',fontsize=18)
plt.legend(loc=0)

axes = plt.gca()
axes.set_xlim([0,11])
axes.set_ylim([0.9,2.0])
axes.set_xticks([0,1,2, 3, 4, 5, 6, 7,8, 9, 10 ,11]) #choose which x locations to have ticks
axes.set_xticklabels(['Ene','Feb','Mar','Abr','May','Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic' ]) 
plt.savefig('CicloAnualAltura1.png')

#ciclo anual con pandas


WavesM=Waves.SWH.resample('M').mean()
#WavesD=Waves[0].resample('D').mean()

WM=np.array(WavesM)
WM=np.reshape(WM,(-1,12))
WMM=np.mean(WM,axis=0)
WMS=np.std(WM, axis=0)
plt.plot(WMM)


####ARCHIVOS TPAR######

TPAR=np.c_[timeRun,alt1[fechaRun1:fechaRun2+1],per1[fechaRun1:fechaRun2+1],dir1[fechaRun1:fechaRun2+1]] 

###rango de fechas###

fechaRun1=np.where(fecha==[datetime.datetime(2004, 1, 1, 0, 0)])[0][0]
fechaRun2=np.where(fecha==[datetime.datetime(2004, 1, 31, 18, 0)])[0][0]

#20110101.0000    1.039     6.588     18.114   30.0 >>formato TPAR


fechas= np.array([fecha[i].strftime('%Y%m%d.%H%M') for i in range(len(fecha))])
timeRun=fechas[fechaRun1:fechaRun2+1]

direc_spread=[30.0]*len(timeRun)

CoordenadasSAI = open("CoordenadasSAI.txt", "w")
for i,k in enumerate(sorted(Dir)):
	print >>CoordenadasSAI,i+1, k[6:-1]
	TPAR = np.c_[timeRun,DataSwh[1][i][fechaRun1:fechaRun2+1],DataPer[1][i][fechaRun1:fechaRun2+1],DataDir[1][i][fechaRun1:fechaRun2+1],direc_spread]  #"np.c_ >>column stack" 
	for j in (range(len(timeRun))):
    		with open('TPAR'+str(i+1)+'.txt','a') as archivo:
			if j==0:
       				archivo.write('TPAR'+"\n")
				line = "%s" % ("    ".join(map(str,(TPAR[j]))))
        			archivo.write(line+"\n")	
			else:
				line = "%s" % ("    ".join(map(str,(TPAR[j]))))
        			archivo.write(line+"\n")
        	    
CoordenadasSAI.close()




#####windrose like a stacked histogram with normed (displayed in percent)

def new_axes(fig, rect):
    ax = WindroseAxes(fig, rect, axisbg='w')
    fig.add_axes(ax)
    return ax


def set_legend(ax):
    l = ax.legend(borderaxespad=-6.8)
    plt.setp(l.get_texts(), fontsize=8)


rect1 = [0.1, 0.1, 0.4, 0.4]
rect2 = [0.6, 0.1, 0.4, 0.4]
rect3 = [1.1, 0.1, 0.4, 0.4]
rect4 = [0.1, -0.45, 0.4, 0.4]
rect5 = [0.6, -0.45, 0.4, 0.4]
rect6 = [1.1, -0.45, 0.4, 0.4]
rect7 = [0.1, -1.05, 0.4, 0.4]
rect8 = [0.6, -1.05, 0.4, 0.4]
rect9 = [1.1, -1.05, 0.4, 0.4]
rect10 = [0.1, -1.6, 0.4, 0.4]
rect11= [0.6, -1.6, 0.4, 0.4]
rect12 = [1.1, -1.6, 0.4, 0.4]


rect= [rect1,rect2,rect3,rect4,rect5,rect6,rect7,rect8,rect9,rect10,rect11,rect12]
namemes=('Enero','Febrero','Marzo','Abril','Mayo','Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre' )

fig = plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='w')

for j,i in enumerate(rect):
    ax1 = new_axes(fig, i)
    ax1.bar(Waves[Waves.index.month==j+1].DIRECCION, Waves[Waves.index.month==j+1].SWH, normed=True, opening=0.8, edgecolor='white')
    set_legend(ax1)
    ax1.set_title(namemes[j],  fontsize=20)

















