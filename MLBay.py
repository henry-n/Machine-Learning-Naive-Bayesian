#import library for Naive Bayes models
from sklearn.naive_bayes import GaussianNB

#numpy to create graph models
import numpy as np

#pip install scipy and import arff to open arff Files
#reference documentation on using arff from scipy
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.MetaData.html
from scipy.io import arff
import pandas as pd

#creates and loads models
import pickle

#for converting categorical features into numerics
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

#confusion matrix library
from sklearn.metrics import confusion_matrix

def User_Input(GNB_Model):
	
	list = []
	while True:
		print("(1). Enter Case")
		print("(2). Quit")
		
		choice = input("Choose An Option: ")
		
		if(choice == '1'):
			outlook = input("Outlook - (0).Overcast (1).Rainy (2).Sunny : ")
			outlook = int(outlook)
			
			if(outlook < 3 and outlook > -1 ):
				list.append(outlook)
				
				temp = input("Temperature - (0).Cool (1).Hot (2).Mild : ")
				temp = int(temp)
				
				if(temp < 3 and temp > -1):
					list.append(temp)
					
					humidity = input("Humidity - (0).High (1).Normal :")
					humidity = int(humidity)
					
					if(humidity < 2 and humidity > -1):
						list.append(humidity)
					
						windy = input("Windy - (0).False (1).True : ")
						windy = int(windy)
					
						if(windy < 2 and windy > -1):
							list.append(windy)
							
							x = GNB_Model.predict([np.asarray(list)])
							
							if(x[0] == 1):
								print("\nPlay: Yes\n")
							else:
								print("\nPlay: No\n")
							
						else:
							print("Wrong Input\n")
					
					else:
						print("Wrong Input\n")
							
				else:
					print("Wrong Input\n")
				
			else:
				print("Wrong Input\n")
		
		elif(choice == '2'):
			
			return False
			
	

def Split_Data():

	try:
		load_file = input("Entire The Name Of The Testing ARFF File Without Its Extension (.arff):")
		load_file = str(load_file) + ".arff"
	
		data, meta = arff.loadarff(load_file)

		#prints attributes names in arff file
		print(meta.names())
				
		types = meta.types()
		print(types,'\n')
				
		df = pd.DataFrame(data)
				
		label_en = preprocessing.LabelEncoder()
				
				
				
		X = df.apply(label_en.fit_transform)
		print(df,'\n')
		print(X,'\n')

		newSet = np.asarray(X)
				
		list_1 = []
		list_2 = []
		list_temp1=[]
		list_temp2=[]
				
		for i in newSet:
			#input to list everything except last element
			for k in i[:-1]:
				list_1.append(k)
					
			#input to list last element
			for j in i[-1:]:
				list_2.append(j)
					
			#first row/set finished. transfer and erase list for next row
			transfer = np.asarray(list_1)
			list_temp1.append(transfer)
			del list_1[:]
						
		features_train = np.asarray(list_temp1)
		labels_train = np.asarray(list_2)
		
		return(features_train,labels_train)
	
	except ValueError:
		print("FAILED TO OPEN FILE")
	

def Learn_NBC():
	try:
	
		load_file = input("Entire The Name Of The ARFF File Without Its Extension (.arff):")
		load_file = str(load_file) + ".arff"
	
		data, meta = arff.loadarff(load_file)
				
		print ('SUCCESS\n')
				
		#prints attributes names in arff file
		print(meta.names())
				
		types = meta.types()
		print(types,'\n')
				
		df = pd.DataFrame(data)
				
		label_en = preprocessing.LabelEncoder()
				
				
				
		X = df.apply(label_en.fit_transform)
		print(df,'\n')
		print(X,'\n')

		newSet = np.asarray(X)
				
		list_1 = []
		list_2 = []
		list_temp1=[]
		list_temp2=[]
				
		for i in newSet:
			#input to list everything except last element
			for k in i[:-1]:
				list_1.append(k)
					
			#input to list last element
			for j in i[-1:]:
				list_2.append(j)
					
			#first row/set finished. transfer and erase list for next row
			transfer = np.asarray(list_1)
			list_temp1.append(transfer)
			del list_1[:]
						
		features_train = np.asarray(list_temp1)
		labels_train = np.asarray(list_2)
				
		gnb = GaussianNB()
		gnb.fit(features_train,labels_train)
				
		filename = load_file.replace(".arff",".bin")
		pickle.dump(gnb, open(filename, 'wb'))
	
	
	except ValueError:
		print("FAILED TO OPEN FILE")



def menu():
	
	print("(1). Load Data Weka/ARFF Format")
	print("(2). Load A Model")
	print("(3). Enter A New Case")
	print("(4). EXIT")
	choice = input("Choose An Option: ")
	
	if(choice == '1'):
		
		Learn_NBC()
			
		return True
		
	elif(choice == '2'):
		try:
		
			model_name = input("Entire The Name Of The Model File Without Its Extension (.bin): ")
			model_name = str(model_name) + ".bin"
			
			loaded_model = pickle.load(open(model_name, 'rb'))
			
			print("SUCCESS\n")
			
			[features,labels] = Split_Data()
			
			predict_list = []
			
			for i in features:
				#predicted value using model
				p = np.asarray(i)
				predict = loaded_model.predict([p])
				#predict is an array so have to extract the target value out of the array
				predict_transfer = predict[0]
				predict_list.append(predict_transfer)
	
			
			#confusion_matrix(true_array,predicted_array)
			print('Confusion Matrix\n',confusion_matrix(labels, np.asarray(predict_list) ),'\n')
			
		except ValueError:
			print("FAILED TO LOAD MODEL")
			return True
		
	
	elif(choice == '3'):
		
		try:
			model_name = input("Entire The Name Of The Model File Without Its Extension (.bin): ")
			model_name = str(model_name) + ".bin"
			GNB_Model = pickle.load(open(model_name, 'rb'))
		
		except ValueError:
			print("FAILED TO LOAD MODEL")
		else:
			User_Input(GNB_Model)

		
	elif(choice == '4'):
		return False
	
	else:
		return True

def nb():

	while True:
	
		if(menu() is False):
			print ('\nEXITING')
			break
	
	
if __name__ == '__main__': nb() #nb as driver function