#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing modules
from flask import Flask, render_template,request
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
print("Modules imported")


# In[2]:


#Flask object
app=Flask("house_price_model")

#ML model
model=pickle.load(open('bengaluru_house_price_model.pkl', 'rb'))


# In[3]:


data=pd.read_csv(r'Bengaluru_House_Data.csv')
data.head()


# In[4]:


data['location'].fillna('unknown',inplace=True)
data['society'].fillna('unknown',inplace=True)

locations=list(data['location'].str.lower())
societies=list(data['society'].str.lower())
societies[:5]


# In[5]:


from sklearn.preprocessing import LabelEncoder
label_enco=LabelEncoder()

loc_labels=label_enco.fit_transform(locations)
soc_labels=label_enco.fit_transform(societies)
soc_labels[:5]


# In[6]:


x=locations.index('unknown')
print(loc_labels[x])
y=societies.index('unknown')
print(loc_labels[y])


# In[7]:


#HTTP GET request method
@app.route('/',methods=['GET'])

#Home function returns index.html
def Home():
    return render_template('index.html')


#HTTP POST request method
@app.route("/predict",methods=['POST'])

#defining predict function to calculate results from ML model based on inputs from the html form
def predict():
    
    #initiating area_type variables
    area_type_carpet_area=0
    area_type_plot_area=0
    area_type_super_built_up_area=0
    
    if request.method=='POST':
        
        #gathering values for features of our ML model
        total_sqft= float(request.form['total_sqft'])
        BHK=int(request.form['BHK'])
        loc=str(request.form['location']).lower()
        if(loc in locations):
            tmp1=locations.index(loc)
            location=loc_labels[tmp1]
        else:
            location=1199
        
        soc=str(request.form['society']).lower()
        if(soc in societies):
            tmp2=societies.index(soc)
            society=soc_labels[tmp2]
        else:
            society=1202
        
        bath=int(request.form['bath'])
        balcony=int(request.form['balcony'])
        area_type=request.form['area_type']
        
        #Area_type is categorised into carpet area, plot area,
        #built up area and super built up area with one hot encoding
        if (area_type=='carpet_area'):
            area_type_carpet_area=1
            area_type_plot_area=0
            area_type_super_built_up_area=0
            
        elif(area_type=='plot_area'):
            area_type_carpet_area=0
            area_type_plot_area=1
            area_type_super_built_up_area=0
            
        elif(area_type=='Super built_up_area'):
            area_type_carpet_area=0
            area_type_plot_area=0
            area_type_super_built_up_area=1
        
        else: 
            area_type_carpet_area=0
            area_type_plot_area=0
            area_type_super_built_up_area=0
        
        
        prediction=model.predict([[location,BHK,society,total_sqft,bath,balcony,area_type_carpet_area,area_type_plot_area,area_type_super_built_up_area]])
        output=round(prediction[0],3)
        
        #condition for invalid values
        if output<0:
            return render_template('index.html',prediction_text="Sorry, this house seems to be haunted")
        
        #condition for valid values
        else:
            return render_template('index.html',prediction_text="The price of this house should be {} lakhs".format(output))
        
    #page to be displayed on-screen with no values inserted
    else:
        return render_template('index.html',prediction_text="Go on, see if what you have in mind is affordable")
        

if __name__=="__main__":
    #run method starts web service
    #Debug : If anything is saved in the structure, server should start again
    app.run(debug=True)
        
