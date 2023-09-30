# %% [code]
# %% [markdown] {"id":"E2VNqi5t6H7L"}
# # **Exploratory Data Analysis (EDA)** 
# #  *Airline Passenger Satisfaction*
# 

# %% [markdown] {"id":"9G51v3na7AHb"}
# ## Business Problem
# 
# To perform exploratory data analysis in order to find what affects the experience of airline passengers

# %% [markdown] {"id":"3m6lXkgS7p_j"}
# ## **Importing libraries**

# %% [code] {"id":"eH9eXB_G7vlo"}
import pandas as pd #data processing
import numpy as np #linear algebra

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#libraries for data visualisations
import seaborn as sns 
import matplotlib.pyplot as plt



#%matplotlib inline

#plt.style.use('bmh')
pd.options.display.float_format = '{:.3f}'.format

# %% [markdown] {"id":"NmM6l9NU-Zz3"}
# ## **Reading data**

# %% [code] {"id":"YODG19pK-nCB","outputId":"c011f405-7c94-4df2-8e44-3576a6ad6f85"}
#from google.colab import drive
#drive.mount('/content/drive')

# %% [code] {"id":"-gYJGItz_m4e"}
df = pd.read_csv('/kaggle/input/airline-passenger-satisfaction/test.csv') #path to the dataset file




# %% [markdown] {"id":"2VPildw3BTFB"}
# ## **Data Information**

# %% [code] {"id":"UBoGtQB4_1eX","outputId":"c68cc24e-0144-4027-dec7-e90113afdff6"}
#printing first 5 rows of the dataset
df.head() 

# %% [code] {"id":"kzys9ZOOBFKK","outputId":"4476fd12-3a08-4284-c17f-b6e6d0ac1745"}
 #printing last five rows of the dataset
df.tail()

# %% [code] {"id":"3YkXS1gLBjsJ","outputId":"72fe1a86-fe2c-410b-a66b-8db67af37fe9"}
#Printing the shape of the Dataframe
print(df.shape)



# %% [markdown] {"id":"pznFKJycDCG8"}
# This dataset has 1,29,880 rows and 24 columns
# 
# 
# 
# 

# %% [code] {"id":"e0UJ2tPMDHZk","outputId":"8c25d09e-0d3b-4ded-8d5b-8abd4a5f3102"}
#Printing the list of columns in the Dataframe
print(df.columns)

# %% [code] {"id":"xd3LUXMDEDpP","outputId":"3a4d3b59-438b-4501-9d96-c99056726ac1"}
#Number of unique values of all columns
df.nunique()

# %% [code] {"id":"vzogd40lEPjt","outputId":"78d0dce3-29de-4bbb-e332-e7df7fb5ddcd"}
#Understanding the content of all unique values
print(df.apply(lambda col: col.unique()))

# %% [markdown] {"id":"mqBsy6mwE4qD"}
# # **Checking Null Values**

# %% [code] {"id":"NMuomQYn62Z3","outputId":"f6d81e07-8394-4bf7-e292-f35470437248"}
sns.heatmap ( df.isnull (),  yticklabels= False ,  cbar= True ,  cmap= 'plasma' )

# %% [code] {"id":"6mGxCkesE91f","outputId":"ba493996-b352-47d2-a094-5f318201acbe"}
#Finding the sum of Missing Values in the dataset
nof_missing=df.isnull().sum()
nof_missing


# %% [code] {"id":"EWDvR8LwFacm","outputId":"a5a671dd-0e97-49c7-a997-c83c09e22350"}
#creating a new column called total and storing the sorted list of missing values in descending order
total = nof_missing.sort_values(ascending=False)
#creating a new column called percent and storing a sorted list of the percentage of missing values
percent = (nof_missing/df.isnull().count()).sort_values(ascending=False) 
#created a dictonary called missing which has total and percent as keys
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing

# %% [markdown] {"id":"tX88TNiqMGpK"}
# It is not required to drop the missing values as it only consists of 0.003% of the dataset

# %% [markdown] {"id":"ZbjgnX1LPApE"}
# # Checking Duplicate values

# %% [code] {"id":"0fmChzwLPFfN","outputId":"4eac0800-b785-4bef-abce-31efa5a9f575"}
#Finding Duplicate rows in the dataframe
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: {}".format(duplicate_rows_df.shape))

# %% [markdown] {"id":"TVdcszTePnWY"}
# # Checking Statistics for different data types

# %% [code] {"id":"MKdvyT0IBGgx","outputId":"e1fa8912-8214-4614-e820-ef84ae7b5612"}
#printing the column names and their data types
df.info() 

# %% [code] {"id":"e9PtEMzZD7tm","outputId":"ab640310-7be7-4e61-8122-a5e31261ec35"}
#Checking statistics for columns of the int64 data type
df.describe(include=['int64'])

# %% [code] {"id":"yB9dPqUTQBet","outputId":"ef3fe1bc-a2f3-4d6a-b44c-4fffce430bf7"}
#Checking statistics for columns of the object data type
df.describe(include=['object'])

# %% [code] {"id":"966tdvdsp6kY","outputId":"b3756609-0bc2-4347-c1e9-8c6564920dc9"}
#Checking statistics for columns of the float64 data type
df.describe(include=['float64'])

# %% [markdown] {"id":"MQpgmprVqZ_u"}
# # Checking for outliers

# %% [code] {"id":"HVYrexD7tZaf","outputId":"236b82a4-c630-4d26-8247-8d10bb19bdca"}
#As detection of outliers can only be applied to numerical values a new dataframe called df_outlier has been created which only has numeric data type. 
df_outlier = df.select_dtypes(exclude='object').columns.to_list()
df_outlier

# %% [code] {"id":"VLjrCcSD19SN","outputId":"0b2825af-07ce-4292-da52-e744184b2b26"}
df_feature=df_outlier[0:5]
df_feature

# %% [markdown] {"id":"zZuMTCExxv-s"}
# A new list of first few attributes was created as the rest of the attributes are customer survey containing ratings from 0-5.

# %% [code] {"id":"ih-MHpoJtbDO","outputId":"0c4d5791-0942-4ff0-e12c-64e292ae5da8"}
#Box plot and Distribution plot
outl=df_feature
for x in outl:
    fig, axes=plt.subplots(1,2,figsize=(15,5))
    sns.histplot(x=df[x], ax=axes[0], kde=True, stat="density", linewidth=0)
    sns.boxplot(x=df[x], ax=axes[1], showmeans=True)
    fig.suptitle(x)

# %% [markdown] {"id":"IeKcLfJo9in2"}
# Observation: There are no outliers in ID and Age attribute but we can see some outliers in Flight Distance, Arrival Delay and Departure Delay.
# 
# 

# %% [code] {"id":"EnXHKAESWAWe","outputId":"9770c918-050b-49f7-8cdc-fb09894243d3"}

corr = df.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, annot=True, cmap = 'summer_r', linewidths=.5, cbar_kws={"shrink": .9})

# %% [markdown] {"id":"-R6stuByR7LS"}
# "Ease_of_Online_booking" is highly correlated with "Inflight_wifi_service". Also "Arrival Delay" and "Departure Delay" are highly correlated too.But no pair is having correlation coefficient exactly equal to 1. So there is no perfect multicollinearity. Hence we are not discarding any variable.

# %% [markdown] {"id":"_9Wp0HVYCKV6"}
# # **Data Visualisation**

# %% [markdown] {"id":"TTVw2RaoqGEm"}
# ## *Overview*

# %% [code] {"id":"6xI6-8Vx3diR","outputId":"4f27188a-5bcb-41c6-b8ec-b702574616b6"}
# Plotting a Pie Chart for satisfaction percentage
labels = ['Neutral or Dissatisfied','Satisfied']

sizes = [df['satisfaction'].value_counts()[0],
         df['satisfaction'].value_counts()[1]]
explode=[0.1,0]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, explode=explode)
ax1.axis('equal')
plt.show()

# %% [markdown] {"id":"7XFe3CF-eJDI"}
# Observation: 56.6% customers are neutral or dissatisfied with theservice provided by the airlines.

# %% [code] {"id":"UvACmgshm-vI","outputId":"a4d957ec-8c07-4bee-9999-6efe1df7c259"}

plt.figure(figsize = (10,5))
sns.kdeplot(data = df, x= "Age", hue = "satisfaction", common_norm = False, palette ="Paired")
plt.suptitle("Satisfaction results by age", fontsize = 14,)
sns.despine(top = True, right = True, left = False, bottom = False)

plt.show()

# %% [markdown] {"id":"ei_nJ8-3euKX"}
# Observation: Average customer age is appoximately 40 years and customers between 40 to 60 years are most likely to find the service satisfactory.

# %% [code] {"id":"YL3-K-yNr-xw","outputId":"3dd06f87-b57b-49bd-fb98-6db8a462c946"}

plt.figure(figsize = (8,5))
sns.countplot(x ="Gender", data = df, hue ="satisfaction", palette ="Paired" )

sns.despine(top = True, right = True, left = False, bottom = False)
plt.title("Satisfaction results by gender")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"0KaAfScbfhLs"}
# Observation: Both the genders have almost the even rate of dissatisfaction with the services.

# %% [markdown] {"id":"UKILK9moqVS7"}
# ## *Demography*

# %% [code] {"id":"0tV7uokxC55v","outputId":"d22137d3-0388-4be6-ea72-5cd0d5b6b182"}

plt.figure(figsize = (8,5))
sns.countplot(x ="Customer Type", data = df, hue ="satisfaction",palette ="Pastel1" )
try:
  plt.title("Satisfaction results by Customer Type")
  sns.despine(top = True, right = True, left = False, bottom = False)
  plt.legend(loc='best', bbox_to_anchor=(1, 0.5),  title = "satisfaction")
except ValueError as e:
  print("Wrong legend location",e)
  plt.show()

# %% [markdown] {"id":"4m5fBTa-jEmO"}
# Observation: Both the first time as well as the returning customers are not satisfied with the airline service.

# %% [code] {"id":"i4v50wbMOnnp","outputId":"fb292df7-2f28-41bb-82d7-c309d66c4cf1"}
sns.catplot( x ="Class", hue ="satisfaction", kind = "count", col = "Type of Travel", 
            data = df ,palette ="Pastel2", height = 6)
plt.suptitle("Satisfaction results by Class and Type of Travel", y = 1.05, fontsize= 14)
sns.despine(top = True, right = True, left = False, bottom = False)

plt.show()

# %% [markdown] {"id":"qBawaDk6kPBj"}
# Observation1: Passenger travelling for business purposes are more satisfied with the service as compared to people travelling for personal purposes.
# 

# %% [markdown] {"id":"tJ3Gl4zJt1Nt"}
# Observation2: Business class passengers travelling for business purposes are satisfied with the services whereas economy class passengers travelling for personal purposes are highly dissatisfied.

# %% [code] {"id":"WRo2lLOQW39c","outputId":"3d8b1429-a683-46c8-dab3-f1e0285af3b6"}
sns.countplot(x = 'Customer Type',  data = df, hue="Class",palette ="Paired")

# %% [markdown] {"id":"BO9rf0oeuAfJ"}
# Observation: Loyal/Returning passengers are mostly customers travelling through Business Class followed by Economy and Economy plus.

# %% [code] {"id":"h8IWWK4xWJ0Y","outputId":"9cfffaf9-2fa8-40c1-d14d-492e3adaaf91"}
#Converting arrival and departure delay from minutes to hours for better interpretation.
df["Departure_delay_hr"] = round(df["Departure Delay in Minutes"]/60,1)
df["Arrival_delay_hr"] = round(df["Arrival Delay in Minutes"]/60,1).fillna(0)
delay = df[["Departure_delay_hr","Arrival_delay_hr"]]
df['total_delay']= df['Arrival Delay in Minutes']+ df["Departure Delay in Minutes"]
sns.barplot(x = 'satisfaction',  data = df,y="total_delay", hue="Customer Type",palette ="Paired")
plt.suptitle("Satisfaction results by Flight Delay in Hours by Customer Type")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5),  title = "Customer Type")
sns.despine(top = True, right = True, left = False, bottom = False)

plt.show()

# %% [markdown] {"id":"UgctFAmPvySj"}
# Observation: Returning passengers are more dissatisfied with the delay in flights than first time passengers.

# %% [code] {"id":"yRhE2IcvGPoF","outputId":"cc494e0e-84f9-4944-ec29-f617a08b3c87"}
sns.barplot(x = 'satisfaction',  data = df,y="total_delay", hue="Class",palette ="Paired")
plt.suptitle("Satisfaction results by Flight Delay in Hours by class")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5),  title = "Class")
sns.despine(top = True, right = True, left = False, bottom = False)

plt.show()

# %% [markdown] {"id":"Ur2aP93Owyy7"}
# Observation: Passengers travelling from all classes are dissatisfied but Business Class people are more dissatisfied.

# %% [code] {"id":"UJtofHTlqcfH","outputId":"e88cdc9d-e10c-4976-bc60-a1938dc68f97"}
sns.displot(x = "Flight Distance", data = df, hue ="satisfaction", height = 8,palette ="Paired_r"  )
plt.title("Satisfaction results by Flight Distance")
plt.show()

# %% [markdown] {"id":"XLPG53lzxtLS"}
# Observation: Passengers flying in long distance flights are more satisfied whereas passengers travelling for short distance are highly dissatisfied.

# %% [markdown] {"id":"VIbCozM6XDME"}
# ## Services

# %% [markdown] {"id":"eLAxr0a2lx9e"}
# Ease of Online Booking

# %% [code] {"id":"T55Gey9_XJFI","outputId":"113eb872-e729-463d-d236-6a611ddde6d4"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Ease of Online booking", data = df, hue ="satisfaction",palette ="Paired" )

plt.title("Satisfaction results by Ease of Online Booking")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"SvtR_EBofSG0"}
# Observation: The overall satisfaction increases as the rating increases which means people who were overall satisfied with the booking service were likely to rate overall service satisfactory.

# %% [markdown] {"id":"lKaHgz2hl7DH"}
# Check-in Service

# %% [code] {"id":"NBR03xMm9icB","outputId":"24c9853d-4e11-4c19-95f4-e6fc70f8a871"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Checkin service", data = df, hue ="satisfaction",palette ="Paired" )

plt.title("Satisfaction results by Check-in Service")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"7WDBjhC5ik2J"}
# Observation: Check-in Service has overall less impact on overall satisfaction, as even on score 4 passengers have rated dissatisfaction

# %% [markdown] {"id":"B93mV-yvl87Z"}
# Online Boarding

# %% [code] {"id":"sCpd_7-jfQzs","outputId":"d1c2805a-2e19-411f-c957-b54853b7628e"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Online boarding", data = df, hue ="satisfaction",palette ="Paired" )

plt.title("Satisfaction results by Online Boarding")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"x5hnBbqFjZeB"}
# Observation: As the scores for online bookings increase, the overall dissatisfaction appears to decrease. 

# %% [markdown] {"id":"MQQQ2cwTmALA"}
# Seat Comfort

# %% [code] {"id":"YiiEDCbbjlGC","outputId":"0730e73f-6b22-4354-fbf7-c7f9a40e982f"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Seat comfort", data = df, hue ="satisfaction",palette ="Paired" )
plt.title("Satisfaction results by Seat Comfort")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"VnCQ6fmLlEz1"}
# Observation: As the rating increases the overall dissatisfaction also decreases, people who voted overall seats extremely comfortable also voted the overall services satisfactory.

# %% [markdown] {"id":"h1RJo2qPmGyv"}
# Leg Room Service

# %% [code] {"id":"TVZods2SmHum","outputId":"73bf6287-7688-4799-dfe7-5ae594f2764e"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Leg room service", data = df, hue ="satisfaction",palette ="Paired" )
plt.title("Satisfaction results by Leg Room Service")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"V0WZsHXwmbMS"}
# Observation: Follows a similar pattern as Seat Comfort

# %% [markdown] {"id":"3m8ifmRfmg7N"}
# Cleanliness

# %% [code] {"id":"LuASY8tLolqy","outputId":"e4b8ffe0-05ee-4463-f273-532c0dc870b7"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Cleanliness", data = df, hue ="satisfaction",palette ="Paired" )
plt.title("Satisfaction results by Cleanliness")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"jFJOv7eZnPNj"}
# Observation: While cleanliness is important it is not a critical factor defining the satisfaction of the people.

# %% [markdown] {"id":"ahxP680vqOBz"}
# Food and Drink

# %% [code] {"id":"SAvuuzmEqVEf","outputId":"bcd171b3-b11b-421a-edb0-9260f34551ad"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Food and drink", data = df, hue ="satisfaction",palette ="Paired" )
plt.title("Satisfaction results by Food and Drink")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"U_ObiaHmqsVu"}
# Observation: Increased ratings shows a decrease in the overall dissatisfaction of the service, but not too impactful.

# %% [markdown] {"id":"SboYPUiaq8-V"}
# In-flight Service

# %% [code] {"id":"oIbcvy1frAwa","outputId":"bfa58362-6bab-465c-ef3a-2558b92c853a"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Inflight service", data = df, hue ="satisfaction",palette ="Paired" )
plt.title("Satisfaction results by In-flight  Service")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction") 

plt.show()

# %% [markdown] {"id":"3gfrR_h-r2sx"}
# Observation: A similar trend can be seen as the seat comfort and leg room service, as the experience gets better the customers are more satisfied

# %% [markdown] {"id":"7hji1RB0sCBu"}
# In-flight Wifi Service

# %% [code] {"id":"uaRcxG2xsGIB","outputId":"9aa6e6a8-871f-436b-c775-70936d601f39"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Inflight wifi service", data = df, hue ="satisfaction",palette ="Paired" )
plt.title("Satisfaction results by In-flight Wifi Service")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"P4ITz30gs34c"}
# Observation: As the rating increases the dissatisfaction completely drps and a huge drop can be seen from 3/4.

# %% [markdown] {"id":"WvRXt9K6ta0a"}
# In-flight Entertainment

# %% [code] {"id":"owrOZADCtaGE","outputId":"3b4a62c9-d7e2-45a4-f5dc-f043263dc206"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Inflight entertainment", data = df, hue ="satisfaction",palette ="Paired" )
plt.title("Satisfaction results by In-flight Entertainment")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"t_QPt-Ybtnzg"}
# Observation: This service also follows a similar trend to seat comfort services

# %% [markdown] {"id":"rTySEijqtzNH"}
# Baggage Handling

# %% [code] {"id":"YCqL8PDrtjHH","outputId":"84dd5c98-4fba-4c3e-c138-d1f7789e07a4"}
plt.figure(figsize = (10,5))
sns.countplot(x ="Baggage handling", data = df, hue ="satisfaction",palette ="Paired" )
plt.title("Satisfaction results by Baggage Handling")
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),  title = "satisfaction")

plt.show()

# %% [markdown] {"id":"Cut7FKnRuYLb"}
# Observation: A similar trend can be seen, as the rating increases the dissatisfaction decreases.

# %% [markdown] {"id":"O1Mox1pKAb0e"}
# #Few Important services to look into as per our business problem.

# %% [markdown] {"id":"fxyl9a4FBQcp"}
# ##Inflight Wifi Service

# %% [code] {"id":"vVIl44HbSc5V","outputId":"eb18f578-3a19-488d-c183-d80b48905b67"}
sns.catplot( x ="Inflight wifi service", hue ="satisfaction", kind = "count", col = "Class", 
            data = df ,palette ="Pastel2", height = 6)
plt.suptitle("Satisfaction results by Class and Type of Travel", y = 1.05, fontsize= 14)
sns.despine(top = True, right = True, left = False, bottom = False)

plt.show()

# %% [code] {"id":"Zmf4H7_TILNY","outputId":"51382de3-782f-4656-f5ed-a39f15680a33"}
sns.catplot( x ="Online boarding", hue ="satisfaction", kind = "count", col = "Class", 
            data = df ,palette ="Pastel2", height = 6)
plt.suptitle("Satisfaction results by Class and Type of Travel", y = 1.05, fontsize= 14)
sns.despine(top = True, right = True, left = False, bottom = False)

plt.show()

# %% [code] {"id":"G_bE52IEJGF3","outputId":"e45a6392-4c0e-4793-a90f-fec388512825"}
sns.catplot( x ="Seat comfort", hue ="satisfaction", kind = "count", col = "Class", 
            data = df ,palette ="Pastel2", height = 6)
plt.suptitle("Satisfaction results by Class and Type of Travel", y = 1.05, fontsize= 14)
sns.despine(top = True, right = True, left = False, bottom = False)

plt.show()

# %% [code] {"id":"TPS5V24x4Jmg"}
#Few parameters were collected into a list called external factors and the total percent score was evaluated

external_facts=df[['Ease of Online booking','Gate location', 'Online boarding', 'Departure/Arrival time convenient','Checkin service','Baggage handling']]
df["Total_escore"] = external_facts.sum(axis = 1)
max_score = len(external_facts.columns)*6
df["Total_escore_percent"] = round((df["Total_escore"]/max_score)*100,1)

# %% [code] {"id":"Przc02_M6QGY","outputId":"8814ecda-142b-4d64-d1ff-385948265f1a"}


plt.figure(figsize = (10,5))
sns.barplot(x = 'satisfaction',  data = df,y="Total_escore_percent", hue="Class",palette ="Paired")
plt.title("Satisfaction results by Total score (%) of  Class", fontsize = 14)
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='best',bbox_to_anchor=(1, 0.5), title = "satisfaction")
plt.show()

# %% [markdown] {"id":"qy5w5OWl-OrL"}
# Observation: Overall, passengers are satisfied with the external factors with people flying Business class being 59% satisfied.

# %% [code] {"id":"9672wZU14gou"}
internal_facts=df[['On-board service',
 'Seat comfort',
 'Leg room service',
 'Cleanliness',
 'Food and drink',
 'Inflight service',
 'Inflight wifi service',
 'Inflight entertainment']]
df["Total_iscore"] = internal_facts.sum(axis = 1)
max_score = len(internal_facts.columns)*8
df["Total_iscore_percent"] = round((df["Total_iscore"]/max_score)*100,1)

# %% [code] {"id":"Z1rN0SDu5wPE","outputId":"dd2ba83d-64f2-4e14-ca36-3803cbbe2d77"}
plt.figure(figsize = (10,5))
sns.barplot(x = 'satisfaction',  data = df,y="Total_iscore_percent", hue="Class",palette ="Paired")
plt.title("Satisfaction results by Total score (%)", fontsize = 14)
sns.despine(top = True, right = True, left = False, bottom = False)
plt.legend(loc='best', title = "satisfaction")
plt.show()