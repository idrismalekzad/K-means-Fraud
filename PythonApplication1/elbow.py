import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris
import pyodbc
from sklearn.cluster import KMeans


class MyClass(object):

   def __init__(self):
       # Load the iris dataset when the class is initialized
       self.X = self.get_sql_data()
   def get_data(self):
       # Return the features and labels
       return self.X, self.y

   def get_sql_data(self):
       # Define the connection string to your SQL Server
       conn = pyodbc.connect(
           'DRIVER={SQL Server};'
           'SERVER=10.81.15.16;'
           'DATABASE=FRAUD_DEV_PSP;'
           'UID=hasin;'
           'PWD=hasin'
       )
       
       query = "SELECT totalcount, sumamount, SwitchTerminal FROM SEP_INTERNET_AGGREGATED_DAILY"
       
       # Retrieve the data into a pandas DataFrame
       df = pd.read_sql(query, conn)
       
       # Close the connection
       conn.close()
       
       # Return the totalcount, totalamount, and SwitchNumber columns as a NumPy array
       return df[['totalcount', 'sumamount', 'SwitchTerminal']].values

   #Elbow Method 
   #Finding the ideal number of groups to divide the data into is a basic stage in any unsupervised algorithm. One of the most common techniques for figuring out this ideal value of k is the elbow approach.
   # def elbow_method(self):
   #     #Find optimum number of cluster
   #     sse = [] #SUM OF SQUARED ERROR
   #     for k in range(1,11):
   #         km = KMeans(n_clusters=k, random_state=2)
   #         km.fit(self.X)
   #         sse.append(km.inertia_)


   #     sns.set_style("whitegrid")
   #     g=sns.lineplot(x=range(1,11), y=sse)
   
   #     g.set(xlabel ="Number of cluster (k)", 
   #           ylabel = "Sum Squared Error", 
   #           title ='Elbow Method')
   
   #     plt.show()



   def elbow_method(self):

       if self.X is None or len(self.X) == 0:
            print("Data not loaded or empty")
            return

       wcss = []
       K = range(1, 70)  # Try different values of k (1 to 10)
       
       for k in K:
           kmeanss = KMeans(n_clusters=k, random_state=23)
           kmeanss.fit(self.X)
           wcss.append(kmeanss.inertia_)  # Inertia is WCSS
       
       # Plot the elbow curve
       plt.plot(K, wcss, 'bx-')
       plt.xlabel('Number of clusters (k)')
       plt.ylabel('Within-cluster Sum of Squares (WCSS)')
       plt.title('Elbow Method For Optimal k')
       plt.grid(True)
       plt.show(block=True)  # Ensure the window stays open until manually closed

   def k_means(self):
       kmeans = KMeans(n_clusters = 3, random_state = 2)
       kmeans.fit(self.X)


  





