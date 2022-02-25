import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans

dir_ClientCourses = 'Data/Client_courses.csv'
df_courses=pd.read_csv(dir_ClientCourses)

def run():
    #dir_ClienType = 'Data/Client_type.csv'
    
    #df_client=pd.read_csv(dir_ClienType)
    #df_client=df_client.set_index("CLIENT_ID")
    
    #df_courses=df_courses.set_index("CLIENT_ID")
    #print(df_client)
    #print(df_courses)

    """"""""""""""""""""""""""""""""""""""""""""""Part 1 """""""""""""""""""""""""""""""""""""""""""""
    """Point 1"""
    
    cost=df_courses["FIRST_COURSE"]+df_courses["SECOND_COURSE"]+df_courses["THIRD_COURSE"]
    mean = np.mean(cost)
    #print(mean)
    dev = np.std(cost)
    #print(dev)
    x_axis = np.sort(cost)
    #print(x_axis) 
    normal = norm.pdf(x_axis, mean, dev)
    fig1=plt.figure()
    plt.plot(x_axis, normal) 
    plt.title("Cost Distribution")
    plt.xlabel("Total Cost")
    plt.ylabel("Normalized Cost")
    plt.savefig("Data/CostDistribution.png")
    plt.close(fig1)

    """Point 2"""
    
    axe_x = df_courses.columns[2:5]
    y= [np.mean(df_courses["FIRST_COURSE"]), np.mean(df_courses["SECOND_COURSE"]),np.mean(df_courses["THIRD_COURSE"])]
    #print(axe_x)
    #print(y)
    fig2=plt.figure()
    plt.bar(axe_x, y)
    plt.title("Cost per Course")
    plt.xlabel("Courses")
    plt.ylabel("Cost")
    plt.savefig("Data/CostperCourse.png")
    plt.close(fig2)

    """Point 3"""

    starters = {"Soup":3.0,"Tomato-Mozarella":15.0, "Oysters":20.0 }
    mains = {"Salad":9.0,"Spaghetti":20.0,"Steak":25.0,"Lobster":40.0}
    desserts = {"Ice cream":15.0, "Pie":10.0}
    starters = pd.Series(starters).sort_values(ascending=True)
    mains = pd.Series(mains).sort_values(ascending=True)
    desserts = pd.Series(desserts).sort_values(ascending=True)
    #print(mains)
    
    dish1 = []
    for i in df_courses.index :
        if (df_courses['FIRST_COURSE'][i] >= starters[0]) & (df_courses['FIRST_COURSE'][i] < starters[1]):
            dish1.append(starters[0])
        elif (df_courses['FIRST_COURSE'][i] >= starters[1]) & (df_courses['FIRST_COURSE'][i] < starters[2]):
            dish1.append(starters[1])
        elif (df_courses['FIRST_COURSE'][i] >= starters[2]):
            dish1.append(starters[2])
        else:
            dish1.append(0.0)
    #print(dish1)
    df_courses['DISH1']=np.array(dish1)
    

    dish2 = []
    for i in df_courses.index :
        if (df_courses['SECOND_COURSE'][i] >= mains[0]) & (df_courses['SECOND_COURSE'][i] < mains[1]):
            dish2.append(mains[0])
        elif (df_courses['SECOND_COURSE'][i] >= mains[1]) & (df_courses['SECOND_COURSE'][i] < mains[2]):
            dish2.append(mains[1])
        elif (df_courses['SECOND_COURSE'][i] >= mains[2]) & (df_courses['SECOND_COURSE'][i] < mains[3]):
            dish2.append(mains[2])
        elif (df_courses['SECOND_COURSE'][i] >= mains[3]):
            dish2.append(mains[3])
        else:
            dish2.append(0.0)
    df_courses['DISH2']=np.array(dish2)
    
    dish3 = []       
    for i in df_courses.index :
        if (df_courses['THIRD_COURSE'][i] >= desserts[0]) & (df_courses['THIRD_COURSE'][i] < desserts[1]):
            dish3.append(desserts[0])
        elif (df_courses['THIRD_COURSE'][i] >= desserts[1]):
            dish3.append(desserts[1])
        else:
            dish3.append(0.0)
    df_courses['DISH3']=np.array(dish3)
    
    df_courses["DRINK1"] = df_courses["FIRST_COURSE"]-df_courses["DISH1"]
    df_courses["DRINK2"] = df_courses["SECOND_COURSE"]-df_courses["DISH2"]
    df_courses["DRINK3"] = df_courses["THIRD_COURSE"]-df_courses["DISH3"]

    df_courses.to_csv('Data/Actual_dishes.csv')
    print(df_courses)

    """Part 2"""
    features = df_courses.loc[:,["FIRST_COURSE","SECOND_COURSE","THIRD_COURSE"]]
    #print(features)

    fig3=plt.figure(figsize=(10,4))

    plt.suptitle("K Means Clustering",fontsize=20)

    plt.subplot(1,5,3)
    plt.title("K = 4",fontsize=16)
    kmeans = KMeans(n_clusters=4)
    features["labels"] = kmeans.fit_predict(features)
    plt.scatter(features.FIRST_COURSE[features.labels == 0],features.SECOND_COURSE[features.labels == 0],features.THIRD_COURSE[features.labels == 0])
    plt.scatter(features.FIRST_COURSE[features.labels == 1],features.SECOND_COURSE[features.labels == 1],features.THIRD_COURSE[features.labels == 1])
    plt.scatter(features.FIRST_COURSE[features.labels == 2],features.SECOND_COURSE[features.labels == 2],features.THIRD_COURSE[features.labels == 2])
    plt.scatter(features.FIRST_COURSE[features.labels == 3],features.SECOND_COURSE[features.labels == 3],features.THIRD_COURSE[features.labels == 3])

    # I drop labels since we only want to use features.
    features.drop(["labels"],axis=1,inplace=True)

    plt.subplots_adjust(top=0.8)
    plt.savefig("Data/KMeansClustering.png")
    plt.close(fig3)

if __name__ == '__main__':
    run()