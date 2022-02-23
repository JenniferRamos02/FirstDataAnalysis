from cgitb import reset
from http import server
from operator import index
from os import access
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics

def run():
    #dir_ClienType = 'Data/Client_type.csv'
    dir_ClientCourses = 'Data/Client_courses.csv'
    #df_client=pd.read_csv(dir_ClienType)
    #df_client=df_client.set_index("CLIENT_ID")
    df_courses=pd.read_csv(dir_ClientCourses)
    #df_courses=df_courses.set_index("CLIENT_ID")
    #print(df_client)
    #print(df_courses)

    """"""""""""""""""""""""""""""""""""""""""""""Parte 1 """""""""""""""""""""""""""""""""""""""""""""
    """Punto 1"""
    #df_courses["COST"]=df_courses["FIRST_COURSE"]+df_courses["SECOND_COURSE"]+df_courses["THIRD_COURSE"] 
    #print(df_courses)
    """mean = np.mean(df_courses["COST"])
    print(mean)
    dev = np.std(df_courses["COST"])
    print(dev)
    x_axis = np.sort(df_courses["COST"])
    print(x_axis) 
    normal = norm.pdf(x_axis, mean, dev)
    plt.plot(x_axis, normal)
    plt.show()"""

    """Punto 2"""
    """axe_x = df_courses.columns[1:4]
    #for i in df_courses.items(1):
    y= [np.mean(df_courses["FIRST_COURSE"]), np.mean(df_courses["SECOND_COURSE"]),np.mean(df_courses["THIRD_COURSE"])]
    print(axe_x)
    print(y)
    vmin = np.amin(df_courses["COST"])
    vmax = np.amax(df_courses["COST"])
    print(vmin) 
    print(vmax) 
    axe_y = np.arange(0, round(vmax), 10) 
    print(axe_y)
    plt.bar(axe_x, y)
    plt.show()"""

    """Punto 3"""
    starters = {"Soup":3.0,"Tomato-Mozarella":15.0, "Oysters":20.0 }
    mains = {"Salad":9.0,"Spaghetti":20.0,"Steak":25.0,"Lobster":40.0}
    desserts = {"Ice cream":15.0, "Pie":10.0}
    starters = pd.Series(starters).sort_values(ascending=True)
    mains = pd.Series(mains).sort_values(ascending=True)
    desserts = pd.Series(desserts).sort_values(ascending=True)
    print(mains)
    #print(df_courses.loc[0,'FIRST_COURSE'])
    #print(starters[0])
    #df_courses.query('FIRST_COURSE >= starters[0]')
    df_courses["DISH1"]= pd.Series()
    #print (df_courses['FIRST_COURSE'] >= starters[0])
    for i in df_courses.index :
        if (df_courses['FIRST_COURSE'][i] >= starters[0]) & (df_courses['FIRST_COURSE'][i] < starters[1]):
            df_courses['DISH1'][i] = starters[0]
        elif (df_courses['FIRST_COURSE'][i] >= starters[1]) & (df_courses['FIRST_COURSE'][i] < starters[2]):
            df_courses['DISH1'][i] = starters[1]
        elif (df_courses['FIRST_COURSE'][i] >= starters[2]):
            df_courses['DISH1'][i] = starters[2]
        else:
            df_courses['DISH1'][i] = 0.0

    df_courses["DISH2"]= pd.Series()
    #print (df_courses['FIRST_COURSE'] >= mains[0])
    for i in df_courses.index :
        if (df_courses['SECOND_COURSE'][i] >= mains[0]) & (df_courses['SECOND_COURSE'][i] < mains[1]):
            df_courses['DISH2'][i] = mains[0]
        elif (df_courses['SECOND_COURSE'][i] >= mains[1]) & (df_courses['SECOND_COURSE'][i] < mains[2]):
            df_courses['DISH2'][i] = mains[1]
        elif (df_courses['SECOND_COURSE'][i] >= mains[2]) & (df_courses['SECOND_COURSE'][i] < mains[3]):
            df_courses['DISH2'][i] = mains[2]
        elif (df_courses['SECOND_COURSE'][i] >= mains[3]):
            df_courses['DISH2'][i] = mains[3]
        else:
            df_courses['DISH2'][i] = 0.0
           
    df_courses["DISH3"]= pd.Series()
    #print (df_courses['THIRD_COURSE'] >= desserts[0])
    for i in df_courses.index :
        if (df_courses['THIRD_COURSE'][i] >= desserts[0]) & (df_courses['THIRD_COURSE'][i] < desserts[1]):
            df_courses['DISH3'][i] = desserts[0]
        elif (df_courses['THIRD_COURSE'][i] >= desserts[1]):
            df_courses['DISH3'][i] = desserts[1]
        else:
            df_courses['DISH3'][i] = 0.0
    df_courses.to_csv('Data/Actual_dishes.csv')      
    #print(df_courses)
if __name__ == '__main__':
    run()