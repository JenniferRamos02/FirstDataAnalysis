from matplotlib import projections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans

dir_ClientCourses = 'Data/Client_courses.csv'
df_courses=pd.read_csv(dir_ClientCourses)
dir_ClienType = 'Data/Client_type.csv'
df_client=pd.read_csv(dir_ClienType)
label = {
    0:"Business",
    1:"Retirement",
    2:"Onetime",
    3:"Healthy",
}

"""Creaamos diccionarios de los platos del restaurante"""
starters = {"Soup":3.0,"Tomato-Mozarella":15.0, "Oysters":20.0 }
mains = {"Salad":9.0,"Spaghetti":20.0,"Steak":25.0,"Lobster":40.0}
desserts = {"Ice cream":15.0, "Pie":10.0}
dish_type = [starters, mains, desserts]

"""Los convertimos en series"""
starters = pd.Series(starters).sort_values(ascending=True)
mains = pd.Series(mains).sort_values(ascending=True)
desserts = pd.Series(desserts).sort_values(ascending=True)
"""diccionario para darle nombre a cada plato"""
typeofdish = {"DISH1": "starter", "DISH2": "main", "DISH3": "dessert" } 

"""Los unimos en un solo diccionario"""
dic_dish = {**starters, **mains, **desserts}

def set_label(value):
    return label.get(value)

"""funcion para calcular la probabilidad de que un plato sea comprado por tipo de plato y tipo de cliente"""
def dish_prob(df, type_dish, dish):
    dish_taked = len(df[df[type_dish]==dic_dish[dish]].loc[:,[type_dish]]) #Aqui calculamos la cantidad de veces que este tipo de cliente tomo el plato
    print(dish_taked) 
    sample = len(df) #Aqui obtenemos el tamaño de la muestra osea el numero de un tipo de clientes en especifico
    print(sample)
    percentage = (dish_taked*100)/sample #Aqui calculamos el porcentaje
    print ("the probability that a",df.iloc[1,12], "client take a", dish, "in the ",typeofdish[type_dish]," is ", percentage,"%")
    #return percentage

"""funcion para graficar la distribucion de las bebidas segun el tipo de plato y el tipo de cliente"""
def drink_distribution(df):
    fig = plt.figure()
    plt.title("Drink ditribution " + df.iloc[1,12] + " client")
    #plt.title(df.iloc[1,12])
    plt.scatter(range(len(df.DRINK1)), df.DRINK1, c="r", label="Drink Starter")
    plt.scatter(range(len(df.DRINK1)), df.DRINK2, c="b", label="Drink Main")
    plt.scatter(range(len(df.DRINK1)), df.DRINK3, c="g", label="Drink Dessert")
    plt.savefig("Data/Drink ditribution " + df.iloc[1,12] + " client.png") #lo guardamos en una imagen
    plt.close(fig)

def name_dish(x, dishtype):
    dic = dish_type[dishtype]
    for key, value in dic.items():
        if value == x:
            return key 

def run():
      
    
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
    
    #print(mains)
    """calculamos el precio real del plato que fue comprado por cada compra 
    y lo guardamos en un arreaglos llamado dish1 2 o 3 segun la compra"""
    
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
    
    """Agregamos tres columnas al dataset cada una con el precio real  de cada plato comprado"""
    df_courses["DRINK1"] = df_courses["FIRST_COURSE"]-df_courses["DISH1"]
    df_courses["DRINK2"] = df_courses["SECOND_COURSE"]-df_courses["DISH2"]
    df_courses["DRINK3"] = df_courses["THIRD_COURSE"]-df_courses["DISH3"]
    
    """guardamos el dataset en un nuevo archivo"""
    df_courses.to_csv('Data/Actual_dishes.csv')
    print(df_courses)

    """Part 2"""
    """Apllicamos el algoritmo de kmean para hacer la clasificacion de los datos"""
    
    
    """Agrupamps nuestros clusters en un solo conjunto de datos"""
    features = df_courses.loc[:,["FIRST_COURSE","SECOND_COURSE","THIRD_COURSE"]]
    print(features)
    fig3=plt.figure(figsize=(10,4))
    ax1=fig3.add_subplot(projection='3d')
    plt.suptitle("K Means Clustering",fontsize=20)
    """usamos la funcion Kmeans de la libreria sklearn.cluster"""
    plt.title("K = 4",fontsize=16)
    kmeans = KMeans(n_clusters=4)
    features["labels"] = kmeans.fit_predict(features)
    print(features)
    """graficamos los resultados de la clasifficacion de datos"""
    ax1.scatter(features.FIRST_COURSE[features.labels == 0],features.SECOND_COURSE[features.labels == 0], features.THIRD_COURSE[features.labels == 0],c="r")
    ax1.scatter(features.FIRST_COURSE[features.labels == 1],features.SECOND_COURSE[features.labels == 1], features.THIRD_COURSE[features.labels == 1],c="g")
    ax1.scatter(features.FIRST_COURSE[features.labels == 2],features.SECOND_COURSE[features.labels == 2], features.THIRD_COURSE[features.labels == 2],c="b")
    ax1.scatter(features.FIRST_COURSE[features.labels == 3],features.SECOND_COURSE[features.labels == 3], features.THIRD_COURSE[features.labels == 3],c="y")
    

    plt.savefig("Data/KMeansClustering.png") #lo guardamos en una imagen
    plt.close(fig3)

    """colocamos las etiquetas obteniadas de la clasificacion de datos a nuestro dataset"""   
    df_courses["CLIENT_TYPE"]=features.labels.apply(set_label) 
    print(df_courses)
    """Guardamos nuestro dataset en un en Clustering.cvs"""
    df_courses.to_csv('Data/Clustering.csv')

    """Parte 3"""
    """Comparamos las etiquetas dadas en el archivo part3 con las obtenidas por nuestro clustering"""
    coparation = df_courses["CLIENT_TYPE"]==df_client["CLIENT_TYPE"] #en este vecotor se guarda la comparacion, coincide = True, no coincide = false
    print(coparation)
    n = coparation.value_counts() #obtenemos la cantidad que coicide 
    print("match percentage = ", (n.loc[True]*100)/len(df_courses.CLIENT_TYPE)) #calculamos el porcentaje con respecto a la muestra total
    Dis_typeClient = df_courses.CLIENT_TYPE.value_counts() #Aqui se cuenta la frecuencia con que cada tipo de cliente aparece en el dataset obtenido despues de clustering
    print("Customer typer distribution = ", Dis_typeClient)


    df = pd.read_csv('Data/Clustering.csv')
    """agrupamos por cada tipo cliente en un dataset"""
    healthy = df[df['CLIENT_TYPE']=="Healthy"]
    retirement = df[df['CLIENT_TYPE']=="Retirement"]
    business = df[df['CLIENT_TYPE']=="Business"]
    onetime = df[df['CLIENT_TYPE']=="Onetime"]
    print(healthy)
    #print(onetime[onetime["FIRST_COURSE"]!=0])
    """para cada tipo de cliente calculamos la probabilidad de que pida una entrada un plato principal o un postre"""
    """"Para esto calculamos para cada tipo de cliente la cantidad de veces que el plato fue comprado de la muestra total y se calcula el porcentaje con respecto a la misma"""


    print("the probability that a healthy client take a starter is ",(len(healthy[healthy["FIRST_COURSE"]!=0])*100)/len(healthy),"%")
    print("the probability that a healthy client take a main is ",(len(healthy[healthy["SECOND_COURSE"]!=0])*100)/len(healthy),"%")
    print("the probability that a healthy client take a dessert is ",(len(healthy[healthy["THIRD_COURSE"]!=0])*100)/len(healthy),"%")

    print("the probability that a business client take a starter is ",(len(business[business["FIRST_COURSE"]!=0])*100)/len(business),"%")
    print("the probability that a business client take a main is ",(len(business[business["SECOND_COURSE"]!=0])*100)/len(business),"%")
    print("the probability that a business client take a dessert is ",(len(business[business["THIRD_COURSE"]!=0])*100)/len(business),"%")

    print("the probability that a onetime client take a starter is ",(len(onetime[onetime["FIRST_COURSE"]!=0])*100)/len(onetime),"%")
    print("the probability that a onetime client take a main is ",(len(onetime[onetime["SECOND_COURSE"]!=0])*100)/len(onetime),"%")
    print("the probability that a onetime client take a dessert is ",(len(onetime[onetime["THIRD_COURSE"]!=0])*100)/len(onetime),"%")

    print("the probability that a retirement client take a starter is ",(len(retirement[retirement["FIRST_COURSE"]!=0])*100)/len(retirement),"%")
    print("the probability that a retirement client take a main is ",(len(retirement[retirement["SECOND_COURSE"]!=0])*100)/len(retirement),"%")
    print("the probability that a retirement client take a dessert is ",(len(retirement[retirement["THIRD_COURSE"]!=0])*100)/len(retirement),"%")

    """para calcular la probabilidad que un cliente tome un plato en especifico en la entrada, el plato principal o el postre
    repetimos el mismo procedimiento anterior solo que esta vez vamos a contar cuantas veces se repite un plato especifico en un mismo
    tipo de plato, ej cuantes veces se pidio una sopa en la muestra de las entradas del grupo de cliente saludable. A ese valor se le 
    calcula la probablidad con respecto a la muestra total del grupo de clientes saludables segun el ejemplo"""

    dish_prob(healthy, "DISH1", "Soup")
    dish_prob(healthy, "DISH1", "Tomato-Mozarella")
    dish_prob(healthy, "DISH1", "Oysters")
    dish_prob(healthy, "DISH2", "Salad")
    dish_prob(healthy, "DISH2", "Spaghetti")
    dish_prob(healthy, "DISH2", "Steak")
    dish_prob(healthy, "DISH2", "Lobster")
    dish_prob(healthy, "DISH3", "Ice cream")
    dish_prob(healthy, "DISH3", "Pie")

    
    dish_prob(retirement, "DISH1", "Soup")    
    dish_prob(retirement, "DISH1", "Tomato-Mozarella")
    dish_prob(retirement, "DISH1", "Oysters")
    dish_prob(retirement, "DISH2", "Salad")
    dish_prob(retirement, "DISH2", "Spaghetti")
    dish_prob(retirement, "DISH2", "Steak")
    dish_prob(retirement, "DISH2", "Lobster")
    dish_prob(retirement, "DISH3", "Ice cream")
    dish_prob(retirement, "DISH3", "Pie")

    dish_prob(business, "DISH1", "Soup")    
    dish_prob(business, "DISH1", "Tomato-Mozarella")
    dish_prob(business, "DISH1", "Oysters")
    dish_prob(business, "DISH2", "Salad")
    dish_prob(business, "DISH2", "Spaghetti")
    dish_prob(business, "DISH2", "Steak")
    dish_prob(business, "DISH2", "Lobster")
    dish_prob(business, "DISH3", "Ice cream")
    dish_prob(business, "DISH3", "Pie")


    dish_prob(onetime, "DISH1", "Soup")    
    dish_prob(onetime, "DISH1", "Tomato-Mozarella")
    dish_prob(onetime, "DISH1", "Oysters")
    dish_prob(onetime, "DISH2", "Salad")
    dish_prob(onetime, "DISH2", "Spaghetti")
    dish_prob(onetime, "DISH2", "Steak")
    dish_prob(onetime, "DISH2", "Lobster")
    dish_prob(onetime, "DISH3", "Ice cream")
    dish_prob(onetime, "DISH3", "Pie")

    #print(healthy.FIRST_COURSE)
    """graficas de distrubución por tipo de cliente y por plato"""
    
    fig = plt.figure()
    plt.title("Dish ditribution heelthy client")
    plt.scatter(range(len(healthy.FIRST_COURSE)), healthy.FIRST_COURSE, c="r", label="Starter")
    plt.scatter(range(len(healthy.FIRST_COURSE)), healthy.SECOND_COURSE, c="b", label="Main")
    plt.scatter(range(len(healthy.FIRST_COURSE)), healthy.THIRD_COURSE, c="g", label="Dessert")
    plt.savefig("Data/ditribution heelthy client.png") #lo guardamos en una imagen
    plt.close(fig)


    fig = plt.figure()
    plt.title("Dish ditribution business client")
    plt.scatter(range(len(business.FIRST_COURSE)), business.FIRST_COURSE, c="r", label="Starter")
    plt.scatter(range(len(business.FIRST_COURSE)), business.SECOND_COURSE, c="b", label="Main")
    plt.scatter(range(len(business.FIRST_COURSE)), business.THIRD_COURSE, c="g", label="Dessert")
    plt.savefig("Data/Dish ditribution business client.png") #lo guardamos en una imagen
    plt.close(fig)
    
    fig = plt.figure()
    plt.title("Dish ditribution retirement client")
    plt.scatter(range(len(retirement.FIRST_COURSE)), retirement.FIRST_COURSE, c="r", label="Starter")
    plt.scatter(range(len(retirement.FIRST_COURSE)), retirement.SECOND_COURSE, c="b", label="Main")
    plt.scatter(range(len(retirement.FIRST_COURSE)), retirement.THIRD_COURSE, c="g", label="Dessert")
    plt.savefig("Data/Dish ditribution retirement client.png") #lo guardamos en una imagen
    plt.close(fig)
    
    fig = plt.figure()
    plt.title("Dish ditribution onetime client")
    plt.scatter(range(len(onetime.FIRST_COURSE)), onetime.FIRST_COURSE, c="r", label="Starter")
    plt.scatter(range(len(onetime.FIRST_COURSE)), onetime.SECOND_COURSE, c="b", label="Main")
    plt.scatter(range(len(onetime.FIRST_COURSE)), onetime.THIRD_COURSE, c="g", label="Dessert")
    plt.savefig("Data/Dish ditribution onetime client.png") #lo guardamos en una imagen
    plt.close(fig)
    
    """Aqui se utiliza esta funciion para graficar la distribución de las bebidas de la misma manera que para los platos"""
    drink_distribution(healthy)
    drink_distribution(business)
    drink_distribution(retirement)
    drink_distribution(onetime)


    """set simulation parameters"""
    days = 365
    years = 5
    courses = 20
    size = courses*days*years

    """Create DataFrame"""

    cluster = df

    client_id = np.random.choice(cluster.CLIENT_ID)
    time = np.random.choice(cluster.TIME, size=size)
    customerType = np.random.choice(cluster.CLIENT_TYPE)
    course_cost1 = np.random.choice(cluster.DISH1, size=size)
    course_cost2 = np.random.choice(cluster.DISH2, size=size)
    course_cost3 = np.random.choice(cluster.DISH3, size=size)
    course1 = [name_dish(dish,0) for dish in course_cost1]
    course2 = [name_dish(dish,1) for dish in course_cost2]
    course3 = [name_dish(dish,2) for dish in course_cost3]
    drink1 = np.random.choice(cluster.DRINK1, size=size)
    drink2 = np.random.choice(cluster.DRINK2, size=size)
    drink3 = np.random.choice(cluster.DRINK3, size=size)
    total1 = np.array(course_cost1)+np.array(drink1)
    total2 = np.array(course_cost2)+np.array(drink2)
    total3 = np.array(course_cost3)+np.array(drink3)
    df = pd.DataFrame({
        'TIME': time, 
        'CUSTOMERID': client_id, 
        'CUSTOMERTYPE': customerType, 
        'COURSE1': course1,
        'COURSE2': course2,
        'COURSE3': course3,
        'DRINKS1': drink1,
        'DRINKS2': drink2, 
        'DRINKS3': drink3,
        'TOTAL1': total1,
        'TOTAL2': total2,
        'TOTAL3': total3,
    })

    df.to_csv('Data/DataSimulation.csv')

if __name__ == '__main__':
    run()