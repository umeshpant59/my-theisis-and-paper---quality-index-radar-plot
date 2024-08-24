from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import math
from sympy.plotting import plot
from sympy.plotting import plot_parametric as pp
from sympy.plotting import plot3d as p3d
from sympy.plotting import plot3d_parametric_line as p3dpp
from sympy import plot_implicit
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import pandas as pd

var('x y z')
init_printing()
graph_groups=['fc/fcp','Ec/Ecp','Rho']
#max_values = [1,1,200]

def get_max_values():
    data = read_data_from_excel()
    max_values = list(data["Max_Val"].values())
    return max_values


def read_data_from_excel():
    df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
    return df.to_dict()

def get_max_val_to_plot():
    data = read_data_from_excel()
    max_val = []
    for dat in list(data['Effectiveness'].values()):
        max_val.append((dat/5))
    return max_val

def squish_it(value,k):
    return (1-exp(-value/k))/(1+exp(-value/k))

def squish_it_normal(value,k):
    return 1/(1+exp(-value/k))

def offset_data(min,max,val):
    #it returns offset and max data after ofsetting
    avg = (min+max)/2
    return val-avg,max-avg

def find_k(upper_limit,conf = 0.95):
    var('k')
    express = (1-exp(-x/k))/(1+exp(-x/k))-conf
    sol = solve(express,k)[0]
    return sol.subs({x:upper_limit})

def find_k_normal(upper_limit,conf = 0.95):
    var('k')
    express = (1)/(1+exp(-x/k))-conf
    sol = solve(express,k)[0]
    return sol.subs({x:upper_limit})


def get_list_of_squished_values(input):
    '''
    input is the list of n item tuples
    '''
    excel_data = read_data_from_excel()
    
    output = []
    #ks = [find_k(item) for item in get_max_values()]
    #print(ks)
    type = excel_data['Type']
    max_val = excel_data['Max_Val']
    min_val = excel_data["Min_Val"]
    for item in input:
        squished = []
        for i,val in enumerate(item):
            max_val_cur = max_val[i]
            min_val_cur = min_val[i]
            if type[i]=="N":
                k = find_k(max_val_cur)
                squished.append(squish_it(val,k))
            elif type[i] == "E":
                offset_dat,max_offset = offset_data(min_val_cur,max_val_cur,val)
                k = find_k_normal(max_offset)
                #k = find_k(max_val_cur)
                squished.append(squish_it_normal(offset_dat,k))
            elif type[i] == "R":
                k = find_k(max_val_cur)
                squish = squish_it(val,k)
                corrected = 0.95-squish
                squished.append(corrected)
            #squished = [squish_it(val,ks[i]) for i,val in enumerate(item)]
        output.append(squished)

    print(output)
    return output

def read_input(file_path):
    inputs = []
    with open(file_path,'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            vals = line.rstrip().split(',')
            values = [float(item) for item in vals]
            inputs.append(values)
    return inputs

def plot_radar_chart(data,fig_name='1'):
    my_data =read_data_from_excel()
    titles = list(my_data['NDT_Method'].values())
    categories = titles
    fig = go.Figure()
    #x = float(data[0])
    #y = float(data[1])
    #z = float(data[2])
    #data = [x,y,z]
    data = [float(i) for i in data]
    print(data,fig_name)

    angles = find_angles()
    theta_val = None
    if angles is None:
        theta_val = categories
    else:
        theta_val=np.cumsum(angles)

    r_val = get_max_val_to_plot()
    #scaled_data = [r_val[i]*data[i] for i in range(len(r_val))]
    #scaled_data = [data[i] for i in range(len(r_val))]
    scaled_data = data
    print(f"r_value:{r_val}")
    diff = [abs(r_val[i]-scaled_data[i]) for i in range(len(scaled_data))]
    inner_ring = [scaled_data[i]-diff[i] for i in range(len(diff))]
    

    #Outer poly
    fig.add_trace(go.Scatterpolar(
        r=[1 for i in data],
        theta = theta_val,
        text=categories,
        fill='toself'
    ))


    #Confidence poly
    #fig.add_trace(go.Scatterpolar(
    #    r=r_val,
    #    theta = theta_val,
    #    fillcolor = 'red',
    #    opacity = 0.3,
    #    text=categories,
    #    fill='toself'
    #))

    #Data Poly
    fig.add_trace(go.Scatterpolar(
        r=scaled_data,
        fillcolor = 'green',
        opacity = 0.5,
        theta=theta_val,
        fill='toself'
    ))


    #Lower Confidence Poly
    """fig.add_trace(go.Scatterpolar(
        r=inner_ring,
        theta=theta_val,
        fill='toself'
    ))"""

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=theta_val,
                ticktext=categories
            )
        ),

        showlegend=False
    )


    #fig.show()
    fig.write_image(f"{fig_name}.jpeg")
    coords_data,coords_extern = get_coords(theta_val,scaled_data)
    confidence_coords,_ = get_coords(theta_val,r_val)
    area_confidence = find_area(confidence_coords)
    outer_area = find_area(coords_extern.copy())
    upd_index = find_UPD_index(coords_data,coords_extern)
    return float(upd_index),area_confidence,outer_area

def get_coords(angles,data):
    coords_data = []
    coords_extern = []
    for i in range(len(angles)):
        c= float(cos(rad(angles[i])))
        s = float((sin(rad(angles[i]))))
        x_coord = (data[i]*c)
        y_coord = (data[i]*s)
        coords_data.append((x_coord,y_coord))
        coords_extern.append((c,s))
    return coords_data,coords_extern

def find_UPD_index(coords_data,coords_extern):
    #x1,y1 = 0,data_set[0]
    #x2,y2 = -data_set[1]*cos(math.pi/6),-data_set[1]*sin(math.pi/6)
    #x3,y3 = data_set[2]*cos(math.pi/6),-data_set[2]*sin(math.pi/6)
    #area = (x1*y2-x2*y1+x2*y3-y2*x3+x3*y1-y3*x1)/2
    #return area/(3*3**(1/2)/4)
    outer_area = find_area(coords_extern.copy())
    inner_area = find_area(coords_data.copy())
    return inner_area/outer_area



def find_area(coords):
    """
    Coords is a set of tuples containing coordinates.
    Returns areaa of the polygon
    """
    coords.append(coords[0])
    area = 0.0
    for i in range(len(coords)-1):
        first_set = coords[i]
        second_set = coords[i+1]
        x_1 = first_set[0]
        y_1 = first_set[1]
        x_2 = second_set[0]
        y_2 = second_set[1]
        temp_val = x_1*y_2 - y_1*x_2
        area += temp_val
    return 1/2* abs(area)
        
def find_angles():
    data = read_data_from_excel()
    values = list(data["Regression"].values())
    regressions = [(1-k**2) for k in values]
    angles = [360/(sum(regressions)) * i for i in regressions]
    return angles
    

def main():
    my_data = read_input("data.txt")
    squished_vals = get_list_of_squished_values(my_data)    
    indexes = []
    #print(squished_vals)
    conf = None
    outer_a = None
    for i,val in enumerate(squished_vals):
        #print(val)
        index,conf,outer_a = plot_radar_chart(val,str(i))
        indexes.append(index)
        print(f"Done for {i}")

    #max_vals = get_max_val_to_plot()
    #conf = 1
    #for i in max_vals:
    #    conf = conf*i

    perc_conf = conf/outer_a*100

    with open("indexes.txt",'w') as f:
        for index in indexes:
            f.write(f"{str(index)},{perc_conf}\n")
            
            
    


if __name__ == '__main__':
    main()
    #x = {("A","B"):0.4,("B","C"):0.2,("C","D"):0.1,("D","E"):0.5,("E","A"):0.6}
    #find_angles(x)
    #plot_radar_chart([0.6,0.7,0.3,0.2,0.1],angles=[50,100,60,50,100])
    #x = [0.907155169306521, 0.961090249071977, 0.932350921793159] 
    #plot_radar_chart(x,'0')







