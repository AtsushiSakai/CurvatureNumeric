#!/usr/bin/env python
# -*- coding:utf-8 -*-
#   Numeric Curvature Calcration Lib
#
#   author: Atsushi Sakai
from PyCubicSpline import PyCubicSpline

import numpy as np
import math

def CalcCurvature(x,y,npo=1):
    u"""
    Calc curvature
    x,y: x-y position list
    npo: the number of points using calcration curvature
    ex) npo=1: using 3 point
        npo=2: using 5 point
        npo=3: using 7 point
    """

    cv=[]

    ndata=len(x)

    for i in range(ndata):
        lind=i-npo
        hind=i+npo+1

        if lind<0:
            lind=0
        if hind>=ndata:
            hind=ndata
        #  print(lind,hind)

        xs=x[lind:hind]
        ys=y[lind:hind]
        #  print(xs,ys)
        (cxe,cye,re)=CircleFitting(xs,ys)
        #  print(re)

        if len(xs)>=3:
            # sign evalation 
            cind=int((len(xs)-1)/2.0)
            sign = (xs[0] - xs[cind]) * (ys[-1] - ys[cind]) - (ys[0] - ys[cind]) * (xs[-1] - xs[cind])

            # check straight line
            a = np.array([xs[0]-xs[cind],ys[0]-ys[cind]])
            b = np.array([xs[-1]-xs[cind],ys[-1]-ys[cind]])
            theta=math.degrees(math.acos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
            #  print(theta)

            if theta==180.0:
                cv.append(0.0)#straight line
            elif sign>0:
                cv.append(1.0/-re)
            else:
                cv.append(1.0/re)
        else:
            cv.append(0.0)

    #  print(cv)
    return cv

def CircleFitting(x,y):
    u"""Circle Fitting with least squared
        input: point x-y positions  

        output  cxe x center position
                cye y center position
                re  radius of circle 

    """

    sumx  = sum(x)
    sumy  = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix,iy) in zip(x,y)])

    F = np.array([[sumx2,sumxy,sumx],
                  [sumxy,sumy2,sumy],
                  [sumx,sumy,len(x)]])

    G = np.array([[-sum([ix ** 3 + ix*iy **2 for (ix,iy) in zip(x,y)])],
                  [-sum([ix ** 2 *iy + iy **3 for (ix,iy) in zip(x,y)])],
                  [-sum([ix ** 2 + iy **2 for (ix,iy) in zip(x,y)])]])

    try:
        T=np.linalg.inv(F).dot(G)
    except:
        return (0,0,float("inf"))

    cxe=float(T[0]/-2)
    cye=float(T[1]/-2)
    #  print (cxe,cye,T)
    try:
        re=math.sqrt(cxe**2+cye**2-T[2])
    except:
        return (cxe,cye,float("inf"))
    return (cxe,cye,re)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #input
    x=[0,1,2,3]
    y=[2.7,6,5,6.5]

    # 3d spline interporation
    spline=PyCubicSpline(y)
    rx=np.arange(0,3,0.01)
    ry=[spline.Calc(i) for i in rx]
    rc=[spline.CalcCurvature(i) for i in rx]

    nc=CalcCurvature(rx,ry)

    #  plt.plot(x,y,"xb")
    #  plt.plot(rx,ry,"-r")
    #  plt.axis("equal")
    plt.plot(rx,rc,"-b",label="True Curvature")
    plt.plot(rx,nc,"xr",label="Numeric Curvature")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()

 
