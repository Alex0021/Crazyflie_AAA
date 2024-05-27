import numpy as np
import matplotlib.pyplot as plt
height_desired = 0.1

def spiral(n):
    n+=1 # Start counting qt 0. Adapting from matlab to Python
    k=np.ceil((np.sqrt(n)-1)/2)
    t=2*k+1
    m=t**2 
    t=t-1
    if n>=m-t:
        return k-(m-n),-k        
    else :
        m=m-t
    if n>=m-t:
        return -k,-k+(m-n)
    else:
        m=m-t
    if n>=m-t:
        return -k+(m-n),k 
    else:
        return k,k-(m-n-t)

def delete_points(points):
    to_keep = []
    to_keep.append(points[0])
    for i in range(1, len(points)-1):
        if points[i][0] == points[i-1][0] and points[i][0] == points[i+1][0]:
            pass
        elif points[i][1] == points[i-1][1] and points[i][1] == points[i+1][1]:
            pass
        else:
            to_keep.append(points[i])

    return to_keep

def create_spiral(points_in_spiral):
    myPoints = []
    for i in range(points_in_spiral):
        myPoints.append(spiral(i))
    myPoints = delete_points(myPoints)
    myPoints = np.concatenate([myPoints, height_desired*np.ones((len(myPoints),1))], axis=1)
    return myPoints

if __name__ == "__main__":
    points_in_spiral = 100
    myPoints = create_spiral(points_in_spiral)
    print(myPoints)
    myPoints = myPoints[:, :2]
    plt.plot(myPoints[:,0], myPoints[:,1])
    plt.scatter(myPoints[:,0], myPoints[:,1])
    plt.show()
