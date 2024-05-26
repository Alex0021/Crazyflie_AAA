# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

# Global variables
on_ground = True
height_desired = 0.5
timer = None
startpos = None
timer_done = None

# Self defined global variables
setpoint_idx = 0
mode = 1 
first_seen = 0
points_in_spiral = 50
created_spiral = False

setpoint_traj = np.array(
[[3.5, 1.5, 0.5],
 [3.67, 2.80, 0.5],
 [3.67, 2.60, 0.5],
 [3.67, 2.40, 0.5],
 [3.67, 2.20, 0.5],
 [3.67, 2.00, 0.5],
 [3.67, 1.80, 0.5],
 [3.67, 1.60, 0.5],
 [3.67, 1.40, 0.5],
 [3.67, 1.20, 0.5],
 [3.67, 1.00, 0.5],
 [3.67, 0.80, 0.5],
 [3.67, 0.60, 0.5],
 [3.67, 0.40, 0.5],
 [3.67, 0.20, 0.5],
 [3.88, 0.20, 0.5],
 [3.88, 0.40, 0.5],
 [3.88, 0.60, 0.5],
 [3.88, 0.80, 0.5],
 [3.88, 1.00, 0.5],
 [3.88, 1.20, 0.5],
 [3.88, 1.40, 0.5],
 [3.88, 1.60, 0.5],
 [3.88, 1.80, 0.5],
 [3.88, 2.00, 0.5],
 [3.88, 2.20, 0.5],
 [3.88, 2.40, 0.5],
 [3.88, 2.60, 0.5],
 [3.88, 2.80, 0.5],
 [4.06, 2.80, 0.5],
 [4.06, 2.60, 0.5],
 [4.06, 2.40, 0.5],
 [4.06, 2.20, 0.5],
 [4.06, 2.00, 0.5],
 [4.06, 1.80, 0.5],
 [4.06, 1.60, 0.5],
 [4.06, 1.40, 0.5],
 [4.06, 1.20, 0.5],
 [4.06, 1.00, 0.5],
 [4.06, 0.80, 0.5],
 [4.06, 0.60, 0.5],
 [4.06, 0.40, 0.5],
 [4.06, 0.20, 0.5],
 [4.26, 0.20, 0.5],
 [4.26, 0.40, 0.5],
 [4.26, 0.60, 0.5],
 [4.26, 0.80, 0.5],
 [4.26, 1.00, 0.5],
 [4.26, 1.20, 0.5],
 [4.26, 1.40, 0.5],
 [4.26, 1.60, 0.5],
 [4.26, 1.80, 0.5],
 [4.26, 2.00, 0.5],
 [4.26, 2.20, 0.5],
 [4.26, 2.40, 0.5],
 [4.26, 2.60, 0.5],
 [4.26, 2.80, 0.5],
 [4.44, 2.80, 0.5],
 [4.44, 2.60, 0.5],
 [4.44, 2.40, 0.5],
 [4.44, 2.20, 0.5],
 [4.44, 2.00, 0.5],
 [4.44, 1.80, 0.5],
 [4.44, 1.60, 0.5],
 [4.44, 1.40, 0.5],
 [4.44, 1.20, 0.5],
 [4.44, 1.00, 0.5],
 [4.44, 0.80, 0.5],
 [4.44, 0.60, 0.5],
 [4.44, 0.40, 0.5],
 [4.44, 0.20, 0.5],
 [4.62, 0.20, 0.5],
 [4.62, 0.40, 0.5],
 [4.62, 0.60, 0.5],
 [4.62, 0.80, 0.5],
 [4.62, 1.00, 0.5],
 [4.62, 1.20, 0.5],
 [4.62, 1.40, 0.5],
 [4.62, 1.60, 0.5],
 [4.62, 1.80, 0.5],
 [4.62, 2.00, 0.5],
 [4.62, 2.20, 0.5],
 [4.62, 2.40, 0.5],
 [4.62, 2.60, 0.5],
 [4.62, 2.80, 0.5],
 [4.82, 2.80, 0.5],
 [4.82, 2.60, 0.5],
 [4.82, 2.40, 0.5],
 [4.82, 2.20, 0.5],
 [4.82, 2.00, 0.5],
 [4.82, 1.80, 0.5],
 [4.82, 1.60, 0.5],
 [4.82, 1.40, 0.5],
 [4.82, 1.20, 0.5],
 [4.82, 1.00, 0.5],
 [4.82, 0.80, 0.5],
 [4.82, 0.60, 0.5],
 [4.82, 0.40, 0.5],
 [4.82, 0.20, 0.5]])

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py lines 296-323. 
# The "item" values that you can later use in the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "range_down": Downward range finder distance (Used instead of Global Z distance)
# "range_front": Front range finder distance
# "range_left": Leftward range finder distance 
# "range_right": Rightward range finder distance
# "range_back": Backward range finder distance
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)

# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data, dt):
    global on_ground, startpos, setpoint_idx, setpoint_traj, mode, first_seen, created_spiral

    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down']]    
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False

    # ---- YOUR CODE HERE ----
    map = occupancy_map(sensor_data)
    
    current_setpoint = setpoint_traj[setpoint_idx]

    if setpoint_reached(sensor_data, current_setpoint, margin=0.06):
         print(f"Setpoint {setpoint_idx} reached!")
         setpoint_idx += 1 
    
    if waypoint_obstructed(current_setpoint, map, margin=0.3):
        print(f"Deleted setpoint: {setpoint_idx}")
        setpoint_idx += 1

    # Detection of pad (needs to be changed for the hardware)
    if mode == 1 and sensor_data['range_down'] < 0.45 and sensor_data['x_global'] > 3.6: 
        if first_seen == 0:
            print("Potential landing pad detected")
            first_seen = sensor_data['t']
        
        if sensor_data['t'] - first_seen > 0.2:
            print("Landing pad detected for more than 0.2s")
            mode += 1  
    else:
        first_seen = 0
    
    if mode == 2: # Go down to landing pad
        current_setpoint = np.array([sensor_data['x_global'],sensor_data['y_global'],0.02])
    
    if sensor_data['range_down'] < 0.06 and mode == 2: 
        mode += 1
    
    if mode == 3: # Go back to takeoff pad
        current_setpoint = np.array([startpos[0],startpos[1], 0.5, 0])
    
    if mode == 3 and setpoint_reached(sensor_data, current_setpoint):
        if not created_spiral:
            myPoints = []
            for i in range(points_in_spiral):
                myPoints.append(spiral(i))
            myPoints = delete_points(myPoints)
            GRID_POINTS = current_setpoint + np.array(myPoints)
        mode += 1
    
    if mode == 4: # Descend to takeoff pad
        current_setpoint = np.array([startpos[0],startpos[1],0.05])
    
    control_command = potential_field(map, sensor_data, current_setpoint)
    
    return control_command # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]


# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 3.0 # meter
range_max = 2.0 # meter, maximum range of distance sensor
res_pos = 0.05 #0.032 #0.035 # meter
conf = 0.2 # certainty given by each measurement
t = 0 # only for plotting

map = np.zeros((int((max_x-min_x)/res_pos), int((max_y-min_y)/res_pos))) # 0 = unknown, 1 = free, -1 = occupied

def occupancy_map(sensor_data):
    global map, t
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    yaw = sensor_data['yaw']
    
    for j in range(4): # 4 sensors
        yaw_sensor = yaw + j*np.pi/2 #yaw positive is counter clockwise
        if j == 0:
            measurement = sensor_data['range_front']
        elif j == 1:
            measurement = sensor_data['range_left']
        elif j == 2:
            measurement = sensor_data['range_back']
        elif j == 3:
            measurement = sensor_data['range_right']
        
        for i in range(int(range_max/res_pos)): # range is 2 meters
            dist = i*res_pos
            idx_x = int(np.round((pos_x - min_x + dist*np.cos(yaw_sensor))/res_pos,0))
            idx_y = int(np.round((pos_y - min_y + dist*np.sin(yaw_sensor))/res_pos,0))

            # make sure the current_setpoint is within the map
            if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > range_max:
                break

            # update the map
            if dist < measurement:
                map[idx_x, idx_y] += conf
            else:
                map[idx_x, idx_y] -= conf
                break
    
    map = np.clip(map, -1, 1) # certainty can never be more than 100%

    # only plot every Nth time step (comment out if not needed)
    if t % 50 == 0:
        plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
        plt.savefig("map.png")
        plt.close()
    t +=1  

    return map


def obstacle_field(map, sensor_data, goal_vec):
    global res_pos, t

    map = map.copy()

    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']

    # Threshold to make binary map: 1 obstacle, 0 not obstacle (elegible)
    map = (map < -0.2) 

    # Make a virtual edge-border in the map (give unstability)
    #map[[0,-1],:] = 1
    #map[:,[0,-1]] = 1
    
    # Intialize vectors and distance variables
    vec_min1 = vec_min2 = np.zeros(2)
    d_min1 = d_min2 = np.inf

    # Maximum distance in which drone is being reppeld from obstacels
    max_obs_dist = 0.3 # 0.3
    # If the angle beteween vec_min1 and vec is greater than "angle_thresh" then vec_min2 = vec
    angle_thresh = np.pi/3
    
    for (x,y), value in np.ndenumerate(map):
        if value:
            vec = np.array([x*res_pos - pos_x, y*res_pos - pos_y])
            d = np.linalg.norm(vec)
            if d < max_obs_dist:
                if np.sum(vec_min1) == 0 or angle_between(vec, vec_min1) < angle_thresh:
                    if np.sum(vec_min2) == 0 or angle_between(vec, vec_min2) > angle_thresh:
                        if d < d_min1:
                            d_min1 = d
                            vec_min1 = vec  
                elif np.sum(vec_min2) == 0 or angle_between(vec, vec_min2) < angle_thresh:
                    if np.sum(vec_min1) == 0 or angle_between(vec, vec_min1) > angle_thresh:
                        if d < d_min2:
                            d_min2 = d
                            vec_min2 = vec

    # Repultion force grows linearly as the drone gets closer to the obstacle
    vec_min1 = vec_min1 * (max_obs_dist/d_min1 - 1) if d_min1 != 0 else np.array([0,0])
    vec_min2 = vec_min2 * (max_obs_dist/d_min2 - 1) if d_min2 != 0 else np.array([0,0])
    # Add the repultion force together In the case of closeby obstacels from multiple angles
    v = vec_min1 + vec_min2                         
                     
    if t % 200 == 0:
        w,h = map.shape
        map = map.astype(np.uint8)*255
        scale = 10
        map = cv2.resize(map, dsize=(h*scale,w*scale), interpolation=cv2.INTER_NEAREST)
        obs_vec = (v*scale/res_pos).astype(int)
        obs_vec1 = (vec_min1*scale/res_pos).astype(int)
        obs_vec2 = (vec_min2*scale/res_pos).astype(int)
        goal_vec = (goal_vec*scale/res_pos).astype(int)
        pos_vec = np.array([pos_x*scale/res_pos, pos_y*scale/res_pos], dtype=int)
        
        map = cv2.arrowedLine(map, (pos_vec[1], pos_vec[0]), (pos_vec[1] - obs_vec1[1], pos_vec[0] - obs_vec1[0]), (160,0,0), 2)
        map = cv2.arrowedLine(map, (pos_vec[1], pos_vec[0]), (pos_vec[1] - obs_vec2[1], pos_vec[0] - obs_vec2[0]), (160,0,0), 2)
        map = cv2.arrowedLine(map, (pos_vec[1], pos_vec[0]), (pos_vec[1] - obs_vec[1], pos_vec[0] - obs_vec[0]) , (255,0,0), 2)
        map = cv2.arrowedLine(map, (pos_vec[1], pos_vec[0]), (pos_vec[1] + goal_vec[1], pos_vec[0] + goal_vec[0]) , (255,0,0), 2)

        #cv2.imshow('Obstacles avoidance', map)
        #cv2.waitKey(1)    
    
    return v

def goal_field(pos_goal, sensor_data):
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    vec = np.array([pos_goal[0] - pos_x, pos_goal[1] - pos_y])
    vec = normalize_vector(vec)
    return vec


def potential_field(map, sensor_data, waypoint):
    
    map = map.copy()
    yaw = sensor_data['yaw']
    
    # Rotation matrix from world to body
    R_wb = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])

    goal_vec_world = goal_field(waypoint, sensor_data)
    goal_vec_body = R_wb @ goal_vec_world
    
    obstacle_vec_world = obstacle_field(map, sensor_data, goal_vec_world)
    obstacle_vec_body = R_wb @ obstacle_vec_world

    # avoid getting stuck when waypoint is on opposite side of obstacle
    l = stuckness_avoidance(goal_vec_body, goal_vec_world, obstacle_vec_world)
   
    # Avoid to abrupt altitude changes
    alt = alltitude_controller(sensor_data, waypoint, alt_error_thresh = 0.1) #
    
    K_goal = 0.3 
    K_obst = 2
    K_stck = 0.15 #0.3
    
    vel = K_goal*goal_vec_body - K_obst*obstacle_vec_body + K_stck*l
    yaw_rate = 1.2 #1.2 #1.4 #1.2

    control_command = [vel[0], vel[1], alt, yaw_rate]

    return control_command

def setpoint_reached(sensor_data, current_setpoint, margin = 0.06, with_z=False):
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    pos_z = sensor_data['z_global']

    setp_x = current_setpoint[0]
    setp_y = current_setpoint[1]
    setp_z = current_setpoint[2]

    if with_z:
        d = np.sqrt((pos_x - setp_x)**2 + (pos_y - setp_y)**2 + (pos_z - setp_z)**2)
    else:
        d = np.sqrt((pos_x - setp_x)**2 + (pos_y - setp_y)**2)

    return d < margin

def waypoint_obstructed(waypoint, map, margin=0.3):
    map_thresh = map.copy() < -0.2
    for (x,y), value in np.ndenumerate(map_thresh):
        if value:
            pos_obstacle = np.array([x,y])*res_pos
            pos_waypoint = np.array([waypoint[0],waypoint[1]])
            vec_obs2wayp = pos_waypoint - pos_obstacle
            if np.linalg.norm(vec_obs2wayp) < margin:
                return True
    return False

def stuckness_avoidance(goal_vec_body, goal_vec_world, obstacle_vec_world):
    l = np.array([0,0])
    
    if np.linalg.norm(goal_vec_world) > 0.3:
        w = -obstacle_vec_world
        v = goal_vec_world
        angle_goal_obs = np.arctan2(w[1]*v[0] - w[0]*v[1], w[0]*v[0] + w[1]*v[1])
        
        if abs(angle_goal_obs) > 120/180 * np.pi:
            if angle_goal_obs > 0:
                l = rotate_2d_vector(goal_vec_body, -np.pi/2) # +
            else:
                l = rotate_2d_vector(goal_vec_body, np.pi/2) # -

    return l


def alltitude_controller(sensor_data, waypoint, alt_error_thresh=0.3):
    alt_current = sensor_data['range_down']
    alt_desired = waypoint[2]
    alt_error = alt_desired - alt_current

    if alt_error > alt_error_thresh:
        return alt_current + alt_error_thresh
    elif alt_error < -alt_error_thresh:
        return alt_current - alt_error_thresh
    else:
        return alt_desired
    


# HELPERS:
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


def normalize_vector(v, threshold=1e-1):
    norm = np.linalg.norm(v)
    if norm < threshold:
        normalized_v = np.zeros_like(v)  # Or any other suitable action
    else:
        normalized_v = v / norm
    return normalized_v

def rotate_2d_vector(vector, angle):
    return np.dot(np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]), vector)

def angle_between(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(dot_product)
    return angle

def clip_angle(angle):
    angle = angle%(2*np.pi)
    if angle > np.pi:
        angle -= 2*np.pi
    if angle < -np.pi:
        angle += 2*np.pi
    return angle

def print_sensor_data(sensor_data):
    # print(f"t: {sensor_data['t']}")
    # print(f"x_global: {sensor_data['x_global']}")
    # print(f"y_global: {sensor_data['y_global']}")
    # print(f"z_global: {sensor_data['z_global']}")
    # print(f"roll: {sensor_data['roll']}")
    # print(f"pitch: {sensor_data['pitch']}")
    # print(f"yaw: {sensor_data['yaw']}")
    # print(f"q_x: {sensor_data['q_x']}")
    # print(f"q_y: {sensor_data['q_y']}")
    # print(f"q_z: {sensor_data['q_z']}")
    # print(f"q_w: {sensor_data['q_w']}")
    # print(f"v_x: {sensor_data['v_x']}")
    # print(f"v_y: {sensor_data['v_y']}")
    # print(f"v_z: {sensor_data['v_z']}")
    # print(f"v_forward: {sensor_data['v_forward']}")
    # print(f"v_left: {sensor_data['v_left']}")
    # print(f"v_down: {sensor_data['v_down']}")
    # print(f"ax_global: {sensor_data['ax_global']}")
    # print(f"ay_global: {sensor_data['ay_global']}")
    # print(f"az_global: {sensor_data['az_global']}")
    # print(f"range_front: {sensor_data['range_front']}")
    # print(f"range_left: {sensor_data['range_left']}")
    # print(f"range_back: {sensor_data['range_back']}")
    # print(f"range_right: {sensor_data['range_right']}")
    # print(f"range_down: {sensor_data['range_down']}")
    # print(f"rate_roll: {sensor_data['rate_roll']}")
    # print(f"rate_pitch: {sensor_data['rate_pitch']}")
    # print(f"rate_yaw: {sensor_data['rate_yaw']}")
    pass



   

   



