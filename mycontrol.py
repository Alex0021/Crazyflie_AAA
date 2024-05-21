# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from matplotlib.animation import FuncAnimation

IN_SIM = False
OUTPUT_CMD = True

# Global variables
if IN_SIM:
    STARTING_POSE = [0,0]
    UPDATE_FREQUENCY = 10
    yaw_desired = 0.3
else:
    STARTING_POSE = [2.0,1.5]
    UPDATE_FREQUENCY = 2
    yaw_desired = 0.3 * 180 / np.pi   

GROWING_FACTOR = 8
PREV_HEIGHT_UPDATE = 4
LANDING_PAD_THRESHOLD = 0.035
on_ground = True
DEFAULT_HEIGHT = 0.3
height_desired = DEFAULT_HEIGHT
timer = None
ctrl_timer = None
startpos = None
timer_done = None
going_down = False
mode = 'takeoff' # 'takeoff', 'find goal', 'land'
firstpass_goal = np.array([4.5, 1.5]) # Location of the first goal
goal = firstpass_goal
canvas = None
fwd_vel_prev = 0
left_vel_prev = 0
prev_pos = []
prev_dpos = []
prev_ddpos = []
next_edge_goal = None
num_loops_stuck = 0
prev_command = np.zeros(4)
k_a = 1.0 # gain attraction force
k_r = 0.85 # gain repulsive force
k_s = 0.1 # gain stucknees force
cmd_alpha = 0.5
permanant_obstacles = 0
first_landpad_location = None
second_landpad_location = None
middle_landpad_location = None
stabilizing = False
stabilize_counter = 0
landpad_timer = 0
is_landed = False
is_landed_finale = False
prev_height = 0
num_possible_pads_locations = None
possible_pad_locations = None
list_of_visited_locations = np.empty((0,2), dtype=object)
grid_switcher = 0
grid_index = 0
prev_range_down = 0
grade = 0 # Change Grade to change type of control
waiting_takeoff = False
"""
Grade 4.0: Take off, avoid obstacles and reach the landing region whilst being airborne
Grade 4.5: Land on the landing pad
Grade 5.0: Take off from the landing pad and leave the landing region whilst being airborne
Grade 5.25: Avoid obstacles and reach the starting region whilst being airborne
Grade 5.5: Land on the take-off pad
Grade + 0.25: Detect and pass through the pink square during flight from the starting region towards the landing region
Grade + 0.25: Pass through the location of the pink square during flight from the landing region towards the starting region
"""


# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py lines 296-323. 
# The "item" values that you can later use in the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "range_down": Downward range finder distance (Used instead of Global Z distance)
# "range_front": Front range finder distance
# "range_left": Leftward range finder distance 
# "range_right": Rightward range finder distance
# "range_back": Backward range finder distance
# "yaw": Yaw angle (rad)

#============================#
# GENERATE_GRID_SEARCH_POINTS
#===========================#
RES_X = 0.2
RES_Y = 0.4
x_coords = np.arange(3.7, 4.7+RES_X, RES_X)
y_coords = np.arange(0.3, 2.75, RES_Y)
x_coords = np.outer(x_coords, np.ones_like(y_coords))
y_coords = np.concatenate([[y_coords if i % 2 == 0 else y_coords[::-1]] for i in range(len(x_coords))])
GRID_POINTS = np.array([x_coords.flatten(), y_coords.flatten()]).T
prev_pos
NB_POINTS_HIST = 200
fig_pose = plt.figure(1, figsize=(8, 6))
axs = fig_pose.subplots(3,1, sharex=True)
artists = {"line_height": axs[0].plot([], [], "r-", lw=2)[0],
           "line_dheight": axs[1].plot([], [], "b-", lw=2)[0],
           "line_ddheight": axs[2].plot([], [], "g-", lw=2)[0]}

anim = None

# Plotting pos history
def init_plots():
    global axs
    axs[0].set_title("Height History")
    axs[0].set_xlim(-NB_POINTS_HIST, 0)
    axs[0].set_ylim(0, 0.75)
    axs[1].set_title("dh History")
    axs[1].set_ylim(-0.15, 0.15)
    axs[2].set_title("ddh History")
    axs[2].set_ylim(-0.15, 0.15)
    return artists.values()

def frame_iter():
    global prev_pos, prev_dpos, prev_ddpos
    yield ([p[2] for p in prev_pos], [p[2] for p in prev_dpos], [p[2] for p in prev_ddpos])

def render(data):
    global artists
    y, dy, ddy = data
    artists["line_height"].set_data(np.arange(-len(y)+1, 1), y)
    artists["line_dheight"].set_data(np.arange(-len(dy)+1, 1), dy)
    artists["line_ddheight"].set_data(np.arange(-len(ddy)+1, 1), ddy)
    return artists.values()

def filter(cmd):
    return cmd_alpha * np.array(cmd) + (1 - cmd_alpha) * np.array(prev_command)

# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data=None, dt=0.1):
    global on_ground, startpos, mode, ctrl_timer, t, fwd_vel_prev, left_vel_prev, yaw_desired, height_desired, prev_range_down
    global prev_pos, num_loops_stuck, firstpass_goal, k_a, k_r, possible_pad_locations, num_possible_pads_locations, anim, waiting_takeoff
    global list_of_visited_locations, grade, goal, is_landed, landpad_timer, first_landpad_location, second_landpad_location, middle_landpad_location, prev_command
    global going_down
    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)

    #print("DT: ", dt)
    
    # Take off
    if startpos is None:
        startpos = [STARTING_POSE[0], STARTING_POSE[1], sensor_data['range_down']]    
    if on_ground and sensor_data['range_down'] < 0.3:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False

    #if anim is None:
        #anim = FuncAnimation(fig_pose, init_func=init_plots, frames=frame_iter, func=render, interval=5, cache_frame_data=True, blit=True)
        #plt.show(block=False)

    # adjust the relative position
    sensor_data['x_global'] += startpos[0]
    sensor_data['y_global'] += startpos[1]

    # ---- YOUR CODE HERE ----
    # Set Control Command
    control_command = [0.0, 0.0, height_desired, 0.0]
    
    # Get the occupancy map data
    map = occupancy_map(sensor_data)

    # Decide Where the Goal is Based on the Grade
    assign_goal(sensor_data, map)

    # Get Drone location
    drone_location = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down']])

    match mode:
        case 'takeoff':
            # Set a timer and rise to desired height for 3 seconds
            # if ctrl_timer is None:
            #     ctrl_timer = time.time()

            # if time.time() - ctrl_timer < 1:
            #     control_command = [0.0, 0.0, height_desired, 0.0]
            #     return control_command
            # else:
            mode = 'find goal'
        case 'find goal':
        

            if t % UPDATE_FREQUENCY == 0:

                # Check if goal is reached
                # if goal_reached(drone_location[:2], goal):
                #     # Set previous velocities
                #     fwd_vel_prev = 0
                #     left_vel_prev = 0
                #     yaw_desired = 0.0
                #     control_command = [0.0, 0.0, height_desired, 0.0]

                # else:
                # Get the vector from the drone to the goal
                attractive_force, attractive_force_wf, attractive_magnitude = calc_attractive_force(sensor_data) 

                # Get the repulsive force from nearby obstacles
                repulsive_force, repulsive_force_wf, repulsive_magnitude = calc_repulsive_force(sensor_data, map)

                # Adjust attractive and repulsive gains based on if the drone is stuck
                #adjust_gains(drone_location, prev_pos)
                # Get the avoidance force if drone is stuck
                stuckness_force = stuckness_avoidance(attractive_force, attractive_force_wf, repulsive_force_wf, attractive_magnitude)

                # Calculate Resultant Force in Body Frame
                resultant_force = (k_a*attractive_force) + (k_r*repulsive_force) + (k_s*stuckness_force)

                update_visualization(sensor_data, map, attractive_force_wf, attractive_magnitude, repulsive_force_wf, repulsive_magnitude)

                # Set the forward and left velocities
                fwd_vel = resultant_force[0] / attractive_magnitude / 5
                left_vel = resultant_force[1] / attractive_magnitude / 5
                #print(f"fwd: {fwd_vel:.3f}, left: {left_vel:.3f}")

                # Set control command to move towards the goal while avoiding obstacles
                control_command = [fwd_vel, left_vel, height_desired, yaw_desired]
                
                # Set previous velocities
                fwd_vel_prev = fwd_vel
                leftprev_pos_vel_prev = left_vel

            else:
                control_command = prev_command

        case 'land':

            if grade == 5.5:
                control_command = [0.0, 0.0, 0.01, 0.0]
                if sensor_data['range_down'] < 0.06:
                    control_command = [0.0, 0.0,-10 , 0.0]
                    print("HELL YEAH!!! 6.0 MAMA")
            else:
                # If we only have one landing pad location, continue moving towards goal until we have the second one
                if second_landpad_location is None and landpad_timer < 4 * 100:
                    # Ignore repulsive forces and move towards goal
                    # print('First Landing Pad Location Found! Continue Moving Towards Goal')
                    attractive_force, attractive_force_wf, attractive_magnitude = calc_attractive_force(sensor_data)
                    control_command = [attractive_force[0] / 20, attractive_force[1] / 20, height_desired, 0.0]
                    landpad_timer += 1
                    #control_command = [0.0, 0.0, height_desired, 0.0]
                    #print('Landpad Timer: ', landpad_timer)
                elif landpad_timer >= 4 * 200:
                    print('No Second Landing Pad Found. Just Fail me already!')
                    mode = 'find goal'
                    first_landpad_location = None
                    second_landpad_location = None 
                    middle_landpad_location = None
                    landpad_timer = 0
                    control_command = [0.0, 0.0, height_desired, 0.0]

                elif second_landpad_location is not None and not goal_reached(drone_location[:2], goal, tol=0.05) and not going_down: # If we have two landing pad locations, move towards the midpoint between the two
                    #print('Second Landing Pad Location Found! Move Towards the Midpoint Between the Two Landing Pads')
                    attractive_force, attractive_force_wf, attractive_magnitude = calc_attractive_force(sensor_data)
                    control_command = [attractive_force[0] / attractive_magnitude / 5, attractive_force[1] / attractive_magnitude / 5, height_desired, 0.0]
                else:
                    if not is_landed:
                        #print('Landing on the Landing Pad')
                        going_down = True
                        prev_command[2] -= 0.02    #0.002
                        control_command = [0.0, 0.0, prev_command[2], 0.0]
                        np.save('pos_hist', {'height': prev_pos, 'dheight': prev_dpos, 'ddheight': prev_ddpos})
                        print("landing sequence")
                        #control_command = [0.0,0.0, -10, 0.0]
                        
                        if sensor_data['range_down'] <  0.01:    #0.02:
                            control_command = [0.0, 0.0, 0.0, 0.0]
                            print("landing pad touched, switch to takeoff")
                            is_landed = True
                            waiting_takeoff = True
                            landpad_timer = 0
                            #mode = 'find goal'
                            goal = STARTING_POSE
                            grade = 5.0
                            print('Landed on the Landing Pad. \nGrade Increased to 5.0')
                    else:
                        if waiting_takeoff:
                            if landpad_timer < 10:
                                control_command = [0.0, 0.0, 0.0 + landpad_timer*0.025, 0.0]
                                landpad_timer += 1.0
                                print("landing timer: ", landpad_timer)
                            else:
                                print("Going HOME")
                                waiting_takeoff = False
                                control_command = [0.0, 0.0, height_desired, 0.0]
                                mode = 'find goal'
                                goal = STARTING_POSE
                                grade = 5.25
                        else:
                            mode = 'find goal' 
                            # landpad_timer = 0
                            control_command = [0.0, 0.0, height_desired, 0.0]
            #if t % UPDATE_FREQUENCY == 0:
                #update_visualization(sensor_data, map, attractive_force_wf, attractive_magnitude, [1,1], )
                        

    #map = occupancy_map(sensor_data)
    if OUTPUT_CMD:
        print("Cmd: ", control_command)

    if t % PREV_HEIGHT_UPDATE == 0:
        prev_range_down = sensor_data['range_down']
        # Update the previous position
        prev_pos.append(drone_location)
        # First derivative of position
        if len(prev_pos) > 1:
            prev_dpos.append((drone_location - prev_pos[-2]))
        # Second order derivative
        if len(prev_dpos) > 1:
            prev_ddpos.append((prev_dpos[-1] - prev_dpos[-2]))
        if len(prev_pos) > NB_POINTS_HIST:
            prev_pos.pop(0)
        if len(prev_dpos) > NB_POINTS_HIST:
            prev_dpos.pop(0)
        if len(prev_ddpos) > NB_POINTS_HIST:
            prev_ddpos.pop(0)
    t += 1
    control_command[0], control_command[1] = clip_cmd(control_command, 0.14)
    control_command = filter(control_command)
    prev_command = control_command
    return control_command # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]

def clip_cmd(cmd, v_max):
    va, vb = cmd[0], cmd[1]
    if abs(va) > abs(vb):
        if abs(va) > v_max:
            reduction = v_max / abs(va)
            va *= reduction
            vb *= reduction
    if abs(vb) > abs(va):
        if abs(vb) > v_max:
            reduction = v_max / abs(vb)
            va *= reduction
            vb *= reduction
    return va, vb

def goal_reached(drone_location, goal, tol=0.075):
    """
    Check if the drone has reached the goal location
    """
    if np.linalg.norm(drone_location - goal) < tol:
        return True
    else:
        return False

def calc_attractive_force(sensor_data):
    '''
    Middle Layer Method: Attractive Force Calculation
    '''
    attractive_force_wf, attractive_magnitude = get_vector_wf_to_goal(sensor_data) # Get world frame Vector from drone location to Goal
    attractive_force = convert_to_body_frame(attractive_force_wf, sensor_data['yaw']) # Convert from world frame to body frame
    attractive_force, attractive_force_wf, attractive_magnitude = ensure_strength(attractive_force, attractive_force_wf, attractive_magnitude) # Ensure the strength of the attractive force is not too high
    return attractive_force, attractive_force_wf, attractive_magnitude

def calc_repulsive_force(sensor_data, map):
    '''
    Middle Layer Method: Repulsive Force Calculation
    '''
    drone_location = np.array([sensor_data['x_global'], sensor_data['y_global']])
    repulsive_force_wf, repulsive_magnitude = compute_repulsive_force(map, drone_location)
    repulsive_force = convert_to_body_frame(repulsive_force_wf, sensor_data['yaw'])
    return repulsive_force, repulsive_force_wf, repulsive_magnitude

def adjust_gains(drone_location, prev_pos):
    '''
    Middle Layer Method: Adjust Gains based on if the drone is stuck.
    '''
    global k_a, k_r
    if len(prev_pos) > 0:
        if is_stuck(drone_location, prev_pos):
            # If the drone is stuck, increase attractive forcea and decrease repulsive force
            k_a *= 1.05
            k_r *= 0.95
        else:
            # Reset the gains
            k_a = 1.0
            k_r = 1.8
    return

def assign_goal(sensor_data, map):
    '''
    Assigns the goal location based on the current goal. Eg. Cross the map, find landing pad, find pink box, etc.
    '''
    global mode, firstpass_goal, grade, list_of_visited_locations, goal, first_landpad_location, second_landpad_location, prev_height, middle_landpad_location, height_desired
    global grid_index, height_desired, stabilizing, stabilize_counter, next_edge_goal, y_coords, GRID_POINTS, prev_command
    drone_location = np.array([sensor_data['x_global'], sensor_data['y_global']])
    match mode:
        case 'takeoff':
            firstpass_goal[1] = 0.5 if drone_location[1] < 1.5 else 2.5 # Location of the first goal
            return firstpass_goal # Take off to the first goal
        case 'find goal':
            
            # First Goal: Get to the Other Side
            if grade == 0.0: # Change later when adding visualization
                if drone_location[0] > 3.5:
                    print('Increase Grade to 4.0')
                    grade = 4.0
                    #height_desired = sensor_data['range_down']
                    if drone_location[1] > 1.5:
                        # Regenerate the grid search points
                        y_coords = np.arange(0.2, 2.8, RES_Y)
                        y_coords = np.concatenate([[y_coords if i % 2 == 1 else y_coords[::-1]] for i in range(len(x_coords))])
                        GRID_POINTS = np.array([x_coords.flatten(), y_coords.flatten()]).T
                    firstpass_goal = GRID_POINTS[grid_index]

                return firstpass_goal
            
            # Second Goal: Find and Land on the Landing Pad
            elif grade == 4.0 or grade == 4.5:
                # Do grid search for the landing pad by assigning next goal in grid
                # Update Visited Locations making sure not to add the same location twice
                # print('Range Down: ', sensor_data['range_down'])
                landing = edge_detected(sensor_data)
                if landing and stabilizing:
                    if stabilize_counter < 10*UPDATE_FREQUENCY:
                        stabilize_counter += 1
                        return goal
                    else:
                        stabilizing = False
                        goal = next_edge_goal
                if landing:
                    #height_desired = sensor_data['range_down']
                    height_desired = DEFAULT_HEIGHT - 0.115
                    print('First Landing Pad Location Found!')
                    #height_desired = sensor_data['range_down']
                    prev_height = sensor_data['range_down']
                    first_landpad_location = drone_location
                    #stabilizing = True
                    print('First Landing Pad Location: ', first_landpad_location)
                    mode = 'land'
                    if goal_reached(drone_location[:2], goal, tol=0.2):
                        grid_index = (grid_index + 1) % len(GRID_POINTS)
                        print('Goal to close to landing zone! Next: ', goal)
                        goal = GRID_POINTS[grid_index]
                    
                    dir = goal - drone_location
                    goal = drone_location + dir/np.linalg.norm(dir) * 0.10
                    middle_landpad_location = goal
                    second_landpad_location = goal
                    print("Current pos: ", drone_location)
                    print("Next goal: ", goal)
                    #next_edge_goal = goal
                    #stabilize_counter = 0
                    #goal = first_landpad_location + np.array([0.0, 0.01])
                    return goal
                else: 
                    if goal_reached(drone_location[:2], goal):
                        # Assign the next goal location
                        # print('Made it to the Goal Location!')
                        grid_index = (grid_index + 1) % len(GRID_POINTS)
                        print('Next Goal: ', goal)
                    elif waypoint_obstructed(goal, map, margin=0.2):
                        grid_index = (grid_index + 1) % len(GRID_POINTS)
                        print('OBSTRUCTED GOAL, going to next: ', goal)

                    goal = GRID_POINTS[grid_index]
                    return goal
        
                
            elif grade >= 5.0:
                
                # If drone is not at starting position, return to starting position
                if goal_reached(drone_location, goal):
                    #print('Return to the Starting Location: ', startpos[:2])
                    goal = STARTING_POSE
                    mode = 'land'
                    grade = 5.5
                    return goal
                else:
                    if edge_detected(sensor_data):
                        height_desired = DEFAULT_HEIGHT
                        print("Adjusting height: ", height_desired)
                    # print('Increase Grade to 5.5')
                    return goal
        
        case 'land':
            # Find when the down sensor jumps to a higher number (When it leaves the landing pad)
            if grade == 5.5:
                # Land on starting pad
                return goal
            
            
            if edge_detected(sensor_data) and second_landpad_location is None:
                #height_desired = sensor_data['range_down']
                print('Second Landing Pad Location Found!')
                second_landpad_location = drone_location
                print('Second Landing Pad Location: ', second_landpad_location)
                # Save numpy data
                np.save('pos_hist', {'height': prev_pos, 'dheight': prev_dpos, 'ddheight': prev_ddpos})
                
                # Change the goal location to the midpoint between the two landing pad locations
                goal = (first_landpad_location + second_landpad_location) / 2
                middle_landpad_location = goal
                print("Moving to middle of landing pad: ", goal)
                return goal
            # elif edge_detected(sensor_data) and second_landpad_location is not None:
            #     # adjust height
            #     print("Adjusting height to landing pad")
            #     height_desired = sensor_data['range_down']
            #     goal = middle_landpad_location
            #     return goal
            elif second_landpad_location is not None:
                # Change the goal location to the middle of the landing pad and the starting location
                goal = middle_landpad_location
                return goal
            else:
                if goal_reached(drone_location[:2], goal):
                    # Set the goal to the first landing pad location
                    grid_index = (grid_index + 1) % len(GRID_POINTS)
                    goal = GRID_POINTS[grid_index]
                    print('Next Goal: ', goal)
                return goal
            
            
def edge_detected(sensor_data):
    '''
    This function checks if the drone is above the landing pad.
    '''
    #print(abs(sensor_data['range_down'] - prev_range_down))
    #print(abs(sensor_data['range_down'] - prev_range_down))
    if abs(sensor_data['range_down'] - prev_range_down) > LANDING_PAD_THRESHOLD:
        return True
    else:
        return False

def waypoint_obstructed(goal, map, margin=0.3):
    map = map.copy() < -0.2
    for (x,y), value in np.ndenumerate(map):
        if value:
            pos_obstacle = np.array([x, y]) * res_pos
            if np.linalg.norm(pos_obstacle - goal) < margin:
                return True
    return False

def is_stuck(current_pos, prev_pos, threshold=0.2, N=25*UPDATE_FREQUENCY):
    """
    Check if the drone is stuck in one position. If more than N loops, then the drone is stuck.
    params:
    current_pos: Current position of the drone in the world frame (x, y)
    prev_pos: Previous 5 positions of the drone in the world frame [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
    num_loops_stuck: Number of loops the drone has been stuck in one position
    threshold: Threshold for determining if the drone is stuck in meters
    """
    global num_loops_stuck, firstpass_goal, goal, mode, first_landpad_location, second_landpad_location, middle_landpad_location
    # Get the Distances between the current position and the previous N positions
    prev_pos_np = np.array(prev_pos[-N:])  # Convert the last N positions to a NumPy array
    distances = np.linalg.norm(current_pos - prev_pos_np, axis=1)

    # Check if the drone is stuck
    if np.all(distances < threshold):
        num_loops_stuck += 1
        # print('Drone is stuck in one position for {} loops'.format(num_loops_stuck))
        mode = 'find goal'
        first_landpad_location = None
        second_landpad_location = None 
        middle_landpad_location = None
        if num_loops_stuck > 75*UPDATE_FREQUENCY and num_loops_stuck < 100*UPDATE_FREQUENCY:
            # print('Drone is stuck in one position for {} loops. Drone is Stuck! Change Goal Location.'.format(num_loops_stuck))
            goal = np.array([4.0, 0.3 ]) # Change the goal location
        elif num_loops_stuck >= 75*UPDATE_FREQUENCY:
            # print('Drone is stuck in one position for {} loops. Drone is Stuck! Change Goal Location.'.format(num_loops_stuck))
            goal = np.array([4.0, 2.7]) # Change the goal location back to the original
        return True
    else:
        if num_loops_stuck > 0:
            if num_loops_stuck > 100*UPDATE_FREQUENCY:
                goal = firstpass_goal # Change the goal location back to the original
            # print('Drone is not stuck anymore!')
        num_loops_stuck = 0
        return False
    
def stuckness_avoidance(attractive_force, attractive_force_wf, repulsive_force_wf, attractive_magnitude):
    """
    Returns an "avoidance_force" when the angle between the "attraction_force_wf" and "replusion_froce_wf" is more than a certain angle.
    Input:
        * attraction_force: 2d-vector from drone to goal given in body-frame
        * attraction_force_wf: 2d-vector from drone to goal given in world-frame
        * repulsive_force_wf: 2d-vector from obstacle to drone given in world-frame
    Output:
        * avoidance_force: 2d-vector perpendicular to "attraction_force_wf" given in body-frame
    """
    
    def rotate_2d_vector(vector, angle):
        """
        Helper function to rotate 2D vector by "angle" amount
        """
        return np.dot(np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]), vector)
    

    avoidance_force = np.array([0,0])
    
    if attractive_magnitude > 0.3: # only do stuckness avoidance if we are more than 0.3m from goal (probably needs tuning)
        
        # Calculate angle between repulsion vector and attraction vector
        w = repulsive_force_wf
        v = attractive_force_wf
        angle_attraction_repulsion = np.arctan2(w[1]*v[0] - w[0]*v[1], w[0]*v[0] + w[1]*v[1])
        #print(angle_attraction_repulsion)
        
        if abs(angle_attraction_repulsion) > 150/180 * np.pi: 
            if angle_attraction_repulsion > 0:
                avoidance_force = rotate_2d_vector(attractive_force, -np.pi/2) # +
            else:
                avoidance_force = rotate_2d_vector(attractive_force, np.pi/2) # -

    return avoidance_force

def compute_repulsive_force(occupancy_map, drone_location):
    repulsive_force = np.zeros(2)  # Initialize repulsive force vector
    drone_location = np.flip(drone_location)  # Swap X and Y axes because input is by default (y,x)
    # Find indices of obstacles in the occupancy map
    obstacle_indices = np.flip(np.where(occupancy_map < 0))
    # print('\n\n#########################################################\n\n')
    # print('Number of Obstacles Before', obstacle_indices[0].shape[0])
    if len(obstacle_indices[0]) > 0:  # Check if obstacles are present
        # Compute repulsive forces from each obstacle
        obstacle_locations = np.column_stack((obstacle_indices[0], obstacle_indices[1]))
        
        # Convert Obstacle Locations to from Grid to Meters
        obstacle_locations = obstacle_locations * res_pos
        # print('Obstacle Locations Before Limit: ', obstacle_locations)
        # print('Drone Location: ', drone_location)
        distances = np.linalg.norm(obstacle_locations - drone_location, axis=1)

        
        directions = (drone_location - obstacle_locations) / distances[:, np.newaxis]
        # print('Distances: ', distances)

        # Delete Obstacles more than N meters away from the drone location
        N = 0.4
        
        obstacle_locations = obstacle_locations[distances < N]
        directions = directions[distances < N]
        distances = distances[distances < N]
        # Get indices of obstacles within N meters
        close_obstacle_indices = np.where(distances < N)
        num_close_obstacles = close_obstacle_indices[0].shape[0]
        # print('Number of Obstacles After', num_close_obstacles)
        # print('Obstacle Locations: ', obstacle_locations)
        # print('Directions: ', directions)

        magnitudes = 1 / distances**1.8 / num_close_obstacles # Example: inverse-distance function
        # Sum up repulsive forces from all obstacles
        repulsive_force = np.flip(np.sum(magnitudes[:, np.newaxis] * directions, axis=0))
    
    # Calculate the magnitude of the repulsive force
    magnitude = np.linalg.norm(repulsive_force)
    # print('Magnitude: ', magnitude)
    
    # Avoid division by zero
    if magnitude == 0: 
        magnitude = 1e-6
    #print('Magnitude: ', magnitude)
    return repulsive_force, magnitude

def get_vector_wf_to_goal(sensor_data):
    """
    Get the  normalized vector from the drone to the goal in the world frame
    """
    global goal
    vector = goal - np.array([sensor_data['x_global'], sensor_data['y_global']])
    magnitude = np.linalg.norm(vector)
    # magnitude = 5 # Set equal to 3 because we always want to move forward at a reasonable speed
    return vector, magnitude

def ensure_strength(vector, vector_wf, magnitude):
    """
    Ensure the strength of the vector is at least N:
    Good values for N range from 3 to 10
    """
    N = 5
    if magnitude < N:
        vector = (vector / magnitude) * N
        vector_wf = (vector_wf / magnitude) * N
        magnitude = N
    return vector, vector_wf, magnitude

def convert_to_body_frame(vector, yaw):
    """
    Convert the vector from world frame to body frame
    """
    rotation_matrix = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    vector = rotation_matrix@vector
    return vector

def update_visualization(sensor_data, map, attractive_force, attractive_magnitude, repulsive_force, repulsive_magnitude):
    global canvas, t, k_a, k_r, goal, possible_pad_locations, num_possible_pads_locations, UPDATE_FREQUENCY
    arrow_size = 15
    map_size_x = int(3/res_pos*GROWING_FACTOR)
    map_size_y = int(5/res_pos*GROWING_FACTOR)
    
    # Calculate Resultant Force in World Frame for Visualization
    resultant_force = (k_a*attractive_force) + (k_r*repulsive_force)
    
    if t % UPDATE_FREQUENCY == 0:
        #print(f'xglobal: {sensor_data["x_global"]}, yglobal: {sensor_data["y_global"]}')
        xdrone = int(sensor_data['y_global'] * 1/res_pos*GROWING_FACTOR)  # Swap X and Y axes
        ydrone = int(sensor_data['x_global'] * 1/res_pos*GROWING_FACTOR)  # Swap X and Y axes
        xgoal = int(goal[1] * 1/res_pos*GROWING_FACTOR)  # Swap X and Y axes
        ygoal = int(goal[0] * 1/res_pos*GROWING_FACTOR)  # Swap X and Y axes
        # print(f'xdrone: {xdrone}, ydrone: {ydrone}, xgoal: {xgoal}, ygoal: {ygoal}')
        if canvas is None:
            # Create an empty canvas
            canvas = np.zeros((map_size_y, map_size_x, 3), dtype=np.uint8) * 255  # Swap canvas dimensions

        # Clear canvas
        canvas.fill(255)
        
        # Plot the map with upscaling (Comment out if maps are the same size)
        map = np.kron(map, np.ones((GROWING_FACTOR, GROWING_FACTOR)))
        idx_obstacles = np.where(map < 0)

        canvas[map_size_y-idx_obstacles[0]-1, map_size_x-idx_obstacles[1]-1] = (0, 0, 255)  # Red
        # Plot Sensor Data
        text_position = (60, 20)  # Adjust as needed
        # text_position = (text_position[0], text_position[1] + 20)  # Move text position down for next item

        # Plot Sensor Data
        text_position = (10, 350)  # Adjust as needed
        cv2.putText(canvas, f'Range Down: {round(sensor_data["range_down"], 3)}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f'Range Front: {round(sensor_data["range_front"], 3)}', (text_position[0], text_position[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f'Range Left: {round(sensor_data["range_left"], 3)}', (text_position[0], text_position[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f'Range Right: {round(sensor_data["range_right"], 3)}', (text_position[0], text_position[1] + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f'Range Back: {round(sensor_data["range_back"], 3)}', (text_position[0], text_position[1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Plot drone and goal positions
        cv2.circle(canvas, (map_size_x - xdrone, map_size_y - ydrone), 8, (0, 0, 255), -1)  # Red for drone, mirror X coordinate
        cv2.circle(canvas, (map_size_x - xgoal, map_size_y - ygoal), 8, (255, 0, 0), -1)  # Blue for goal, mirror X coordinate

        # Plot the possible landing pad locations
        if num_possible_pads_locations is not None and num_possible_pads_locations > 0:
            for location in possible_pad_locations:
                cv2.circle(canvas, (map_size_x - int(location[1] * 10), map_size_y - int(location[0] * 10)), 5, (0, 255, 0), -1)

        # Plot the attractive force vector
        if attractive_magnitude != 0:
            arrow_end_point = (map_size_x - (xdrone + int(attractive_force[1] * arrow_size)), map_size_y - (ydrone + int(attractive_force[0] * arrow_size)))
            cv2.arrowedLine(canvas, (map_size_x - xdrone, map_size_y - ydrone), arrow_end_point, (0, 255, 0), thickness=2, tipLength=0.3)

        # Plot the repulsive force vector
        if repulsive_magnitude != 0:
            arrow_end_point = (map_size_x - (xdrone + int(repulsive_force[1] * arrow_size)), map_size_y - (ydrone + int(repulsive_force[0] * arrow_size)))
            cv2.arrowedLine(canvas, (map_size_x - xdrone, map_size_y - ydrone), arrow_end_point, (255, 0, 0), thickness=2, tipLength=0.3)

        # Plot the resultant force vector
        resultant_magnitude = np.linalg.norm(resultant_force)
        if resultant_magnitude != 0:
            arrow_end_point = (map_size_x - (xdrone + int(resultant_force[1] * arrow_size)), map_size_y - (ydrone + int(resultant_force[0] * arrow_size)))
            cv2.arrowedLine(canvas, (map_size_x - xdrone, map_size_y - ydrone), arrow_end_point, (0, 0, 0), thickness=2, tipLength=0.3)
        
        # Show the updated canvas
        cv2.imshow("Map", canvas)
        cv2.waitKey(1)  # Wait for a short time to update the display


# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 3.0 # meter
range_max = 2.0 # meter, maximum range of distance sensor
res_pos = 0.05 # meter
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

    # add a perimeter of obstacles around the occupancy map
    map = add_perimeter_obstacles(map)

    # only plot every Nth time step (comment out if not needed)
    # if t % 50 == 0:
    #     plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
    #     plt.savefig("map.png")
    #     plt.close()
    return map

# Add a boarder of obstacles around the perimeter of the occupancy map
def add_perimeter_obstacles(map):
    '''
    This function adds a perimeter of obstacles around the occupancy map
    '''
    map[0, :] = -1
    map[-1, :] = -1
    map[:, 0] = -1
    map[:, -1] = -1
    return map


# Control from the exercises
index_current_setpoint = 0
def path_to_setpoint(path,sensor_data,dt):
    global on_ground, height_desired, index_current_setpoint, timer, timer_done, startpos

    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down']]    
    if on_ground and sensor_data['range_down'] < 0.49:
        current_setpoint = [startpos[0], startpos[1], height_desired, 0.0]
        return current_setpoint
    else:
        on_ground = False

    # Start timer
    if (index_current_setpoint == 1) & (timer is None):
        timer = 0
        print("Time recording started")
    if timer is not None:
        timer += dt
    # Hover at the final setpoint
    if index_current_setpoint == len(path):
        # Uncomment for KF
        control_command = [startpos[0], startpos[1], startpos[2]-0.05, 0.0]

        if timer_done is None:
            timer_done = True
            print("Path planing took " + str(np.round(timer,1)) + " [s]")
        return control_command

    # Get the goal position and drone position
    current_setpoint = path[index_current_setpoint]
    x_drone, y_drone, z_drone, yaw_drone = sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down'], sensor_data['yaw']
    distance_drone_to_goal = np.linalg.norm([current_setpoint[0] - x_drone, current_setpoint[1] - y_drone, current_setpoint[2] - z_drone, clip_angle(current_setpoint[3]) - clip_angle(yaw_drone)])

    # When the drone reaches the goal setpoint, e.g., distance < 0.1m
    if distance_drone_to_goal < 0.1:
        # Select the next setpoint as the goal position
        index_current_setpoint += 1
        # Hover at the final setpoint
        if index_current_setpoint == len(path):
            current_setpoint = [0.0, 0.0, height_desired, 0.0]
            return current_setpoint

    return current_setpoint

def clip_angle(angle):
    angle = angle%(2*np.pi)
    if angle > np.pi:
        angle -= 2*np.pi
    if angle < -np.pi:
        angle += 2*np.pi
    return angle