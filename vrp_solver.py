# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 09:42:02 2023

@author: Ryan Marizza

References:
    
    NPTEL Lecture series on Advanced Operations Research
    Lecture 29 - Vehicle Routing Problem
    by Prof. G.Srinivasan, Department of Management Studies, IIT Madras
    https://www.youtube.com/watch?v=_Y3eh23BcFQ&ab_channel=nptelhrd
    
    Urban Operations Research Section 6.4.12 by Richard C. Larson and Amedeo R. Odoni
    https://web.mit.edu/urban_or_book/www/book/chapter6/6.4.12.html     

"""

import sys
import numpy as np
import io
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def distance_between(p1, p2):
    delta_x = p1.x - p2.x
    delta_y = p1.y - p2.y
    return math.sqrt(delta_x**2 + delta_y**2)


class Load:
    def __init__(self, id, pickup, dropoff):
        self.id = id
        self.pickup = pickup
        self.dropoff = dropoff
    
    
class VRP:
    def __init__(self, loads):
        self.loads = loads

def load_problem_from_file(filePath):
    f = open(filePath, "r")
    problem_str = f.read()
    f.close()
    return problem_string_to_vrp_problem(problem_str)
    

def problem_string_to_vrp_problem(problem_str):
    loads = []
    buf = io.StringIO(problem_str)
    
    found_header = False
    
    while True:
        line = buf.readline()
        if not found_header:
            found_header = True
            continue
        if len(line) == 0:
            break
        line = line.replace("\n", "")
        splits = line.split()
        id = splits[0]
        pickup = get_point_from_point_string(splits[1])
        dropoff = get_point_from_point_string(splits[2])
        loads.append(Load(id, pickup, dropoff))
    
    return VRP(loads)


def get_point_from_point_string(point_str):
    point_str = point_str.replace("(","").replace(")","")
    points = point_str.split(",")
    
    return Point(float(points[0]), float(points[1]))


def calculate_savings(problem):
    depot = Point(0., 0.)
    num_locations = len(problem.loads)
    savings = np.zeros((num_locations, num_locations))

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                savings[i, j] = distance_between(depot, problem.loads[i].pickup)\
                + distance_between(problem.loads[j].pickup, depot)\
                - distance_between(problem.loads[i].dropoff, problem.loads[j].pickup)
                
    return savings


def sort_savings(savings):
    sorted_indices = np.argsort(-savings.flatten())
    sorted_pairs = np.unravel_index(sorted_indices, savings.shape)
    
    return sorted_pairs


def get_distance_of_route_with_return_to_depot(problem, schedule):
    distance = 0.0
    depot = Point(0,0)
    current_loc = depot
    
    for load_id in schedule:
        load = problem.loads[load_id]
        distance += distance_between(current_loc, load.pickup)
        current_loc = load.pickup
        distance += distance_between(current_loc, load.dropoff)
        current_loc = load.dropoff
        
    distance += distance_between(current_loc, depot)
    
    return distance


def get_cost_of_solution(problem, solution):

    total_distance = 0.0
    for idx, route in enumerate(solution):
        route_distance = get_distance_of_route_with_return_to_depot(problem, route)
        total_distance += route_distance
    
    return 500*len(solution) + total_distance


def get_best_solution(problem, sorted_pairs, branches):
    solutions = []
    costs = []
    current_branch = 1
    
    max_branches = get_branches_limit(problem, sorted_pairs, branches)
    
    while current_branch < max_branches:
        solution = solve_from_sorted_pairs(problem, sorted_pairs)
        cost = get_cost_of_solution(problem, solution)
        
        solutions.append(solution)
        costs.append(cost)
        
        sorted_pairs = get_sorted_pairs_for_next_branch(sorted_pairs)
        
        current_branch += 1
        
    return solutions[np.argmin(costs)]


def get_branches_limit(problem, sorted_pairs, branches):
    return min(branches, len(sorted_pairs[0]) - len(problem.loads))


def get_sorted_pairs_for_next_branch(sorted_pairs):
    new_sorted_pairs_from = sorted_pairs[0][1:]
    new_sorted_pairs_to = sorted_pairs[1][1:]
    return (new_sorted_pairs_from, new_sorted_pairs_to)


def solve_from_sorted_pairs(problem, sorted_pairs):
    solution = []
    load2route = {}
    final_routes = []
    
    for i,j in zip(sorted_pairs[0], sorted_pairs[1]):
        if i == j:
            continue
        
        #Case 1: loads i and j have both already been assigned to a route
        if is_load_in_a_route(i, load2route) and is_load_in_a_route(j, load2route):
            i_route_index = load2route[i]
            j_route_index = load2route[j]
            if i_route_index == j_route_index:
                continue
            #if i is at the end of a route and j is at the start of another, those routes can potentially be merged
            elif solution[i_route_index][-1] == i and solution[j_route_index][0] == j:
                proposed_route = solution[i_route_index] + solution[j_route_index]
                if is_proposed_route_feasible(problem, proposed_route):
                    solution[i_route_index] = proposed_route
                    for load_index in solution[j_route_index]:
                        load2route.update({load_index : i_route_index})
                    solution[j_route_index] = []
        
        #Case 2: load i has been assigned a route, but load j hasn't            
        elif is_load_in_a_route(i, load2route) and not is_load_in_a_route(j, load2route):
            i_route_index = load2route[i]
            #if load i is at the end of a route, j can potentially be added to the end of that route
            if solution[i_route_index][-1] == i:
                proposed_route = solution[i_route_index] + [j]
                if is_proposed_route_feasible(problem, proposed_route):
                    solution[i_route_index] = proposed_route
                    load2route.update({j : i_route_index})
                    
        #Case 3: load j has been assigned a route, but load i hasnt        
        elif is_load_in_a_route(j, load2route) and not is_load_in_a_route(i, load2route):
            j_route_index = load2route[j]
            
            if solution[j_route_index][0] == j:
                proposed_route = [i] + solution[j_route_index]
                if is_proposed_route_feasible(problem, proposed_route):
                    solution[j_route_index] = proposed_route
                    load2route.update({i : j_route_index})
        
        #Case 4: neither load i nor load j have been assigned to a route
        if not is_load_in_a_route(i, load2route) and not is_load_in_a_route(j, load2route):
            proposed_route = [i,j]
            if is_proposed_route_feasible(problem, proposed_route):
                load2route.update({i : len(solution), j : len(solution)})
                solution.append([i,j])
            else:
                load2route.update({i : len(solution)})
                solution.append([i])
                load2route.update({j : len(solution)})
                solution.append([j])
        
        #If load i still hasnt been assigned to a route, make a new route with only load i 
        if not is_load_in_a_route(i, load2route):
            load2route.update({i : len(solution)})
            solution.append([i])
        
        #If load j still hasnt been assigned to a route, make a new route with only load j
        if not is_load_in_a_route(j, load2route):
            load2route.update({j : len(solution)})
            solution.append([j])            

    final_routes = delete_empty_routes_from(solution)      
    
    return final_routes

def is_proposed_route_feasible(problem, proposed_route):
    MAX_TOTAL_DISTANCE = 12*60
    return get_distance_of_route_with_return_to_depot(problem, proposed_route) < MAX_TOTAL_DISTANCE

def is_load_in_a_route(load_id, load2route):
    return load_id in load2route
    

def delete_empty_routes_from(routes):
    final_routes = []
    for route in routes:
        if len(route) > 0:
            final_routes.append(route)   
    return final_routes


def print_solution(solution):
    for route in solution:
        output = str.encode('{}\n'.format(list(np.asarray(route) + 1), 'utf-8'))
        sys.stdout.buffer.write(output)
     
        
def main(argv):
    
    N_BRANCHES = 100
    
    #load the problem as a VRP object
    problem_file = argv[0]
    problem = load_problem_from_file(problem_file)
    
    #calculate then sort savings for Clarke & Wright savings
    savings = calculate_savings(problem)
    sorted_pairs = sort_savings(savings) 
    
    #get routes with Holmes and Parker algorithm
    solution = get_best_solution(problem, sorted_pairs, branches=N_BRANCHES)
        
    print_solution(solution)

if __name__ == "__main__":
   main(sys.argv[1:])
