import pickle
import os
from pypolycontain.utils.random_polytope_generator import *
import copy

def load_polytopes_from_file(file_path):
    with open(file_path, 'rb') as f:
        polytopes_matrices = pickle.load(f)
    polytopes = []
    for pm in polytopes_matrices:
        polytopes.append(AH_polytope(pm[0], pm[1], polytope(pm[2], pm[3])))
    return polytopes

def get_pickles_in_dir(dir_path):
    filenames = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".p"):
            filenames.append(filename)
    return sort_by_filename_time(filenames)

def sort_by_filename_time(file_list):
    times = []
    for file in file_list:
        times.append(float(file.split('_')[0]))
    # sort names from file
    times, file_list = zip(*sorted(zip(times, file_list)))
    return file_list, times

def get_polytope_sets_in_dir(dir_path):
    files, times = get_pickles_in_dir(dir_path)
    polytope_sets = []
    print('Loading files...')
    for f in files:
        print(f)
        polytopes = load_polytopes_from_file(dir_path + '/' + f)
        polytope_sets.append(copy.deepcopy(polytopes))
    print('Files loaded!')
    return polytope_sets, times