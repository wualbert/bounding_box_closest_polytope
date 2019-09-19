import pickle
from pypolycontain.utils.random_polytope_generator import *

def load_polytopes_from_file(file_path):
    file = open(file_path, 'rb')
    polytopes = pickle.load(file)
    file.close()
    return polytopes

p = get_uniform_random_zonotopes(3, 5)
