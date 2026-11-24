# stress constraint as according to literature
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from sys import exit
import time
import nlopt
from numpy import *
import builtins  # use builtins.min for scalar min(a,b); "from numpy import *" shadows built-in min
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib.colors import LinearSegmentedColormap
import joblib
import gc  # Garbage collection for memory management
# Add this import at the top
from line_profiler import LineProfiler
np.set_printoptions(precision=15)    #maby remove again later
np.set_printoptions(threshold=np.inf)
#from scipy.sparse import csr_matrix
#from scipy.linalg import cho_factor, cho_solve

from tempfile import TemporaryFile

from tempfile import TemporaryFile
outfile = TemporaryFile()

# Start timing the optimization process
import time
optimization_start_time = time.time()

# Initialize compliance history tracking
compliance_history = []

### inputs
# domain size
length_x = 2  # length of domain in x-direction (make x an integer-multiple of y length)
length_y = 1  # legnth of domain in y-direction
# number of elements on x-axis
Nel_x = 96  # number of elements on x-axis 48, 96, 192, 384
Force_level = -240
# to ensure square elements
Nel_y = int(Nel_x * length_y / length_x)  # number of elements on y-axis

Delta_x = length_x / Nel_x
r = 1.0 * Delta_x  # set to 3.5 for finite difference checks in the functions below, since they all use 3.5
r_original = 1.0 * Delta_x  # make this match global r
# p-factor for getting rid of gray area's
p = 4
p_2 = 4

ETA_1 = 0.5  # ETA_1 for solid/void and material 1/2 filtering
beta_max = 50
# analysis = "plain_stress"           #plain_strain or plain_stress

grid_size_x = Nel_x + 1
grid_size_y = Nel_y + 1

NUM_LOAD_NODES = 7  # bottom-right node plus 6 nodes to the left along x


def build_force_vector(dtype=np.float64):
    """Y-direction load on bottom row: rightmost node and 6 nodes to its left."""
    Force = np.zeros(2 * grid_size_x * grid_size_y, dtype=dtype)
    f_per_node = Force_level / NUM_LOAD_NODES
    i_bottom = grid_size_y - 1
    for k in range(NUM_LOAD_NODES):
        j = grid_size_x - 1 - k
        node = j + i_bottom * grid_size_x
        Force[2 * node + 1] = f_per_node
    return Force


# create the mesh
x = np.linspace(0, length_x, grid_size_x)
y = np.linspace(length_y, 0, grid_size_y)

xv, yv = np.meshgrid(x, y, sparse=False)

# create list of elements (list of nodes)
elements = [[] for _ in range(Nel_x * Nel_y)]

for i in range(Nel_y):  # row number (horizontal) (y)
    for j in range(Nel_x):  # column number (vertical) (x)
        elements[j + i * (grid_size_x - 1)] = [j + grid_size_x * (i + 1), j + 1 + grid_size_x * (i + 1),
                                               j + 1 + grid_size_x * i, j + grid_size_x * i]



# create list of coordinates of the nodes for every element
coordinates = [[] for _ in range(Nel_x * Nel_y)]

rnd = 6
for i in range(Nel_y):
    for j in range(Nel_x):
        # coordinates[j + i*(grid_size_x-1)] = [round(xv[i+1,j],rnd), round(yv[i+1,j],rnd), round(xv[i+1,j+1],rnd), round(yv[i+1,j+1],rnd), round(xv[i,j+1],rnd), round(yv[i,j+1],rnd), round(xv[i,j],rnd), round(yv[i,j],rnd)]
        coordinates[j + i * (grid_size_x - 1)] = [xv[i + 1, j], yv[i + 1, j],
                                                  xv[i + 1, j + 1], yv[i + 1, j + 1],
                                                  xv[i, j + 1], yv[i, j + 1],
                                                  xv[i, j], yv[i, j]]

#print('coordinates [quad element coordinates [x, y ...]]', coordinates)

## create a list of all the coordinates of every single node (standalone)
nodes = [[] for _ in range(grid_size_x * grid_size_y)]

for i in range(grid_size_y):
    for j in range(grid_size_x):
        # nodes[j + i*grid_size_x] = [round(xv[i,j],rnd), round(yv[i,j],rnd)]
        nodes[j + i * grid_size_x] = [xv[i, j], yv[i, j]]

#print('nodes [coordinates of labeled nodes [x, y]]', nodes)

# get list of centres for all elements for the Bendsoe & Sigmund filter
centre_of_element = []
for i in range(len(elements)):
    centre_of_element.append([(coordinates[i][0] + coordinates[i][2] + coordinates[i][4] + coordinates[i][6]) / 4,
                              (coordinates[i][1] + coordinates[i][3] + coordinates[i][5] + coordinates[i][7]) / 4])


beta = 0.01


# Add this after your mesh setup (around line 100)
# Pre-compute and cache the Helmholtz matrix
def create_helmholtz_matrix(r, Nel_x, Nel_y, Delta_x):
    """Create the Helmholtz matrix once and cache it"""
    N = Nel_x * Nel_y
    H = sparse.lil_matrix((N, N))
    r2 = r * r
    
    for i in range(N):
        row = i // Nel_x
        col = i % Nel_x
        
        num_neighbors = 4
        if col == 0 or col == Nel_x-1:
            num_neighbors -= 1
        if row == 0 or row == Nel_y-1:
            num_neighbors -= 1
            
        H[i, i] = 1 + num_neighbors*r2/(Delta_x**2)
        
        if col > 0:
            H[i, i-1] = -r2/(Delta_x**2)
        if col < Nel_x-1:
            H[i, i+1] = -r2/(Delta_x**2)
        if row > 0:
            H[i, i-Nel_x] = -r2/(Delta_x**2)
        if row < Nel_y-1:
            H[i, i+Nel_x] = -r2/(Delta_x**2)
    
    return H.tocsr()


def create_helmholtz_nodal_matrix(r, Nel_x, Nel_y, Delta_x):
    """Create the nodal Helmholtz matrix once and cache it"""
    Nnodes_x = Nel_x + 1
    Nnodes_y = Nel_y + 1
    N = Nnodes_x * Nnodes_y
    H = sparse.lil_matrix((N, N), dtype=np.float64)
    r2 = r * r

    for i in range(N):
        row = i // Nnodes_x
        col = i % Nnodes_x

        num_neighbors = 4
        if col == 0 or col == Nnodes_x-1:
            num_neighbors -= 1
        if row == 0 or row == Nnodes_y-1:
            num_neighbors -= 1

        H[i, i] = 1 + num_neighbors*r2/(Delta_x**2)

        if col > 0:
            H[i, i-1] = -r2/(Delta_x**2)
        if col < Nnodes_x-1:
            H[i, i+1] = -r2/(Delta_x**2)
        if row > 0:
            H[i, i-Nnodes_x] = -r2/(Delta_x**2)
        if row < Nnodes_y-1:
            H[i, i+Nnodes_x] = -r2/(Delta_x**2)

    return H.tocsr()


# Create cached matrices
H_solid = create_helmholtz_matrix(r, Nel_x, Nel_y, Delta_x)
H_solid_2 = create_helmholtz_matrix(0.8*r, Nel_x, Nel_y, Delta_x)
H_nodal_solid = create_helmholtz_nodal_matrix(r, Nel_x, Nel_y, Delta_x)
H_nodal_solid_2 = create_helmholtz_nodal_matrix(0.8*r, Nel_x, Nel_y, Delta_x)
#H_type = create_helmholtz_matrix(r, Nel_x, Nel_y, Delta_x)

from scipy.sparse.linalg import splu

# Pre-factorize the Helmholtz matrices for faster repeated solves
H_solid_T_factor = splu(H_solid.T.tocsc())
#H_type_T_factor = splu(H_type.T.tocsc())

#helmholtz PDE
def helmholtz_filter(g, r, Nel_x, Nel_y, Delta_x):
    # Use pre-calculated Helmholtz matrix H_solid or H_solid_2 based on r value
    # Store original mean for normalization
    original_mean = np.mean(g)

    # Determine which cached matrix to use based on r value (must match Helmholtz_cursor21 logic)
    # Use fixed reference: r_original = base radius, r_2_value = 0.8 * r_original, so that
    # when caller passes r = 0.8*r (second stage), we select H_solid_2.
    #r_original = 3.5 * Delta_x  # matches global r
    r_2_value = 0.8 * r_original
    if abs(r - r_2_value) < 1e-10:
        H_to_use = H_solid_2
    else:
        H_to_use = H_solid

    # Solve the system H*new_g = g using the cached matrix
    new_g = spsolve(H_to_use, g)

    # Normalize to preserve mean density
    new_g = new_g * (original_mean / np.mean(new_g))

    return new_g

# Test case - verify optimization works correctly
#r = 3.5 * Delta_x  # Filter radius
g_test = np.full(Nel_x * Nel_y, 0.2)  # Test array with all values = 0.2
g_filtered = helmholtz_filter(g_test, r, Nel_x, Nel_y, Delta_x)

# Verification: Test that optimized functions produce same results
print("Verifying Helmholtz optimization...")
print(f"Test array mean: {np.mean(g_test):.6f}")
print(f"Filtered array mean: {np.mean(g_filtered):.6f}")
print(f"Mean preservation check: {np.abs(np.mean(g_test) - np.mean(g_filtered)) < 1e-10}")
print("Helmholtz optimization verification completed.")


def step_filter_element(g_test, r, Nel_x, Nel_y, Delta_x, beta):
    # First Helmholtz filter with radius r
    densities_1 = helmholtz_filter(g_test, r, Nel_x, Nel_y, Delta_x)

    # First projection step with ETA_1
    Filtered_variables = (np.tanh(beta*ETA_1) + np.tanh(beta*(densities_1 - ETA_1))) / (np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1)))

    del densities_1

    return Filtered_variables

# Example usage:
#r = 3.5 * Delta_x  # Base filter radius
g_test = np.full(Nel_x * Nel_y, 0.2)  # Test array
Filtered_densities = step_filter_element(g_test, r, Nel_x, Nel_y, Delta_x, beta)

## Lazy sensitivity matrix (from ISC2 lines 2003+) - re-used for compliance and constraints
class _TransposeLazySensitivity:
    """Wrapper so that (LazySensitivityMatrix.T @ v) works without forming full matrix."""
    def __init__(self, lazy_mat):
        self._lazy = lazy_mat

    def __matmul__(self, v):
        v = np.asarray(v).ravel()
        N = self._lazy.N
        out = np.zeros(N, dtype=np.float64)
        for j in range(N):
            out[j] = np.dot(v, self._lazy.get_column(j))
        return out


class LazySensitivityMatrix:
    """Lazy sensitivity matrix that computes columns on-demand to save memory. Uses ETA_1."""
    def __init__(self, g_test, r, Nel_x, Nel_y, Delta_x, beta):
        self.N = Nel_x * Nel_y
        self.Nel_x = Nel_x
        self.Nel_y = Nel_y
        self.shape = (self.N, self.N)

        densities_1 = helmholtz_filter(g_test, r, Nel_x, Nel_y, Delta_x)

        self.H = H_solid
        beta_x = beta * (densities_1 - ETA_1)
        cosh_term = np.cosh(beta_x)
        tanh_term = np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1))
        self.proj_derivative = beta * (1.0 / (cosh_term * cosh_term)) / tanh_term
        self.mean_preservation = float(np.mean(densities_1) / np.mean(g_test))
        del densities_1, beta_x, cosh_term, tanh_term

        self.H_T_lu = splu(self.H.T.tocsc())
        self._column_cache = {}

    def get_column(self, j):
        if j in self._column_cache:
            return self._column_cache[j]
        if self.proj_derivative[j] != 0:
            rhs_j = np.zeros(self.N, dtype=np.float64)
            rhs_j[j] = 1.0
            col = self.proj_derivative[j] * self.H_T_lu.solve(rhs_j)
            col = self.mean_preservation * col
        else:
            col = np.zeros(self.N, dtype=np.float64)
        return col

    @property
    def T(self):
        return _TransposeLazySensitivity(self)


def step_filter_sensitivity(g_test, r, Nel_x, Nel_y, Delta_x, beta, vec=None):
    """
    If vec is provided: compute (sensitivity_matrix.T @ vec) with one solve (fast matvec).
    If vec is None: return LazySensitivityMatrix for .T @ v and column access.
    Uses ETA_1. Same as ISC2 first-part 7/8-arg behavior.
    """
    if vec is not None:
        N = Nel_x * Nel_y
        densities_1 = helmholtz_filter(g_test, r, Nel_x, Nel_y, Delta_x)
        beta_diff = beta * (densities_1 - ETA_1)
        sech_squared = 1.0 / np.cosh(beta_diff)**2
        tanh_denom = np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1))
        proj_derivative = beta * sech_squared / tanh_denom
        mean_factor = np.mean(densities_1) / np.mean(g_test)
        H_inv_vec = spsolve(H_solid, np.asarray(vec).ravel())
        return mean_factor * proj_derivative * H_inv_vec
    return LazySensitivityMatrix(g_test, r, Nel_x, Nel_y, Delta_x, beta)

#RAM optimized version (slower, 196)
def Compliance_sensitivity(U, K_e_sparse, K_e_2_sparse, new_g, new_g_2, der_solid, der_type):
    
    num_elements = len(elements)
    elements_array = np.array(elements)

    DOF = np.column_stack([
        np.column_stack((2*elements_array[:, 0], 2*elements_array[:, 0]+1)),
        np.column_stack((2*elements_array[:, 1], 2*elements_array[:, 1]+1)),
        np.column_stack((2*elements_array[:, 2], 2*elements_array[:, 2]+1)),
        np.column_stack((2*elements_array[:, 3], 2*elements_array[:, 3]+1))
    ])

    u_all = U[DOF]

    K_e_dense = K_e_sparse.toarray()
    K_e_2_dense = K_e_2_sparse.toarray()

    temp_1 = u_all @ K_e_dense
    temp_2 = u_all @ K_e_2_dense

    uKu_1_all = np.sum(temp_1 * u_all, axis=1)
    uKu_2_all = np.sum(temp_2 * u_all, axis=1)

    del temp_1, temp_2, u_all, K_e_dense, K_e_2_dense

    ng2p = new_g_2 ** p_2
    # Keep term1/term2 strictly 1D — do not mix (N,) with (N,1) or NumPy broadcasts to (N,N).
    term1 = -p * new_g ** (p - 1) * (ng2p * uKu_2_all + (1.0 - ng2p) * uKu_1_all)
    term2 = -(new_g ** p) * p_2 * new_g_2 ** (p_2 - 1) * (uKu_2_all - uKu_1_all)

    del uKu_1_all, uKu_2_all

    sens_solid = np.zeros(num_elements, dtype=np.float64)
    sens_type = np.zeros(num_elements, dtype=np.float64)

    term1_1d = np.asarray(term1, dtype=np.float64).ravel()
    term2_1d = np.asarray(term2, dtype=np.float64).ravel()

    for i in range(num_elements):
        if term1_1d[i] != 0:
            if isinstance(der_solid, LazySensitivityMatrix):
                der_solid_col_i = der_solid.get_column(i)
            else:
                der_solid_col_i = der_solid[:, i]

            sens_solid += term1_1d[i] * der_solid_col_i

        if term2_1d[i] != 0:
            if isinstance(der_type, LazySensitivityMatrix):
                der_type_col_i = der_type.get_column(i)
            else:
                der_type_col_i = der_type[:, i]

            sens_type += term2_1d[i] * der_type_col_i

    del term1, term2, term1_1d, term2_1d

    return sens_solid, sens_type


#fast version
def Compliance_sensitivity_fast(U, K_e_sparse, K_e_2_sparse, new_g, new_g_2, g_solid, g_type, beta, ETA_1):
    
    num_elements = len(elements)
    elements_array = np.array(elements)

    DOF = np.column_stack([
        np.column_stack((2*elements_array[:, 0], 2*elements_array[:, 0]+1)),
        np.column_stack((2*elements_array[:, 1], 2*elements_array[:, 1]+1)),
        np.column_stack((2*elements_array[:, 2], 2*elements_array[:, 2]+1)),
        np.column_stack((2*elements_array[:, 3], 2*elements_array[:, 3]+1))
    ])

    u_all = U[DOF]
    K_e_dense = K_e_sparse.toarray()
    K_e_2_dense = K_e_2_sparse.toarray()

    uKu_1_all = np.einsum('ei,ij,ej->e', u_all, K_e_dense, u_all)
    uKu_2_all = np.einsum('ei,ij,ej->e', u_all, K_e_2_dense, u_all)

    new_g_p_m1 = new_g ** (p - 1)
    new_g_p = new_g ** p
    new_g_2_p2 = new_g_2 ** p_2
    new_g_2_p2_m1 = new_g_2 ** (p_2 - 1)

    term1 = -p * new_g_p_m1 * (new_g_2_p2 * uKu_2_all + (1 - new_g_2_p2) * uKu_1_all)
    term2 = -new_g_p * p_2 * new_g_2_p2_m1 * (uKu_2_all - uKu_1_all)

    sens_solid = step_filter_sensitivity(g_solid, r, Nel_x, Nel_y, Delta_x, beta, term1)
    sens_type = step_filter_sensitivity(g_type, r, Nel_x, Nel_y, Delta_x, beta, term2)

    return sens_solid, sens_type


#initialize 
g = np.full(2 * Nel_x * Nel_y, 0.2)  # Test array
step = 0.01   #0,01      0.001

mid_index = len(g) // 2
g_solid = g[:mid_index].copy()  # Make explicit copy
g_type = g[mid_index:].copy()   # Make explicit copy
g_type[0] = 0.4                 # Set the first element of g_type to 0.4
# Create new arrays for FD instead of modifying copies
g_solid_FD = g_solid.copy()     # Make a fresh copy for FD
g_type_FD = g_type.copy()       # Make a fresh copy for FD

# Now modify the FD arrays
g_solid_FD[0] = g_solid[0] + step  # Use explicit value instead of +=
g_type_FD[0] = g_type[0] + step    # Use explicit value instead of +=

print('Original vs Perturbed values:')
print(f'g_solid[0]: {g_solid[0]}, g_solid_FD[0]: {g_solid_FD[0]}')
print(f'g_type[0]: {g_type[0]}, g_type_FD[0]: {g_type_FD[0]}')
#r = 3.5 * Delta_x  # Base filter radius  3.5

#new_g, step1 = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta, ETA_1)
#new_g_2, step1 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta, ETA_1)


#stiffness matrix (same logic as ISC2 lines 2635+, with Faster material order: K_e_2 type 2, K_e type 1)
def StiffnessMatrix(new_g, new_g_2, elements, K_e, K_e_2, p, p_2, grid_size_x, grid_size_y):
    num_elements = len(elements)
    nodes_per_element = 8  # 4 nodes × 2 DOF per node

    # Convert sparse to dense only once
    K_e_dense = K_e.toarray()
    K_e_2_dense = K_e_2.toarray()

    # Pre-compute all K_temp matrices at once using broadcasting (K_e_2 for type 2, K_e for type 1)
    K_temp_all = (new_g[:, np.newaxis, np.newaxis] ** p) * (
        new_g_2[:, np.newaxis, np.newaxis] ** p_2 * K_e_2_dense +
        (1 - new_g_2[:, np.newaxis, np.newaxis] ** p_2) * K_e_dense
    )

    # Free dense matrices immediately
    del K_e_dense, K_e_2_dense

    # Convert elements to numpy array for faster indexing
    elements = np.array(elements)

    # Create global DOF indices for all elements at once
    idx_i = np.array([[2*node, 2*node+1] for node in range(grid_size_x * grid_size_y)]).flatten()
    DOF = np.column_stack([
        np.column_stack((2*elements[:, 0], 2*elements[:, 0]+1)),
        np.column_stack((2*elements[:, 1], 2*elements[:, 1]+1)),
        np.column_stack((2*elements[:, 2], 2*elements[:, 2]+1)),
        np.column_stack((2*elements[:, 3], 2*elements[:, 3]+1))
    ])

    # Create row and column indices for the sparse matrix
    rows = np.repeat(DOF, nodes_per_element, axis=1)
    cols = np.tile(DOF, nodes_per_element)

    # Flatten K_temp_all to match the row/col indices
    data = K_temp_all.reshape(num_elements, -1)

    # Free K_temp_all immediately after reshaping
    del K_temp_all

    # Create flattened arrays for sparse matrix construction
    rows_flat = rows.flatten()
    cols_flat = cols.flatten()
    data_flat = data.flatten()

    # Free intermediate arrays
    del rows, cols, data, DOF

    # Create the sparse matrix directly
    K = sparse.csc_matrix(
        (data_flat, (rows_flat, cols_flat)),
        shape=(2 * grid_size_x * grid_size_y, 2 * grid_size_x * grid_size_y)
    )

    # Free flattened arrays after creating sparse matrix
    del rows_flat, cols_flat, data_flat

    return K


def multi_step_filter_element(g_test, r, Nel_x, Nel_y, Delta_x, beta):
    
    # First Helmholtz filter with radius r
    densities_1 = helmholtz_filter(g_test, r, Nel_x, Nel_y, Delta_x)
    
    
    # First projection step with ETA_1 = 0.5
    ETA_1 = 0.5
    densities_2 = (np.tanh(beta*ETA_1) + np.tanh(beta*(densities_1 - ETA_1))) / (np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1)))
    
    # densities_1 no longer needed after creating densities_2
    del densities_1
    
    # Second Helmholtz filter with radius 0.8*r
    r_2 = 0.8 * r
    densities_3 = helmholtz_filter(densities_2, r_2, Nel_x, Nel_y, Delta_x)
    
    # densities_2 no longer needed after creating densities_3
    del densities_2
    
    # Second projection step with ETA_2 = 0.05
    ETA_2 = 0.05
    final_densities = (np.tanh(beta*ETA_2) + np.tanh(beta*(densities_3 - ETA_2))) / (np.tanh(beta*ETA_2) + np.tanh(beta*(1 - ETA_2)))
    
    # densities_3 no longer needed after creating final_densities
    del densities_3
    
    return final_densities

# Example usage:
#r = 3.5 * Delta_x  # Base filter radius
g_test = np.full(Nel_x * Nel_y, 0.2)  # Test array

# Memory-optimized: Create a lazy multi-step sensitivity matrix class
class LazyMultiStepSensitivityMatrix:
    """Lazy multi-step sensitivity matrix that computes columns on-demand to save memory"""
    def __init__(self, g_test, r, Nel_x, Nel_y, Delta_x, beta):
        self.N = Nel_x * Nel_y
        self.Nel_x = Nel_x
        self.Nel_y = Nel_y
        self.shape = (self.N, self.N)  # For compatibility with array operations
        
        # Get intermediate values from forward pass
        densities_1 = helmholtz_filter(g_test, r, Nel_x, Nel_y, Delta_x)
        ETA_1 = 0.5
        beta_x_1 = beta * (densities_1 - ETA_1)
        densities_2 = (np.tanh(beta*ETA_1) + np.tanh(beta_x_1)) / (np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1)))
        
        r_2 = 0.8 * r
        densities_3 = helmholtz_filter(densities_2, r_2, Nel_x, Nel_y, Delta_x)
        
        ETA_2 = 0.05
        beta_x_2 = beta * (densities_3 - ETA_2)
        
        # Use pre-calculated Helmholtz operator matrices
        self.H1 = H_solid
        self.H2 = H_solid_2
        
        # Calculate projection derivatives
        # First projection
        cosh_term_1 = np.cosh(beta_x_1)
        tanh_term_1 = np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1))
        self.proj_derivative_1 = beta * (1.0 / (cosh_term_1 * cosh_term_1)) / tanh_term_1
        
        # Second projection
        cosh_term_2 = np.cosh(beta_x_2)
        tanh_term_2 = np.tanh(beta*ETA_2) + np.tanh(beta*(1 - ETA_2))
        self.proj_derivative_2 = beta * (1.0 / (cosh_term_2 * cosh_term_2)) / tanh_term_2
        
        # Create diagonal matrices from projection derivatives
        self.P1 = sparse.diags(self.proj_derivative_1)
        
        # Calculate mean preservation factors
        self.mean_preservation_1 = float(np.mean(densities_1) / np.mean(g_test))
        self.mean_preservation_2 = float(np.mean(densities_3) / np.mean(densities_2))
        
        # Free intermediate arrays immediately
        del densities_1, densities_2, densities_3, beta_x_1, beta_x_2, cosh_term_1, cosh_term_2, tanh_term_1, tanh_term_2
        
        # Pre-factorize for efficient column computation
        from scipy.sparse.linalg import splu
        self.H2_T_lu = splu(self.H2.T.tocsc())
        self.H1_T_lu = splu(self.H1.T.tocsc())
        
        # Cache for computed columns (optional, can be removed if memory is tight)
        self._column_cache = {}
    
    def get_column(self, j):
        """Get column j of the sensitivity matrix"""
        if j in self._column_cache:
            return self._column_cache[j]
        
        if self.proj_derivative_2[j] != 0:
            # Solve H2.T @ lambda3_col = proj_derivative_2[j] * e_j
            rhs_j = np.zeros(self.N, dtype=np.float64)
            rhs_j[j] = 1.0
            lambda3_col = self.proj_derivative_2[j] * self.H2_T_lu.solve(rhs_j)
            
            # Apply P1
            lambda2_col = self.P1.dot(lambda3_col)
            del lambda3_col  # Free immediately
            
            # Solve H1.T @ sensitivity_col = lambda2_col
            col = self.H1_T_lu.solve(lambda2_col)
            del lambda2_col  # Free immediately
        else:
            col = np.zeros(self.N, dtype=np.float64)
        
        col = self.mean_preservation_1 * self.mean_preservation_2 * col
        
        # Optionally cache (comment out if memory is very tight)
        # self._column_cache[j] = col
        
        return col
    
    def get_full_matrix(self):
        """Get full matrix (only use if absolutely necessary)"""
        sensitivity_matrix = np.zeros((self.N, self.N), dtype=np.float64)
        for j in range(self.N):
            sensitivity_matrix[:, j] = self.get_column(j)
        return sensitivity_matrix
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'H2_T_lu'):
            del self.H2_T_lu
        if hasattr(self, 'H1_T_lu'):
            del self.H1_T_lu
        if hasattr(self, 'P1'):
            del self.P1
        if hasattr(self, '_column_cache'):
            del self._column_cache

def multi_step_filter_sensitivity(g_test, r, Nel_x, Nel_y, Delta_x, beta):
    """Legacy function - returns LazyMultiStepSensitivityMatrix for memory efficiency"""
    return LazyMultiStepSensitivityMatrix(g_test, r, Nel_x, Nel_y, Delta_x, beta)


## second version with nodal densities
def helmholtz_nodal_filter(g_test, r, Nel_x, Nel_y, Delta_x):
    
    # Ensure input is float64
    g_test = np.asarray(g_test, dtype=np.float64)   # misschien weer verwijderen
    Nnodes_x = Nel_x + 1
    Nnodes_y = Nel_y + 1
    N = Nnodes_x * Nnodes_y
    
    # Use pre-calculated nodal Helmholtz matrix H_nodal_solid
    H = H_nodal_solid
    
    # First, map element densities to initial nodal values
    g_nodal = np.zeros(N)
    for i in range(Nnodes_y):
        for j in range(Nnodes_x):
            node_idx = i * Nnodes_x + j
            contributing_elements = []
            
            # Check all possible surrounding elements
            if i > 0 and j > 0:  # Top-left element
                contributing_elements.append((i-1) * Nel_x + (j-1))
            if i > 0 and j < Nel_x:  # Top-right element
                contributing_elements.append((i-1) * Nel_x + j)
            if i < Nel_y and j > 0:  # Bottom-left element
                contributing_elements.append(i * Nel_x + (j-1))
            if i < Nel_y and j < Nel_x:  # Bottom-right element
                contributing_elements.append(i * Nel_x + j)
            
            # Average the densities of contributing elements
            if contributing_elements:
                g_nodal[node_idx] = np.mean([g_test[idx] for idx in contributing_elements])
    
    # Store original mean for normalization
    original_mean = np.mean(g_nodal)
    
    # Solve the system
    filtered_nodal = spsolve(H, g_nodal).astype(np.float64)   # misschien .astype weer verwijderen
    
    # g_nodal no longer needed after solving
    del g_nodal
    
    # Normalize to preserve mean density
    filtered_nodal = filtered_nodal * (original_mean / np.mean(filtered_nodal))
    
    return filtered_nodal

# Example usage:
#r = 3.5 * Delta_x  # Filter radius


### sensitivity for helmholz

def helmholtz_nodal_sensitivity(g, r, Nel_x, Nel_y, Delta_x, node_index, element_index):
    
    Nnodes_x = Nel_x + 1
    Nnodes_y = Nel_y + 1
    N = Nnodes_x * Nnodes_y
    
    # Use pre-calculated nodal Helmholtz matrix H_nodal_solid
    H = H_nodal_solid
    
    # Create right-hand side sensitivity vector
    rhs = np.zeros(N)
    
    # Find which nodes are affected by this element
    element_row = element_index // Nel_x
    element_col = element_index % Nel_x
    
    # Count how many elements contribute to each affected node
    def count_contributing_elements(node_row, node_col):
        count = 0
        if node_row > 0 and node_col > 0:
            count += 1
        if node_row > 0 and node_col < Nel_x:
            count += 1
        if node_row < Nel_y and node_col > 0:
            count += 1
        if node_row < Nel_y and node_col < Nel_x:
            count += 1
        return max(count, 1)  # Ensure we don't divide by zero
    
    # The four nodes of this element and their contributions
    nodes_and_weights = [
        (element_row, element_col),                    # Top-left
        (element_row, element_col + 1),                # Top-right
        (element_row + 1, element_col),                # Bottom-left
        (element_row + 1, element_col + 1)             # Bottom-right
    ]
    
    # Set the contributions in rhs vector
    for node_row, node_col in nodes_and_weights:
        node_idx = node_row * Nnodes_x + node_col
        num_contributors = count_contributing_elements(node_row, node_col)
        rhs[node_idx] = 1.0 / num_contributors
    
    # Solve the sensitivity system
    sensitivity = spsolve(H, rhs)
    
    # Return sensitivity for the requested node
    return sensitivity[node_index]




def helmholtz_nodal_to_nodal_filter(nodal_densities, r, Nel_x, Nel_y, Delta_x):
    
    Nnodes_x = Nel_x + 1
    Nnodes_y = Nel_y + 1
    N = Nnodes_x * Nnodes_y
    
    # Use pre-calculated nodal Helmholtz matrix H_nodal_solid_2 (for 0.8*r)
    H = H_nodal_solid_2
    
    # Store original mean for normalization
    original_mean = np.mean(nodal_densities)
    
    # Solve the system
    filtered_nodal = spsolve(H, nodal_densities)
    
    # Normalize to preserve mean density
    filtered_nodal = filtered_nodal * (original_mean / np.mean(filtered_nodal))
    
    return filtered_nodal

def multi_step_filter_nodal(g_test, r, Nel_x, Nel_y, Delta_x, beta, ETA_1):
    
    # Ensure input is float64
    g_test = np.asarray(g_test, dtype=np.float64)   # misschien weer verwijderen
    # First Helmholtz filter (element to nodal)
    nodal_densities = helmholtz_nodal_filter(g_test, r, Nel_x, Nel_y, Delta_x).astype(np.float64)  # misschien .astype weer verwijderen
    
    
    # Projection step
    nodal_densities_step = (np.tanh(beta*ETA_1) + np.tanh(beta*(nodal_densities - ETA_1))) / (np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1))).astype(np.float64)  # misschien .astype weer verwijderen
    
    
    # Second Helmholtz filter (nodal to nodal)
    r_2 = 0.8 * r   
    mu_wiggle = helmholtz_nodal_to_nodal_filter(nodal_densities_step, r_2, Nel_x, Nel_y, Delta_x).astype(np.float64)  # misschien .astype weer verwijderen
    
    
    return mu_wiggle, nodal_densities, nodal_densities_step

def multi_step_sensitivity_nodal(g_test, r, Nel_x, Nel_y, Delta_x, beta, ETA_1):
    
    Nnodes_x = Nel_x + 1
    Nnodes_y = Nel_y + 1
    N = Nnodes_x * Nnodes_y
    num_elements = Nel_x * Nel_y
    
    # Create element-to-nodal mapping matrix
    # This matrix maps element densities to initial nodal values
    E_to_N = sparse.lil_matrix((N, num_elements))
    
    for i in range(Nnodes_y):
        for j in range(Nnodes_x):
            node_idx = i * Nnodes_x + j
            contributing_elements = []
            
            # Check all possible surrounding elements
            if i > 0 and j > 0:  # Top-left element
                contributing_elements.append(((i-1) * Nel_x + (j-1), 1.0))
            if i > 0 and j < Nel_x:  # Top-right element
                contributing_elements.append(((i-1) * Nel_x + j, 1.0))
            if i < Nel_y and j > 0:  # Bottom-left element
                contributing_elements.append((i * Nel_x + (j-1), 1.0))
            if i < Nel_y and j < Nel_x:  # Bottom-right element
                contributing_elements.append((i * Nel_x + j, 1.0))
            
            # Set the mapping weights (average of contributing elements)
            if contributing_elements:
                num_contributing = len(contributing_elements)
                for element_idx, weight in contributing_elements:
                    E_to_N[node_idx, element_idx] = weight / num_contributing
    
    E_to_N = E_to_N.tocsr()
    
    # Get intermediate values from forward pass
    nodal_densities = helmholtz_nodal_filter(g_test, r, Nel_x, Nel_y, Delta_x)
    beta_x = beta * (nodal_densities - ETA_1)
    nodal_densities_step = (np.tanh(beta*ETA_1) + np.tanh(beta_x)) / (np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1)))
    
    # Use pre-calculated nodal Helmholtz operator matrices
    H1 = H_nodal_solid
    H2 = H_nodal_solid_2
    
    # Calculate projection derivative
    cosh_term = np.cosh(beta_x)
    tanh_term = np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1))
    proj_derivative = beta * (1.0 / (cosh_term * cosh_term)) / tanh_term
    
    # Create diagonal matrix from projection derivative
    P = sparse.diags(proj_derivative)
    
    # Calculate mean preservation factors
    mean_preservation_1 = float(np.mean(nodal_densities) / np.mean(g_test))
    mean_preservation_2 = float(np.mean(nodal_densities_step) / np.mean(nodal_densities))
    
    # Clean up intermediate arrays after mean preservation calculation
    del nodal_densities, nodal_densities_step, beta_x, cosh_term, tanh_term
    
    # Solve the complete sensitivity system backwards
    # Start from the end and work backwards through the chain
    # Memory-optimized: Compute sensitivity_matrix directly using sparse operations and chunks
    # Instead of storing full nodal_sensitivity (N, N) = 2.61 GiB, compute in chunks
    Nnodes_x = Nel_x + 1
    Nnodes_y = Nel_y + 1
    N = Nnodes_x * Nnodes_y
    num_elements = Nel_x * Nel_y
    
    # Allocate sensitivity_matrix directly (N, num_elements)
    sensitivity_matrix = np.zeros((N, num_elements), dtype=np.float64)
    
    # Use LU decompositions for efficiency
    from scipy.sparse.linalg import splu
    H2_T_lu = splu(H2.T.tocsc())
    H1_T_lu = splu(H1.T.tocsc())
    
    # Keep E_to_N as sparse for efficient operations
    # Process in chunks to balance memory and performance
    chunk_size = 100  # Process 100 columns at a time
    nodal_sensitivity_chunk = np.zeros((N, chunk_size), dtype=np.float64)
    
    # Process columns in chunks
    for chunk_start in range(0, N, chunk_size):
        chunk_end = builtins.min(chunk_start + chunk_size, N)
        chunk_cols = chunk_end - chunk_start
        
        # Compute chunk of nodal_sensitivity
        for j_idx, j in enumerate(range(chunk_start, chunk_end)):
            if proj_derivative[j] != 0:
                # Solve H2.T @ lambda2_col = proj_derivative[j] * e_j
                rhs_j = np.zeros(N, dtype=np.float64)
                rhs_j[j] = 1.0
                lambda2_col = proj_derivative[j] * H2_T_lu.solve(rhs_j)
                
                # Apply P
                lambda1_col = P.dot(lambda2_col)
                
                # Solve H1.T @ nodal_sens_col = lambda1_col
                nodal_sensitivity_chunk[:, j_idx] = H1_T_lu.solve(lambda1_col)
        
        # Apply mean preservation to chunk
        nodal_sensitivity_chunk[:, :chunk_cols] *= (mean_preservation_1 * mean_preservation_2)
        
        # Multiply chunk with corresponding rows of E_to_N using sparse operations
        # sensitivity_matrix += nodal_sensitivity_chunk @ E_to_N[chunk_start:chunk_end, :]
        # Use sparse matrix multiplication for efficiency
        E_to_N_chunk = E_to_N[chunk_start:chunk_end, :]
        if hasattr(E_to_N_chunk, 'toarray'):
            # Convert to dense only for this chunk
            E_to_N_chunk_dense = E_to_N_chunk.toarray()
            sensitivity_matrix += nodal_sensitivity_chunk[:, :chunk_cols] @ E_to_N_chunk_dense
            del E_to_N_chunk_dense
        else:
            sensitivity_matrix += nodal_sensitivity_chunk[:, :chunk_cols] @ E_to_N_chunk
        
        # Clear chunk for next iteration
        nodal_sensitivity_chunk[:, :chunk_cols] = 0.0
    
    # Keep as float64 for precision (user requirement)
    # sensitivity_matrix is already float64 from np.zeros
    
    # Clean up intermediate matrices
    del P, H2_T_lu, H1_T_lu, nodal_sensitivity_chunk, E_to_N
    
    return sensitivity_matrix


### stiffness matrix part
ZD = []
for i in range(grid_size_y):
    ZD.append(grid_size_x * 2 * i)
    ZD.append(grid_size_x * 2 * i + 1)

Reduced_stiffness_indices = []

for i in range(len(2 * nodes)):
    if i in ZD:
        pass
    else:
        Reduced_stiffness_indices.append(i)

# Make stiffness matrix with 0's



# shapefunctions - optimized: direct numerical computation (no sympy)
# For square elements, the Jacobian is constant: J = [[Delta_x/2, 0], [0, Delta_x/2]]
# So J_inv = [[2/Delta_x, 0], [0, 2/Delta_x]]
Delta_x = length_x / Nel_x
J_inv_scale = 2.0 / Delta_x  # Constant for square elements

# Shape function derivatives in natural coordinates (analytical expressions)
# dN/dX and dN/dY for each shape function
def dN_dX_natural(X, Y):
    """Shape function derivatives w.r.t. natural coordinate X"""
    return np.array([
        -0.25 * (1 - Y),  # dN1/dX
         0.25 * (1 - Y),  # dN2/dX
         0.25 * (1 + Y),  # dN3/dX
        -0.25 * (1 + Y)   # dN4/dX
    ])

def dN_dY_natural(X, Y):
    """Shape function derivatives w.r.t. natural coordinate Y"""
    return np.array([
        -0.25 * (1 - X),  # dN1/dY
        -0.25 * (1 + X),  # dN2/dY
         0.25 * (1 + X),  # dN3/dY
         0.25 * (1 - X)   # dN4/dY
    ])

# For square elements, transformation from natural to physical coordinates is simple:
# dN/dx = (2/Delta_x) * dN/dX, dN/dy = (2/Delta_x) * dN/dY
# Pre-compute shape function derivatives at Gauss points
GP = np.array([[-1 / math.sqrt(3), -1 / math.sqrt(3)], [1 / math.sqrt(3), -1 / math.sqrt(3)],
               [1 / math.sqrt(3), 1 / math.sqrt(3)], [-1 / math.sqrt(3), 1 / math.sqrt(3)]])

# Pre-compute all shape function derivatives at all Gauss points
dN_dx_GP = np.zeros((4, 4))  # (Gauss_point, shape_function)
dN_dy_GP = np.zeros((4, 4))  # (Gauss_point, shape_function)

for Q in range(4):
    dN_dX = dN_dX_natural(GP[Q][0], GP[Q][1])
    dN_dY = dN_dY_natural(GP[Q][0], GP[Q][1])
    # Transform to physical coordinates (for square elements)
    dN_dx_GP[Q, :] = J_inv_scale * dN_dX
    dN_dy_GP[Q, :] = J_inv_scale * dN_dY

# Gauss Points (quad) - already defined above
J_det = Delta_x ** 2 / 4

# material properties of E_1
E_1 = 10 ** 6  # Young's modulus
v_1 = 0.3  # Poisson Ratio

# material properties
matDT = E_1/(1 - v_1**2) * np.array([[1, v_1, 0], [v_1, 1, 0], [0, 0, 1/2*(1-v_1)]])

K_e = np.zeros((8, 8), order='F')
B = np.zeros((3, 8), order='F')
for Q in range(4):
    # Use pre-computed shape function derivatives
    dN_1_X_GP = dN_dx_GP[Q, 0]
    dN_1_Y_GP = dN_dy_GP[Q, 0]
    dN_2_X_GP = dN_dx_GP[Q, 1]
    dN_2_Y_GP = dN_dy_GP[Q, 1]
    dN_3_X_GP = dN_dx_GP[Q, 2]
    dN_3_Y_GP = dN_dy_GP[Q, 2]
    dN_4_X_GP = dN_dx_GP[Q, 3]
    dN_4_Y_GP = dN_dy_GP[Q, 3]

    B_1 = np.array([[dN_1_X_GP, 0, dN_2_X_GP, 0, dN_3_X_GP, 0, dN_4_X_GP, 0],
                    [0, dN_1_Y_GP, 0, dN_2_Y_GP, 0, dN_3_Y_GP, 0, dN_4_Y_GP],
                    [dN_1_Y_GP, dN_1_X_GP, dN_2_Y_GP, dN_2_X_GP, dN_3_Y_GP, dN_3_X_GP, dN_4_Y_GP, dN_4_X_GP]])
    B_2 = np.transpose(B_1)

    # Weight for squares (W) = 1
    W = 1

    # the local stiffness matrix every element uses (since they are all squares of equal size)
    K_e = K_e + J_det * np.dot(np.dot(B_2, matDT), B_1) * W
    B = B + B_1

K_e = np.float32(K_e)
K_e_sparse = sparse.csr_matrix(K_e)  # Convert to sparse format
K_e_backup = K_e

#material properties of E_2
E_2 = 2*10 ** 6  # Young's modulus
v_2 = 0.3  # Poisson Ratio

# material properties
matDT_2 = E_2/(1 - v_2**2) * np.array([[1, v_2, 0], [v_2, 1, 0], [0, 0, 1/2*(1-v_2)]])

K_e_2 = np.zeros((8, 8), order='F')

for Q in range(4):
    # Use pre-computed shape function derivatives
    dN_1_X_GP = dN_dx_GP[Q, 0]
    dN_1_Y_GP = dN_dy_GP[Q, 0]
    dN_2_X_GP = dN_dx_GP[Q, 1]
    dN_2_Y_GP = dN_dy_GP[Q, 1]
    dN_3_X_GP = dN_dx_GP[Q, 2]
    dN_3_Y_GP = dN_dy_GP[Q, 2]
    dN_4_X_GP = dN_dx_GP[Q, 3]
    dN_4_Y_GP = dN_dy_GP[Q, 3]

    B_1 = np.array([[dN_1_X_GP, 0, dN_2_X_GP, 0, dN_3_X_GP, 0, dN_4_X_GP, 0],
                    [0, dN_1_Y_GP, 0, dN_2_Y_GP, 0, dN_3_Y_GP, 0, dN_4_Y_GP],
                    [dN_1_Y_GP, dN_1_X_GP, dN_2_Y_GP, dN_2_X_GP, dN_3_Y_GP, dN_3_X_GP, dN_4_Y_GP, dN_4_X_GP]])
    B_2 = np.transpose(B_1)

    # Weight for squares (W) = 1
    W = 1

    # the local stiffness matrix every element uses (since they are all squares of equal size)
    K_e_2 = K_e_2 + J_det * np.dot(np.dot(B_2, matDT_2), B_1) * W

K_e_2 = np.float32(K_e_2)
K_e_2_sparse = sparse.csr_matrix(K_e_2)  # Convert to sparse format
K_e_2_backup = K_e_2
# Make a Back Up of the original full stiffness matrix for filling in the derivatives of the local matrices for the grad functions
K = np.zeros((2 * grid_size_x * grid_size_y, 2 * grid_size_x * grid_size_y), dtype='float')
K_der_BU = K  # derivative Back up

num_elements = len(elements)


"""
def continuation_beta(current_eval, max_eval):
    #Shared projection sharpness beta for objective and all active constraints.
    t = current_eval / max_eval
    if t < 0.6:  # First 60% of optimization
        beta = 0.01 + t * 8
    else:  # Last 40% of optimization
        remaining_t = (t - 0.6) / 0.4  # Normalize to 0-1
        beta = 8 + (20 - 8) * (1 - np.exp(-3 * remaining_t))
    return beta
""" 
def continuation_beta(current_eval, max_eval):
    """Shared projection sharpness beta for objective and all active constraints."""
    t = current_eval / max_eval
    if t < 0.6:  # First 60% of optimization
        beta = 0.01 + t * 8
    else:  # Last 40% of optimization
        remaining_t = (t - 0.6) / 0.4  # Normalize to 0-1
        beta = 0.01 + 0.6 * 8 + remaining_t*12
    return beta


# optimization function
def myfunc(g, grad):
    beta = continuation_beta(opt.get_numevals(), max_eval)

    # Add this in your myfunc function after setting beta:
    if opt.get_numevals() % 10 == 0:  # Print every 10 iterations
        print(f"Iteration {opt.get_numevals()}: beta = {beta}")
     # Print design variables every few iterations
    if opt.get_numevals() % 5 == 0:  # Print every 5 iterations
        print(f"\nIteration {opt.get_numevals()}:")
        print(f"  Raw g[0:5]: {g[0:5]}")
        print(f"  Raw g[mid_index:mid_index+5]: {g[mid_index:mid_index+5]}")
        print(f"  Beta: {beta}")
    
    g_solid = g[:mid_index].copy()  # Make explicit copy
    g_type = g[mid_index:].copy()   # Make explicit copy

    new_g = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
    new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)

    min_density = 1e-9  # Small but non-zero
    new_g = np.maximum(new_g, min_density)
    #new_g_2 = np.maximum(new_g_2, min_density)

    """
    # Print filtered values
    if opt.get_numevals() % 5 == 0:
        print(f"  Filtered new_g[0:5]: {new_g[0:5]}")
        print(f"  Filtered new_g_2[0:5]: {new_g_2[0:5]}")
    """
    # (K_e, K_e_2) order: myfunc runs before the later StiffnessMatrix redefinition; this keeps the
    # original v1 blend for the optimizer. The stress block passes swapped args for v2 consistency.
    K = StiffnessMatrix(new_g, new_g_2, elements, K_e_sparse, K_e_2_sparse, p, p_2, grid_size_x, grid_size_y)
    sK_reduced = sparse.csr_matrix(K[Reduced_stiffness_indices][:, Reduced_stiffness_indices].astype(float))

    Force = build_force_vector(dtype=np.float32)
    Force_reduced = Force[Reduced_stiffness_indices].astype(K.dtype)

    U = spsolve(sK_reduced, Force_reduced)
    #print(f"Min density: {np.min(new_g):.6f}, Max density: {np.max(new_g):.6f}")
    
    for i in range(grid_size_y):
        # insert x-displacement (0)
        U = np.insert(U, 2 * grid_size_x * i, 0)
        
        # insert y-displacement (0)
        U = np.insert(U, 2 * grid_size_x * i + 1, 0)

    # Calculate compliance (objective function value)
    compliance = float(np.dot(Force, U))
    
    # Record compliance history
    compliance_history.append(compliance)
    
    if grad.size > 0:
        if Nel_x == 48 or Nel_x == 96:
            sensi_solid, sensi_type = Compliance_sensitivity_fast(U, K_e_sparse, K_e_2_sparse, new_g, new_g_2,
                                                           g_solid, g_type, beta, ETA_1)
            grad[:] = np.concatenate([sensi_solid, sensi_type])
        else:
            der_solid = step_filter_sensitivity(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
            der_type = step_filter_sensitivity(g_type, r, Nel_x, Nel_y, Delta_x, beta)
            sensi_solid, sensi_type = Compliance_sensitivity(
                U, K_e_sparse, K_e_2_sparse, new_g, new_g_2, der_solid, der_type)
            grad[:] = np.concatenate([sensi_solid, sensi_type])
    return compliance

# Constraint 1: Limit solid elements to 1/2 of total elements
def myconstraint7(g, grad):
    beta = continuation_beta(opt.get_numevals(), max_eval)

    if grad.size > 0:
        
        # Split the design variables
        g_solid = g[:mid_index].copy()
        g_type = g[mid_index:].copy()
        
        # Apply the same filtering as in the objective function
        new_g = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
        min_density = 1e-9  # Small but non-zero
        new_g = np.maximum(new_g, min_density)
        
        # Fast matvec: one solve instead of N
        ones_vec = np.ones(len(g_solid))
        grad_solid = step_filter_sensitivity(g_solid, r, Nel_x, Nel_y, Delta_x, beta, ones_vec)
        
        # Set gradients
        grad[:mid_index] = grad_solid
        grad[mid_index:] = 0.0  # No dependency on g_type for this constraint
    
    # Apply filtering to get actual densities
    g_solid = g[:mid_index].copy()
    new_g = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
    min_density = 1e-9  # Small but non-zero
    new_g = np.maximum(new_g, min_density)

    # Constraint: sum of solid densities - 0.5 * total_elements <= 0
    return np.sum(new_g) - 0.5 * len(elements)

constraint8_call_count = 0
"""
# Constraint 2: Limit material type 2 elements to 1/4 of total elements  
def myconstraint8(g, grad):
    global constraint8_call_count
    constraint8_call_count += 1
    
    if opt.get_numevals() <= 20:
            beta = 0.00001
    elif opt.get_numevals() <= 40:
        beta = 1
    elif opt.get_numevals() <= 60:
        beta = 3
    elif opt.get_numevals() <= 80:
        beta = 8
    elif opt.get_numevals() <= 100:
        beta = 20
    elif opt.get_numevals() <= 120:
        beta = 40
    else:
        beta = 50
    if grad.size > 0:
        
        # Split the design variables
        g_solid = g[:mid_index].copy()
        g_type = g[mid_index:].copy()
        
        # Apply filtering to both
        new_g = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
        new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)
        
        min_density = 1e-9  # Small but non-zero
        new_g = np.maximum(new_g, min_density)

        # Calculate sensitivities
        der_solid = step_filter_sensitivity(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
        der_type = step_filter_sensitivity(g_type, r, Nel_x, Nel_y, Delta_x, beta)
        
        # Gradient w.r.t. g_solid: d(constraint)/d(g_solid) = new_g_2 * sensitivity_solid
        grad_solid = np.sum(der_solid * new_g_2[:, np.newaxis], axis=0)
        
        # Gradient w.r.t. g_type: d(constraint)/d(g_type) = new_g * sensitivity_type  
        grad_type = np.sum(der_type * new_g[:, np.newaxis], axis=0)
        
        # Set gradients
        grad[:mid_index] = grad_solid
        grad[mid_index:] = grad_type
    
    # Apply filtering to get actual densities
    g_solid = g[:mid_index].copy()
    g_type = g[mid_index:].copy()
    new_g = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
    new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)
    
    min_density = 1e-9  # Small but non-zero
    new_g = np.maximum(new_g, min_density)
    # Constraint: sum of (solid * type2) - 0.25 * total_elements <= 0
    return np.sum(new_g * new_g_2) - 0.25 * len(elements)
"""

def myconstraint8(g, grad):
    beta = continuation_beta(opt.get_numevals(), max_eval)

    total_elements = len(elements)
    if grad.size > 0:
        # Split the design variables
        g_solid = g[:mid_index].copy()
        g_type = g[mid_index:].copy()
        
        # Apply filtering to both
        new_g = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
        new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)
        
        min_density = 1e-9  # Small but non-zero
        new_g = np.maximum(new_g, min_density)

        # OPTIMIZED: Use matvec instead of forming full sensitivity matrices
        # grad_solid = der_solid.T @ new_g_2 computed directly
        # grad_type = der_type.T @ new_g computed directly
        # Fast matvec: one solve per sensitivity
        grad_solid = step_filter_sensitivity(g_solid, r, Nel_x, Nel_y, Delta_x, beta, new_g_2)
        grad_type = step_filter_sensitivity(g_type, r, Nel_x, Nel_y, Delta_x, beta, new_g)
        
        # Set gradients
        grad[:mid_index] = grad_solid / total_elements
        grad[mid_index:] = grad_type / total_elements
    
    # Apply filtering to get actual densities
    g_solid = g[:mid_index].copy()
    g_type = g[mid_index:].copy()
    new_g = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
    new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)
    
    min_density = 1e-9  # Small but non-zero
    new_g = np.maximum(new_g, min_density)

    constraint_value = np.sum(new_g * new_g_2) - 0.25 * total_elements
    normalized_constraint = constraint_value / total_elements
    # Constraint: sum of (solid * type2) - 0.25 * total_elements <= 0
    #return np.sum(new_g * new_g_2) - 0.25 * len(elements)
    return normalized_constraint

"""
def myconstraint8(g, grad):
    current_eval = opt.get_numevals()
    # Linear interpolation from 0.01 to 20
    k = 6.0  # Higher k = stays low longer
    t = current_eval / max_eval
    
    if t < 0.6:  # First 60% of optimization
        # Linear growth from 0.01 to 8
        beta = 0.01 + t * 8
    else:  # Last 40% of optimization
        # Exponential growth from 8 to 20
        remaining_t = (t - 0.6) / 0.4  # Normalize to 0-1
        beta = 8 + (20 - 8) * (1 - np.exp(-3 * remaining_t))
    
    total_elements = len(elements)
    if grad.size > 0:
        # Split the design variables
        g_solid = g[:mid_index].copy()
        g_type = g[mid_index:].copy()
        
        # Apply filtering to both
        new_g = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
        new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)
        
        min_density = 1e-9  # Small but non-zero
        new_g = np.maximum(new_g, min_density)

        # Fast matvec: one solve per sensitivity
        # df/dg_solid = sensitivity_solid.T @ (p * new_g^(p-1) * new_g_2^p_2)
        vec_solid = p * (new_g ** (p-1)) * (new_g_2 ** p_2)
        grad_solid = step_filter_sensitivity(g_solid, r, Nel_x, Nel_y, Delta_x, beta, vec_solid)
        # df/dg_type = sensitivity_type.T @ (new_g^p * p_2 * new_g_2^(p_2-1))
        vec_type = (new_g ** p) * p_2 * (new_g_2 ** (p_2-1))
        grad_type = step_filter_sensitivity(g_type, r, Nel_x, Nel_y, Delta_x, beta, vec_type)
        
        # Set gradients
        grad[:mid_index] = grad_solid / total_elements
        grad[mid_index:] = grad_type / total_elements
    
    # Apply filtering to get actual densities
    g_solid = g[:mid_index].copy()
    g_type = g[mid_index:].copy()
    new_g = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
    new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)
    
    min_density = 1e-9  # Small but non-zero
    new_g = np.maximum(new_g, min_density)

    # CORRECTED: Apply penalty parameters to constraint value
    penalized_constraint = np.sum((new_g ** p) * (new_g_2 ** p_2)) - 0.25 * total_elements
    normalized_constraint = penalized_constraint / total_elements
    return normalized_constraint
"""


def myconstraint9(g, grad):
    """
    Constraint: sum of type2 elements - 0.25 * total_elements <= 0
    This directly limits material 2 to 25% of total elements
    """
    current_eval = opt.get_numevals()
    # Linear interpolation from 0.01 to 20
    k = 6.0  # Higher k = stays low longer
    t = current_eval / max_eval
    
    if t < 0.6:  # First 60% of optimization
        # Linear growth from 0.01 to 8
        beta = 0.01 + t * 8
    else:  # Last 40% of optimization
        # Exponential growth from 8 to 20
        remaining_t = (t - 0.6) / 0.4  # Normalize to 0-1
        beta = 8 + (20 - 8) * (1 - np.exp(-3 * remaining_t))
    
    total_elements = len(elements)
    if grad.size > 0:
        # Split the design variables
        g_solid = g[:mid_index].copy()
        g_type = g[mid_index:].copy()
        
        # Apply filtering to type variables only (we only need new_g_2)
        new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)
        
        # OPTIMIZED: Use matvec with ones vector
        # grad_type = der_type.T @ ones computed directly without forming matrix
        ones_vec = np.ones(len(new_g_2))
        grad_type = step_filter_sensitivity(g_type, r, Nel_x, Nel_y, Delta_x, beta, ones_vec)
        
        # Set gradients (only for type variables, solid variables get zero gradient)
        grad[:mid_index] = 0  # No gradient for solid variables
        grad[mid_index:] = grad_type / total_elements
    
    # Apply filtering to get actual densities
    g_type = g[mid_index:].copy()
    new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)
    
    # Constraint: sum of type2 elements - 0.25 * total_elements <= 0
    constraint_value = np.sum(new_g_2) - 0.25 * total_elements
    normalized_constraint = constraint_value / total_elements
    return normalized_constraint

def myconstraint10(g, grad):
    """
    Constraint: sum of type2 elements - 0.25 * total_elements <= 0
    This directly limits material 2 to 25% of total elements
    """
    current_eval = opt.get_numevals()
    # Linear interpolation from 0.01 to 20
    k = 6.0  # Higher k = stays low longer
    t = current_eval / max_eval
    
    if t < 0.6:  # First 60% of optimization
        # Linear growth from 0.01 to 8
        beta = 0.01 + t * 8
    else:  # Last 40% of optimization
        # Exponential growth from 8 to 20
        remaining_t = (t - 0.6) / 0.4  # Normalize to 0-1
        beta = 8 + (20 - 8) * (1 - np.exp(-3 * remaining_t))
    
    total_elements = len(elements)
    if grad.size > 0:
        # Split the design variables
        g_solid = g[:mid_index].copy()
        g_type = g[mid_index:].copy()
        
        # Apply filtering to type variables only (we only need new_g_2)
        new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)
        
        # OPTIMIZED: Use matvec with ones vector
        # grad_type = der_type.T @ ones computed directly without forming matrix
        ones_vec = np.ones(len(new_g_2))
        grad_type = step_filter_sensitivity(g_type, r, Nel_x, Nel_y, Delta_x, beta, ones_vec)
        
        # Set gradients (only for type variables, solid variables get zero gradient)
        grad[:mid_index] = 0  # No gradient for solid variables
        grad[mid_index:] = grad_type / total_elements
    
    # Apply filtering to get actual densities
    g_type = g[mid_index:].copy()
    new_g_2 = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)
    
    # Constraint: sum of type2 elements - 0.25 * total_elements <= 0
    constraint_value = np.sum(new_g_2) - 0.15 * total_elements
    normalized_constraint = constraint_value / total_elements
    return normalized_constraint

def mystressconstraint(g, beta):
    """Interface stress p-norm aggregate and design sensitivities (same pipeline as ISC15 stress block).

    Returns:
        g_sigma (scalar), dg_sigma_dx1 (ndarray), dg_sigma_dx2 (ndarray)

    Uses module-level matDT, matDT_2, B, K_e, etc. Local names must not shadow those globals
    (e.g. never assign to matDT inside this function).
    """
    # Local ndarray for connectivity; do not assign to `elements` (would shadow the global and break StiffnessMatrix).
    elements_arr = np.array(elements)

    # Interface stress pipeline (g_sigma aggregation)

    ### inputs

    # number of elements on x-axis



    # p-factor for getting rid of gray area's


    # analysis = "plain_stress"           #plain_strain or plain_stress





    #r = 3.5 * Delta_x  # set to 3.5 for finite difference checks in the functions below, since they all use 3.5
    #r_original = 3.5 * Delta_x  # make this match global r

    H_solid = create_helmholtz_matrix(r, Nel_x, Nel_y, Delta_x)
    H_solid_2 = create_helmholtz_matrix(0.8*r, Nel_x, Nel_y, Delta_x)
    H_nodal_solid = create_helmholtz_nodal_matrix(r, Nel_x, Nel_y, Delta_x)
    H_nodal_solid_2 = create_helmholtz_nodal_matrix(0.8*r, Nel_x, Nel_y, Delta_x)

    # (helmholtz_filter, step_filter_element, LazySensitivityMatrix, step_filter_sensitivity: use definitions from first part)


    #initialize 


    mid_index = len(g) // 2
    g_solid = g[:mid_index].copy()  # Make explicit copy
    g_type = g[mid_index:].copy()   # Make explicit copy

    #r = 3.5 * Delta_x  # Base filter radius  3.5
    g_sf = step_filter_element(g_solid, r, Nel_x, Nel_y, Delta_x, beta)  #solid filtered
    g_tf = step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta)   #type filtered
    g_tsf = multi_step_filter_element(g_type, r, Nel_x, Nel_y, Delta_x, beta) #type second filter
    # Global stiffness: pass (K_e_2, K_e) so effective blend is g_tf^p_2 * K_e + (1 - g_tf^p_2) * K_e_2,
    # matching K_e_stack / D_e / sensitivities (myfunc uses unswapped args for the v1 blend in optimization).

    #K = StiffnessMatrix(new_g, new_g_2, elements, K_e_2_sparse, K_e_sparse, p, p_2, grid_size_x, grid_size_y)
    K = StiffnessMatrix(g_sf, g_tf, elements, K_e_2_sparse, K_e_sparse, p, p_2, grid_size_x, grid_size_y)
    Force = build_force_vector(dtype=np.float64)

    Force_reduced = Force[Reduced_stiffness_indices].astype(K.dtype)

    # Force no longer needed after creating Force_reduced
    del Force

    # the same as using np.delete, but copy elements from K to K_reduced
    sK_reduced = sparse.csr_matrix(K[Reduced_stiffness_indices][:, Reduced_stiffness_indices].astype(float))

    # K is no longer needed after creating sK_reduced (it's very large)
    del K

    U = spsolve(sK_reduced, Force_reduced)
    U_solved = U

    # Force_reduced no longer needed after solving
    del Force_reduced


    for i in range(grid_size_y):
        # insert x-displacement (0)
        U = np.insert(U, 2 * grid_size_x * i, 0)
    
        # insert y-displacement (0)
        U = np.insert(U, 2 * grid_size_x * i + 1, 0)
    

    der_solid = step_filter_sensitivity(g_solid, r, Nel_x, Nel_y, Delta_x, beta)
    der_type = step_filter_sensitivity(g_type, r, Nel_x, Nel_y, Delta_x, beta)

    #sensitivities
    g_tsf_sens = multi_step_filter_sensitivity(g_type, r, Nel_x, Nel_y, Delta_x, beta)

    ### filtered variables on nodes
    mu_wiggle, nodal_densities, nodal_densities_step = multi_step_filter_nodal(g_type, r, Nel_x, Nel_y, Delta_x, beta, ETA_1)

    ### Chi_int
    X_int = g_sf * g_tsf * (1 - g_tf)

    # dChi/dg  solid
    # Memory-optimized: Compute dX_dg column-by-column to avoid storing full der_solid
    # dX_dg[i, j] = der_solid[i, j] * (g_tsf * (1 - g_tf))[i]
    g_tsf_1_minus_g_tf = (g_tsf * (1 - g_tf))  # (num_elements,)
    num_elements = len(g_tsf_1_minus_g_tf)
    dX_dg = np.zeros((num_elements, num_elements), dtype=np.float64)
    for j in range(num_elements):
        if isinstance(der_solid, LazySensitivityMatrix):
            dX_dg[:, j] = der_solid.get_column(j) * g_tsf_1_minus_g_tf
        else:
            dX_dg[:, j] = der_solid[:, j] * g_tsf_1_minus_g_tf
    del g_tsf_1_minus_g_tf

    # dChi/dg type
    # Don't create full array - will compute in chunks when needed to avoid memory allocation
    # Store intermediate values needed for chunked computation
    g_sf_1_minus_g_tf = g_sf * (1 - g_tf)  # Precompute for efficiency
    g_sf_g_tsf = g_sf * g_tsf  # Precompute for efficiency
    # Keep g_tsf_sens for now - will use it in chunks later and delete after


    ### sigma_e 
    num_elements = len(elements_arr)

    # Step 1: Compute D_e for all elements
    g_tf_p2 = g_tf ** p_2  # (1152,)
    D_e = g_tf_p2 * matDT[:, :, np.newaxis] + (1 - g_tf_p2) * matDT_2[:, :, np.newaxis]  # (3, 3, 1152)

    # g_tf_p2 no longer needed after creating D_e (will be recomputed later if needed)
    del g_tf_p2

    # Vectorized version - replaces lines 955-959
    # Create all EL indices at once for all elements
    # Shape: (num_elements, 4) -> (num_elements, 8)
    EL_all = np.column_stack([
        2 * elements_arr[:, 0],           # 2*ele[0]
        2 * elements_arr[:, 0] + 1,       # 2*ele[0] + 1
        2 * elements_arr[:, 1],           # 2*ele[1]
        2 * elements_arr[:, 1] + 1,       # 2*ele[1] + 1
        2 * elements_arr[:, 2],           # 2*ele[2]
        2 * elements_arr[:, 2] + 1,       # 2*ele[2] + 1
        2 * elements_arr[:, 3],           # 2*ele[3]
        2 * elements_arr[:, 3] + 1        # 2*ele[3] + 1
    ])  # shape: (num_elements, 8)

    # Extract all displacement values at once using advanced indexing
    # EL_all[i, :] gives the 8 DOF indices for element i
    # U[EL_all] gives shape (num_elements, 8)
    # Transpose to get (8, num_elements) to match original shape
    u_vecs = U[EL_all].T  # shape: (8, num_elements)

    Bu = np.dot(B, u_vecs)  # shape: (3, 1152)

    D_e = np.asarray(D_e, dtype=np.float64)
    Bu = np.asarray(Bu, dtype=np.float64)
    # Step 3: D_e[:,:,i] @ Bu[:,i] for all i
    D_e_Bu = np.einsum('ijk,jk->ik', D_e, Bu)  # shape: (3, 1152)

    # Bu no longer needed after creating D_e_Bu
    del Bu

    sigma_e = (g_sf**p)[:, np.newaxis] * D_e_Bu.T  # (1152, 3) -> transpose to (3, 1152)
    sigma_e = sigma_e.T  # (3, 1152) - final shape

    # Note: u_vecs and EL_all are kept for later use (needed around line 1204)
    # They will be freed after they're used

    # Step 4: Compute the scalar part - memory optimized
    # Don't create full first_term array - will compute in chunks when needed
    # Store intermediate values needed for chunked computation (keep as float64 to avoid dtype promotion issues)
    g_sf_pow_p_minus_1 = p * g_sf**(p-1)  # (num_elements,) - keep as float64
    D_e_Bu_T = D_e_Bu.T  # (num_elements, 3) - keep as float64
    # Keep der_solid for chunked computation later

    #second term (dMatdx) = 0
    second_term = 0   #matDT part not dependent on the solid/void variables

    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu
    num_dof = 2 * grid_size_x * grid_size_y
    num_vars = num_elements  # 1152


    # Memory-optimized: Compute K_temp on-the-fly instead of storing full tensor
    # This avoids creating the huge (1152, 1152, 8, 8) array
    K_e_arr = K_e.toarray() if hasattr(K_e, 'toarray') else K_e
    K_e_2_arr = K_e_2.toarray() if hasattr(K_e_2, 'toarray') else K_e_2

    # Pre-compute reusable parts (keep as float64 for computation)
    g_sf_pow = g_sf ** (p-1)  # (num_elements,)
    g_tf_pow = g_tf ** p_2  # (num_elements,)
    K_e_arr = K_e_arr.astype(np.float64)
    K_e_2_arr = K_e_2_arr.astype(np.float64)

    # Pre-compute K_e_stack once (same for all variables)
    K_e_stack = (g_tf_pow[:, None, None] * K_e_arr[None, :, :] + 
                 (1 - g_tf_pow[:, None, None]) * K_e_2_arr[None, :, :])  # (num_elements, 8, 8)

    # Free intermediate arrays
    del g_tf_pow

    DOF = np.column_stack([
        np.column_stack((2*elements_arr[:, 0], 2*elements_arr[:, 0]+1)),
        np.column_stack((2*elements_arr[:, 1], 2*elements_arr[:, 1]+1)),
        np.column_stack((2*elements_arr[:, 2], 2*elements_arr[:, 2]+1)),
        np.column_stack((2*elements_arr[:, 3], 2*elements_arr[:, 3]+1))
    ])  # shape: (num_elements, 8)

    row_idx = np.repeat(DOF, 8, axis=1).reshape(num_elements, 8, 8)
    col_idx = np.tile(DOF, (1, 8)).reshape(num_elements, 8, 8)

    # Memory-optimized: Compute rhs_matrix columns on-the-fly and solve immediately
    # This avoids storing the full (37248, 18432) rhs_matrix array (5.12 GiB)

    # Pre-compute reduced DOF mapping  
    Reduced_stiffness_indices_arr = np.array(Reduced_stiffness_indices, dtype=np.int32)
    num_reduced = len(Reduced_stiffness_indices)
    reduced_map = np.full(num_dof, -1, dtype=np.int32)
    reduced_map[Reduced_stiffness_indices_arr] = np.arange(num_reduced)

    # Pre-compute row and column indices in reduced space
    row_idx_flat = row_idx.reshape(-1)  # (num_elements * 64,)
    col_idx_flat = col_idx.reshape(-1)  # (num_elements * 64,)

    # Filter to reduced DOF space once
    mask_base = (reduced_map[row_idx_flat] >= 0) & (reduced_map[col_idx_flat] >= 0)
    row_idx_reduced_base = reduced_map[row_idx_flat[mask_base]]
    col_idx_reduced_base = reduced_map[col_idx_flat[mask_base]]

    # Pre-factorize the system matrix for efficient solving
    lu = splu(sK_reduced.tocsc())

    # Allocate dUdx_full directly (keep as float64 for precision)
    dUdx_full = np.zeros((num_dof, num_vars), dtype=np.float64)

    # Process each variable - compute rhs column, solve, and store directly
    for j in range(num_vars):
        # Compute factor for this variable only (keep as float64)
        # Get column j of der_solid (compute on-demand if lazy)
        if isinstance(der_solid, LazySensitivityMatrix):
            der_solid_col_j = der_solid.get_column(j)  # (num_elements,)
        else:
            der_solid_col_j = der_solid[:, j]  # (num_elements,)
        factor_j = p * g_sf_pow * der_solid_col_j  # (num_elements,)
    
        # Compute K_temp for this variable only: (num_elements, 8, 8)
        K_temp_j = factor_j[:, None, None] * K_e_stack  # (num_elements, 8, 8)
    
        # Get data for this variable
        data_j = K_temp_j.reshape(-1)[mask_base]
    
        # Free K_temp_j immediately
        del K_temp_j
    
        # Build reduced sparse matrix
        K_der_reduced = coo_matrix(
            (data_j, (row_idx_reduced_base, col_idx_reduced_base)),
            shape=(num_reduced, num_reduced)
        ).tocsc()
    
        # Compute RHS column on-the-fly (don't store in full rhs_matrix)
        rhs_column = -K_der_reduced @ U_solved  # (num_reduced,)
    
        # Free intermediate
        del data_j, K_der_reduced
    
        # Solve immediately and store directly in dUdx_full
        dUdx_reduced_col = lu.solve(rhs_column)  # (num_reduced,)
        dUdx_full[Reduced_stiffness_indices_arr, j] = dUdx_reduced_col.astype(np.float64)
    
        # Free rhs column immediately
        del rhs_column, dUdx_reduced_col

    # Free memory
    del g_sf_pow, K_e_stack, K_e_arr, K_e_2_arr, factor_j
    # Free U after dUdx_full is computed (U_solved and lu still needed for dUdx_full_2)
    # U is no longer needed since we have dUdx_full
    del U
    gc.collect()  # Force garbage collection after freeing large arrays
    # Note: dUdx_full is kept for later use
    # Note: U_solved, sK_reduced, lu are kept for computing dUdx_full_2

    # make third term a 1152x1152x3x1 structure
    num_elements = len(elements_arr)

    num_vars = num_elements  # 1152

    # Prepare all EL indices for all elements
    EL_all = np.array([
        [2 * ele[0], 2 * ele[0] + 1, 2 * ele[1], 2 * ele[1] + 1,
         2 * ele[2], 2 * ele[2] + 1, 2 * ele[3], 2 * ele[3] + 1]
        for ele in elements_arr
    ])  # shape: (num_elements, 8)

    # Prepare g_sf^p for all elements
    g_sf_pow = g_sf ** p  # shape: (num_elements,)

    # Prepare B (shape: (3, 8)), constant for all elements

    # Prepare dUdx_full (shape: (num_dof, num_vars))
    # Memory-optimized: Compute Bu element-by-element to avoid creating huge (num_elements, 8, num_vars) array
    # Instead of: dudx_all = dUdx_full[EL_all, :] which would be (18432, 8, 18432) = 10.1 GiB
    # We compute B @ dUdx_full[EL_all[i], :] for each element i directly

    B_numeric = np.array(B, dtype=np.float64)  # (3, 8)
    # Memory-optimized: Don't store full third_term array (7.59 GiB)
    # Instead, compute third_term in chunks when needed later
    # Keep the components needed to compute third_term on-the-fly:
    # - dUdx_full, EL_all, B_numeric, D_e, g_sf_pow
    # third_term will be computed in chunks in the later loop

    ## Compute dsigma_e_dx1 in chunks to avoid storing full (num_elements, num_elements, 3) array
    ## Instead of: dsigma_e_dx1 = first_term + third_term
    ## We'll compute chunks and use them immediately in the einsum later
    ## For now, keep third_term and the components needed to compute first_term chunks
    # Note: dsigma_e_dx1 will be computed in chunks when needed for the einsum operation
    # Keep EL_all for computing third_term chunks later (it's needed for dUdx_full[EL_all[i], :])
    # Free some intermediate arrays that are no longer needed
    del D_e_Bu
    # Free first D_e after it's been used (it will be recomputed later)
    # Note: D_e is recomputed at line 1187, so the first one can be freed
    del D_e
    gc.collect()  # Force garbage collection

    #now for the second part -> var E_1 or E_2
    ### vectorized version (first_term = 0 for this case, not dependent on whether material is void/solid)
    # matDT and matDT_2 are (3, 3)
    num_elements = g_tf.shape[0]

    # Compute powers
    g_tf_p2 = g_tf ** p_2           # shape (num_elements,)

    # Reshape for broadcasting
    g_tf_p2 = g_tf_p2.reshape(1, 1, num_elements)

    # Broadcast matDT and matDT_2 to (3, 3, num_elements)
    matDT_b = matDT[:, :, np.newaxis]
    matDT_2_b = matDT_2[:, :, np.newaxis]

    # Compute D_e and D_e_FD as (3, 3, num_elements)
    D_e = g_tf_p2 * matDT_b + (1 - g_tf_p2) * matDT_2_b            # (3, 3, 1152)

    ## Memory-optimized: Compute second_term directly without storing full dD_e_matrix
    num_elements = len(elements_arr)
    _matDT_d = np.asarray(matDT, dtype=np.float64)
    _matDT_2_d = np.asarray(matDT_2, dtype=np.float64)
    matDT_diff = _matDT_d - _matDT_2_d  # (3, 3) - compute once

    # Pre-compute reusable parts (keep as float64 for computation)
    # Memory-optimized: Don't store temp_ij as full array - compute on-the-fly in chunks
    # temp_ij[i,j] = p_2 * (g_tf ** (p_2 - 1))[i] * der_type[i, j]
    g_tf_pow_p2_minus_1 = p_2 * (g_tf ** (p_2 - 1))  # (num_elements,) - precompute this part
    g_sf_pow = g_sf ** p  # shape: (num_elements,)

    # Compute B @ u_vecs for all elements
    u_vecs = np.asarray(u_vecs, dtype=np.float64)
    Bu_all = np.dot(B, u_vecs)  # shape: (3, num_elements)
    Bu_all = Bu_all.T  # shape: (num_elements, 3)

    # Free u_vecs and EL_all after they're used (no longer needed)
    del u_vecs, EL_all
    gc.collect()  # Force garbage collection

    # Memory-optimized: Don't store full second_term array (7.59 GiB)
    # Instead, compute second_term in chunks when needed later
    # Keep the components needed to compute second_term on-the-fly:
    # second_term[i,j,:] = g_sf_pow[i] * temp_ij[i,j] * matDT_diff @ Bu_all[i]
    # where temp_ij[i,j] = g_tf_pow_p2_minus_1[i] * der_type[i, j] (computed on-the-fly)
    matDT_diff_Bu = (matDT_diff @ Bu_all.T).T  # (num_elements, 3)
    # Keep g_tf_pow_p2_minus_1, matDT_diff_Bu, g_sf_pow, der_type for chunked computation later
    # Don't delete them yet - they'll be used when computing dsigma_e_dx2 in chunks


    ### Vectorized version for dUdx_full_2 (similar to dUdx_full approach)
    ### Memory-optimized: Compute on-the-fly instead of storing full arrays

    # Precompute reusable parts (K_e_arr was deleted earlier, so we need to recreate)
    K_e_arr = K_e.toarray() if hasattr(K_e, 'toarray') else K_e
    K_e_2_arr = K_e_2.toarray() if hasattr(K_e_2, 'toarray') else K_e_2
    K_e_arr = K_e_arr.astype(np.float64)
    K_e_2_arr = K_e_2_arr.astype(np.float64)
    K_e_diff = K_e_arr - K_e_2_arr  # shape: (8, 8)

    # Pre-compute reusable parts (keep as float64 for computation)
    g_sf_pow = g_sf ** p  # (num_elements,)
    g_tf_pow = g_tf ** (p_2 - 1)  # (num_elements,)

    # Use the same DOF structure as before (reuse if already computed)
    if 'DOF' not in locals() or DOF is None:
        DOF = np.column_stack([
            np.column_stack((2*elements_arr[:, 0], 2*elements_arr[:, 0]+1)),
            np.column_stack((2*elements_arr[:, 1], 2*elements_arr[:, 1]+1)),
            np.column_stack((2*elements_arr[:, 2], 2*elements_arr[:, 2]+1)),
            np.column_stack((2*elements_arr[:, 3], 2*elements_arr[:, 3]+1))
        ])  # shape: (num_elements, 8)
        row_idx = np.repeat(DOF, 8, axis=1).reshape(num_elements, 8, 8)
        col_idx = np.tile(DOF, (1, 8)).reshape(num_elements, 8, 8)

    # Vectorized version for rhs_matrix_2
    # Memory-optimized: Compute rhs_matrix_2 columns on-the-fly and solve immediately
    # This avoids storing the full (37248, 18432) rhs_matrix_2 array (5.12 GiB)

    # Reuse reduced DOF mapping if already computed
    if 'reduced_map' not in locals() or reduced_map is None:
        Reduced_stiffness_indices_arr = np.array(Reduced_stiffness_indices, dtype=np.int32)
        num_reduced = len(Reduced_stiffness_indices)
        reduced_map = np.full(num_dof, -1, dtype=np.int32)
        reduced_map[Reduced_stiffness_indices_arr] = np.arange(num_reduced)
    else:
        num_reduced = len(Reduced_stiffness_indices)

    # Reuse row/col indices in reduced space if already computed
    if 'row_idx_reduced_base' not in locals() or row_idx_reduced_base is None:
        row_idx_flat = row_idx.reshape(-1)
        col_idx_flat = col_idx.reshape(-1)
        mask_base = (reduced_map[row_idx_flat] >= 0) & (reduced_map[col_idx_flat] >= 0)
        row_idx_reduced_base = reduced_map[row_idx_flat[mask_base]]
        col_idx_reduced_base = reduced_map[col_idx_flat[mask_base]]

    # Allocate dUdx_full_2 directly (keep as float64 for precision)
    dUdx_full_2 = np.zeros((num_dof, num_vars), dtype=np.float64)

    # Process each variable - compute rhs column, solve, and store directly
    for j in range(num_vars):
        # Compute element-wise factors for this j only (keep as float64)
        # Get column j of der_type (compute on-demand if lazy)
        if isinstance(der_type, LazySensitivityMatrix):
            der_type_col_j = der_type.get_column(j)  # (num_elements,)
        else:
            der_type_col_j = der_type[:, j]  # (num_elements,)
        factor_j = p_2 * g_tf_pow * der_type_col_j  # (num_elements,)
        element_factors = g_sf_pow * factor_j  # shape: (num_elements,)
    
        # Compute data for this variable only
        data_j = (element_factors[:, np.newaxis, np.newaxis] * K_e_diff[np.newaxis, :, :]).reshape(-1)[mask_base]
    
        # Free intermediate
        del factor_j, element_factors
    
        # Build reduced sparse matrix
        K_der_global_2 = coo_matrix(
            (data_j, (row_idx_reduced_base, col_idx_reduced_base)),
            shape=(num_reduced, num_reduced)
        ).tocsc()
    
        # Compute RHS column on-the-fly (don't store in full rhs_matrix_2)
        rhs_column_2 = -K_der_global_2 @ U_solved  # (num_reduced,)
    
        # Free intermediate
        del data_j, K_der_global_2
    
        # Solve immediately and store directly in dUdx_full_2
        dUdx_reduced_col_2 = lu.solve(rhs_column_2)  # (num_reduced,)
        dUdx_full_2[Reduced_stiffness_indices_arr, j] = dUdx_reduced_col_2.astype(np.float64)
    
        # Free rhs column immediately
        del rhs_column_2, dUdx_reduced_col_2

    # Free memory after loop
    del g_sf_pow, g_tf_pow, K_e_arr, K_e_2_arr, K_e_diff
    # Free U, U_solved, sK_reduced, lu and related arrays after dUdx_full_2 is computed
    # They're no longer needed since dUdx_full_2 contains all the derivative information
    del U_solved, sK_reduced, lu
    del Reduced_stiffness_indices_arr, reduced_map, row_idx_flat, col_idx_flat
    del mask_base, row_idx_reduced_base, col_idx_reduced_base
    del row_idx, col_idx, DOF
    gc.collect()  # Force garbage collection after freeing large arrays
    # Note: dUdx_full_2 is kept for later use
    # make third term a 1152x1152x3x1 structure
    num_elements = len(elements_arr)
    num_vars = num_elements  # 1152

    # Prepare all EL indices for all elements
    EL_all = np.array([
        [2 * ele[0], 2 * ele[0] + 1, 2 * ele[1], 2 * ele[1] + 1,
         2 * ele[2], 2 * ele[2] + 1, 2 * ele[3], 2 * ele[3] + 1]
        for ele in elements_arr
    ])  # shape: (num_elements, 8)

    # Prepare g_sf^p for all elements (keep as float64 for computation)
    g_sf_pow = g_sf ** p  # shape: (num_elements,)

    # Prepare B (shape: (3, 8)), constant for all elements

    # Prepare dUdx_full_2 (shape: (num_dof, num_vars))
    # Memory-optimized: Compute Bu element-by-element to avoid creating huge (num_elements, 8, num_vars) array
    # Instead of: dudx_all_2 = dUdx_full_2[EL_all, :] which would be (18432, 8, 18432) = 10.1 GiB
    # We compute B @ dUdx_full_2[EL_all[i], :] for each element i directly

    B_numeric = np.array(B, dtype=np.float64)  # (3, 8)
    # Memory-optimized: Don't store full third_term_2 array (7.59 GiB)
    # Instead, compute third_term_2 in chunks when needed later
    # Keep the components needed to compute third_term_2 on-the-fly:
    # - dUdx_full_2, EL_all, B_numeric, D_e, g_sf_pow
    # third_term_2 will be computed in chunks when computing dsigma_e_dx2
    # Ensure D_e is float64 for computation (keep it for later use)
    D_e = np.asarray(D_e, dtype=np.float64)
    ### dPsi/dx  (only non-zero for when x is type (mat1/mat2))

    def compute_mu_wiggle_gradients(mu_wiggle, Nel_x, Nel_y, Delta_x):
        # Ensure input is float64
        mu_wiggle = np.asarray(mu_wiggle, dtype=np.float64)
    
        Nnodes_x = Nel_x + 1
        Nnodes_y = Nel_y + 1
    
        # Reshape mu_wiggle to 2D grid for easier indexing
        mu_grid = mu_wiggle.reshape((Nnodes_y, Nnodes_x))
    
        # Vectorized version - replaces the nested loops
        # Extract all corner nodes at once using array slicing
        # Shape: (Nel_y, Nel_x) for each
        top_left = mu_grid[:-1, :-1]      # mu_grid[0:Nel_y, 0:Nel_x]
        top_right = mu_grid[:-1, 1:]      # mu_grid[0:Nel_y, 1:Nel_x+1]
        bottom_left = mu_grid[1:, :-1]     # mu_grid[1:Nel_y+1, 0:Nel_x]
        bottom_right = mu_grid[1:, 1:]     # mu_grid[1:Nel_y+1, 1:Nel_x+1]
    
        # Compute averages using vectorized operations
        avg_left = (top_left + bottom_left) / 2   # shape: (Nel_y, Nel_x)
        avg_right = (top_right + bottom_right) / 2  # shape: (Nel_y, Nel_x)
        avg_top = (top_left + top_right) / 2     # shape: (Nel_y, Nel_x)
        avg_bottom = (bottom_left + bottom_right) / 2  # shape: (Nel_y, Nel_x)
    
        # Free memory after computing averages
        del top_left, top_right, bottom_left, bottom_right
    
        # Compute finite differences using vectorized operations
        dmu_dx = (avg_right - avg_left) / Delta_x   # shape: (Nel_y, Nel_x)
        dmu_dy = (avg_top - avg_bottom) / Delta_x   # shape: (Nel_y, Nel_x)
    
        # Free memory after computing gradients
        del avg_left, avg_right, avg_top, avg_bottom, mu_grid
    
        return dmu_dx, dmu_dy

    # Example usage after mu_wiggle is computed:
    dmu_dx, dmu_dy = compute_mu_wiggle_gradients(mu_wiggle, Nel_x, Nel_y, Delta_x)   #reshaped to 24 x 48


    #reshaped to 24 x 48
    length_dmu = (dmu_dx**2 + dmu_dy**2)**(1/2)
    g_xe = np.zeros_like(dmu_dx)
    g_ye = np.zeros_like(dmu_dy)

    g_xe = -np.divide(dmu_dx, length_dmu, out=g_xe, where=length_dmu != 0)
    g_ye = -np.divide(dmu_dy, length_dmu, out=g_ye, where=length_dmu != 0)

    # 1152 , 1
    X = Nel_x * Nel_y
    g_xe_flat = g_xe.reshape(X, 1)
    g_ye_flat = g_ye.reshape(X, 1)

    #Psi = np.array([[g_xe**2, g_ye**2, 2*g_xe*g_ye],[-g_xe*g_ye, g_xe*g_ye, g_xe**2-g_ye**2]])
    g_xe_flat = np.squeeze(g_xe_flat)
    g_ye_flat = np.squeeze(g_ye_flat)

    Psi = np.array([[g_xe_flat**2, g_ye_flat**2, 2*g_xe_flat*g_ye_flat],[-g_xe_flat*g_ye_flat, g_xe_flat*g_ye_flat, g_xe_flat**2-g_ye_flat**2]])

    # Free memory: g_xe, g_ye, and length_dmu are no longer needed after Psi is created
    del g_xe, g_ye, length_dmu

    ## sensitivities of g_xe and g_ye
    # Memory-optimized: Process dmu_wiggle in chunks to avoid storing full (N, X) = (18721, 18432) array
    # Instead of: dmu_wiggle = multi_step_sensitivity_nodal(...) which allocates 2.57 GiB
    # We'll process columns in chunks and compute dPsi_dx2_full column by column

    # Get the original mu_wiggle gradients (mu_x, mu_y) - these are the "current" values
    mu_x = dmu_dx  # Original spatial gradients, shape: (Nel_y, Nel_x)
    mu_y = dmu_dy  # shape: (Nel_y, Nel_x)
    L = np.sqrt(mu_x**2 + mu_y**2)  # Original length, shape: (Nel_y, Nel_x)

    # Free memory: dmu_dx and dmu_dy are no longer needed after copying to mu_x and mu_y
    del dmu_dx, dmu_dy

    # dmu_wiggle shape is (N, X) where N = Nnodes_x * Nnodes_y, X = Nel_x * Nel_y
    Nnodes_x = Nel_x + 1
    Nnodes_y = Nel_y + 1
    N = Nnodes_x * Nnodes_y

    # Expand mu_x and mu_y for broadcasting: (Nel_y, Nel_x, 1)
    mu_x_expanded = mu_x[:, :, np.newaxis]  # (Nel_y, Nel_x, 1)
    mu_y_expanded = mu_y[:, :, np.newaxis]  # (Nel_y, Nel_x, 1)
    L_expanded = L[:, :, np.newaxis]         # (Nel_y, Nel_x, 1)
    mask_expanded = L_expanded != 0  # (Nel_y, Nel_x, 1)
    mask_squeezed = mask_expanded.squeeze(axis=2)  # (Nel_y, Nel_x)

    # Free memory: expanded arrays are no longer needed after mask_squeezed is created
    del mu_x_expanded, mu_y_expanded, L_expanded, mask_expanded

    # Build base matrices for dPsi_dx2 (these are the same for all variables)
    # g_xe_flat and g_ye_flat are shape (X,)
    g_xe_flat_1d = g_xe_flat.flatten() if g_xe_flat.ndim > 1 else g_xe_flat
    g_ye_flat_1d = g_ye_flat.flatten() if g_ye_flat.ndim > 1 else g_ye_flat

    # For dPsi_dx2x: [[2*gx, 0, 2*gy], [-gy, gy, 2*gx]]
    dPsi_dx2x_base = np.stack([
        np.stack([2 * g_xe_flat_1d, np.zeros(X), 2 * g_ye_flat_1d], axis=0),
        np.stack([-g_ye_flat_1d, g_ye_flat_1d, 2 * g_xe_flat_1d], axis=0)
    ], axis=0)  # shape: (2, 3, X)

    # For dPsi_dx2y: [[0, 2*gy, 2*gx], [-gx, gx, -2*gy]]
    dPsi_dx2y_base = np.stack([
        np.stack([np.zeros(X), 2 * g_ye_flat_1d, 2 * g_xe_flat_1d], axis=0),
        np.stack([-g_xe_flat_1d, g_xe_flat_1d, -2 * g_ye_flat_1d], axis=0)
    ], axis=0)  # shape: (2, 3, X)

    # Free memory: g_xe_flat, g_ye_flat, and their 1d versions are no longer needed after base matrices are created
    del g_xe_flat, g_ye_flat, g_xe_flat_1d, g_ye_flat_1d

    # Memory-optimized: Instead of storing full dPsi_dx2_full (2, 3, X, X) = 15.2 GiB,
    # or even term1 (2, X, X) = 5.1 GiB, or sigma_ne_sigma_te_dx2 (X, X, 2) = 5.06 GiB,
    # we'll process everything in chunks and only store what we need when we need it
    num_elements = X  # X = Nel_x * Nel_y = number of elements
    # Don't initialize sigma_ne_sigma_te_dx2 yet - we'll process it in chunks
    # We'll accumulate term1 contributions in a temporary buffer for each variable chunk

    # Process dmu_wiggle in column chunks to avoid storing full (N, X) array
    # We need to modify multi_step_sensitivity_nodal to yield columns, or process in chunks
    # For now, we'll compute dmu_wiggle in chunks by processing columns of the sensitivity matrix
    chunk_size = 500  # Process 500 columns at a time - adjust based on available memory

    # Get the sensitivity matrix computation components (we'll need to access them)
    # We need to call multi_step_sensitivity_nodal but process it in chunks
    # Since the function returns the full matrix, we'll need to modify our approach
    # Instead, we'll compute the sensitivity matrix internally in chunks

    # Recompute the necessary components for chunked processing
    nodal_densities = helmholtz_nodal_filter(g_type, r, Nel_x, Nel_y, Delta_x)
    beta_x = beta * (nodal_densities - ETA_1)
    nodal_densities_step = (np.tanh(beta*ETA_1) + np.tanh(beta_x)) / (np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1)))

    # Use pre-calculated nodal Helmholtz operator matrices
    H1 = H_nodal_solid
    H2 = H_nodal_solid_2

    # Calculate projection derivative
    cosh_term = np.cosh(beta_x)
    tanh_term = np.tanh(beta*ETA_1) + np.tanh(beta*(1 - ETA_1))
    proj_derivative = beta * (1.0 / (cosh_term * cosh_term)) / tanh_term

    # Create diagonal matrix from projection derivative
    P = sparse.diags(proj_derivative)

    # Calculate mean preservation factors
    mean_preservation_1 = float(np.mean(nodal_densities) / np.mean(g_type))
    mean_preservation_2 = float(np.mean(nodal_densities_step) / np.mean(nodal_densities))

    # Clean up intermediate arrays
    del nodal_densities, nodal_densities_step, beta_x, cosh_term, tanh_term

    # Free original mu_wiggle, nodal_densities, nodal_densities_step from line 955 (no longer needed)
    # They were used to compute gradients at line 1379, and nodal_densities/nodal_densities_step are recomputed above
    if 'mu_wiggle' in locals() and mu_wiggle is not None:
        del mu_wiggle
    if 'nodal_densities' in locals() and nodal_densities is not None:
        del nodal_densities
    if 'nodal_densities_step' in locals() and nodal_densities_step is not None:
        del nodal_densities_step

    # Use LU decompositions for efficiency
    from scipy.sparse.linalg import splu
    H2_T_lu = splu(H2.T.tocsc())
    H1_T_lu = splu(H1.T.tocsc())

    # Create element-to-nodal mapping matrix
    E_to_N = sparse.lil_matrix((N, X))
    for i in range(Nnodes_y):
        for j in range(Nnodes_x):
            node_idx = i * Nnodes_x + j
            contributing_elements = []
        
            # Check all possible surrounding elements
            if i > 0 and j > 0:  # Top-left element
                contributing_elements.append(((i-1) * Nel_x + (j-1), 1.0))
            if i > 0 and j < Nel_x:  # Top-right element
                contributing_elements.append(((i-1) * Nel_x + j, 1.0))
            if i < Nel_y and j > 0:  # Bottom-left element
                contributing_elements.append((i * Nel_x + (j-1), 1.0))
            if i < Nel_y and j < Nel_x:  # Bottom-right element
                contributing_elements.append((i * Nel_x + j, 1.0))
        
            # Set the mapping weights (average of contributing elements)
            if contributing_elements:
                num_contributing = len(contributing_elements)
                for element_idx, weight in contributing_elements:
                    E_to_N[node_idx, element_idx] = weight / num_contributing

    E_to_N = E_to_N.tocsr()

    # Process dmu_wiggle in column chunks (by variable) to avoid storing full (N, X) array
    # Each column of dmu_wiggle corresponds to one element variable
    # We'll compute and process columns in chunks
    nodal_sensitivity_chunk = np.zeros((N, chunk_size), dtype=np.float64)

    # Accumulate term1 contributions in column chunks to avoid storing full sigma_ne_sigma_te_dx2
    # We'll store term1 contributions in a list of column chunks, then combine with term2 later
    term1_column_chunks = []  # List to store (X, var_chunk_size, 2) chunks
    term1_var_chunk_starts = []  # Track which variables each chunk corresponds to

    # Process variable columns in chunks
    for var_chunk_start in range(0, X, chunk_size):
        var_chunk_end = builtins.min(var_chunk_start + chunk_size, X)
        var_chunk_size = var_chunk_end - var_chunk_start
    
        # For this chunk of variables, we need to compute the corresponding columns of dmu_wiggle
        # dmu_wiggle[:, var_idx] = nodal_sensitivity @ E_to_N[:, var_idx]
        # So we need to compute nodal_sensitivity first, then multiply with E_to_N columns
    
        # Compute nodal_sensitivity in row chunks (process nodes in chunks)
        nodal_sensitivity_for_vars = np.zeros((N, var_chunk_size), dtype=np.float64)
    
        # Process rows of nodal_sensitivity in chunks
        for row_chunk_start in range(0, N, chunk_size):
            row_chunk_end = builtins.min(row_chunk_start + chunk_size, N)
            row_chunk_cols = row_chunk_end - row_chunk_start
        
            # Compute chunk of nodal_sensitivity
            for j_idx, j in enumerate(range(row_chunk_start, row_chunk_end)):
                if proj_derivative[j] != 0:
                    # Solve H2.T @ lambda2_col = proj_derivative[j] * e_j
                    rhs_j = np.zeros(N, dtype=np.float64)
                    rhs_j[j] = 1.0
                    lambda2_col = proj_derivative[j] * H2_T_lu.solve(rhs_j)
                    del rhs_j  # Free immediately after use
                
                    # Apply P
                    lambda1_col = P.dot(lambda2_col)
                    del lambda2_col  # Free immediately after use
                
                    # Solve H1.T @ nodal_sens_col = lambda1_col
                    nodal_sensitivity_chunk[:, j_idx] = H1_T_lu.solve(lambda1_col)
                    del lambda1_col  # Free immediately after use
        
            # Apply mean preservation to chunk
            nodal_sensitivity_chunk[:, :row_chunk_cols] *= (mean_preservation_1 * mean_preservation_2)
        
            # Multiply with E_to_N columns for this variable chunk
            # nodal_sensitivity_chunk is (N, row_chunk_cols)
            # E_to_N[row_chunk_start:row_chunk_end, var_chunk_start:var_chunk_end] is (row_chunk_cols, var_chunk_size)
            # Result: (N, var_chunk_size)
            E_to_N_chunk = E_to_N[row_chunk_start:row_chunk_end, var_chunk_start:var_chunk_end]
            if hasattr(E_to_N_chunk, 'toarray'):
                E_to_N_chunk_dense = E_to_N_chunk.toarray()
                nodal_sensitivity_for_vars += nodal_sensitivity_chunk[:, :row_chunk_cols] @ E_to_N_chunk_dense
                del E_to_N_chunk_dense
            else:
                nodal_sensitivity_for_vars += nodal_sensitivity_chunk[:, :row_chunk_cols] @ E_to_N_chunk
        
            # Free E_to_N_chunk immediately
            del E_to_N_chunk
        
            # Clear chunk for next iteration
            nodal_sensitivity_chunk[:, :row_chunk_cols] = 0.0
    
        # Now nodal_sensitivity_for_vars is (N, var_chunk_size) - this is dmu_wiggle for this variable chunk
        # Free E_to_N_chunk if it still exists in memory
        if 'E_to_N_chunk' in locals():
            del E_to_N_chunk
        gc.collect()  # Force garbage collection before large allocation
    
        # Accumulate term1 contributions for this variable chunk
        # Memory-optimized: Pre-allocate array instead of using list to avoid memory overhead
        term1_chunk = np.zeros((X, var_chunk_size, 2), dtype=np.float64)  # Pre-allocate
    
        # Process each column (variable) in this chunk
        for var_idx_local, var_idx in enumerate(range(var_chunk_start, var_chunk_end)):
            # Extract column for this variable: shape (N,)
            dmu_wiggle_col = nodal_sensitivity_for_vars[:, var_idx_local]
        
            # Reshape to (Nnodes_y, Nnodes_x) for gradient computation
            dmu_wiggle_reshaped = dmu_wiggle_col.reshape((Nnodes_y, Nnodes_x))
        
            # Extract corner nodes
            top_left = dmu_wiggle_reshaped[:-1, :-1]      # (Nel_y, Nel_x)
            top_right = dmu_wiggle_reshaped[:-1, 1:]      # (Nel_y, Nel_x)
            bottom_left = dmu_wiggle_reshaped[1:, :-1]    # (Nel_y, Nel_x)
            bottom_right = dmu_wiggle_reshaped[1:, 1:]    # (Nel_y, Nel_x)
        
            # Free dmu_wiggle_reshaped after extracting corners
            del dmu_wiggle_reshaped
        
            # Compute averages
            avg_left = (top_left + bottom_left) / 2          # (Nel_y, Nel_x)
            avg_right = (top_right + bottom_right) / 2       # (Nel_y, Nel_x)
            avg_top = (top_left + top_right) / 2            # (Nel_y, Nel_x)
            avg_bottom = (bottom_left + bottom_right) / 2    # (Nel_y, Nel_x)
        
            # Free corner nodes after computing averages
            del top_left, top_right, bottom_left, bottom_right
        
            # Compute finite differences
            dmu_x_dg = (avg_right - avg_left) / Delta_x   # (Nel_y, Nel_x)
            dmu_y_dg = (avg_top - avg_bottom) / Delta_x   # (Nel_y, Nel_x)
        
            # Free averages after computing differences
            del avg_left, avg_right, avg_top, avg_bottom
        
            # Compute dg_xe and dg_ye for this variable using broadcasting
            # dg_xe = -(mu_y * (mu_y * dmu_x_dg - mu_x * dmu_y_dg)) / L^3
            # dg_ye = -(mu_x * (mu_x * dmu_y_dg - mu_y * dmu_x_dg)) / L^3
            dg_xe = np.zeros((Nel_y, Nel_x), dtype=np.float64)
            dg_ye = np.zeros((Nel_y, Nel_x), dtype=np.float64)
        
            dg_xe[mask_squeezed] = -(
                mu_y[mask_squeezed] * 
                (mu_y[mask_squeezed] * dmu_x_dg[mask_squeezed] - 
                 mu_x[mask_squeezed] * dmu_y_dg[mask_squeezed])
            ) / (L[mask_squeezed] ** 3)
        
            dg_ye[mask_squeezed] = -(
                mu_x[mask_squeezed] * 
                (mu_x[mask_squeezed] * dmu_y_dg[mask_squeezed] - 
                 mu_y[mask_squeezed] * dmu_x_dg[mask_squeezed])
            ) / (L[mask_squeezed] ** 3)
        
            # Free dmu_x_dg and dmu_y_dg after computing dg_xe and dg_ye
            del dmu_x_dg, dmu_y_dg
        
            # Flatten to (X,) for this variable
            dg_xe_flat_var = dg_xe.flatten()  # (X,)
            dg_ye_flat_var = dg_ye.flatten()  # (X,)
        
            # Free dg_xe and dg_ye after flattening
            del dg_xe, dg_ye
        
            # Compute dPsi_dx2 for this variable: (2, 3, X)
            # dPsi_dx2x = dPsi_dx2x_base * dg_xe_flat_var (broadcast)
            # dPsi_dx2y = dPsi_dx2y_base * dg_ye_flat_var (broadcast)
            dPsi_dx2x_var = dPsi_dx2x_base * dg_xe_flat_var[np.newaxis, np.newaxis, :]  # (2, 3, X)
            dPsi_dx2y_var = dPsi_dx2y_base * dg_ye_flat_var[np.newaxis, np.newaxis, :]  # (2, 3, X)
            dPsi_dx2_var = dPsi_dx2x_var + dPsi_dx2y_var  # (2, 3, X)
        
            # Free intermediate arrays
            del dg_xe_flat_var, dg_ye_flat_var, dPsi_dx2x_var, dPsi_dx2y_var
        
            # Instead of storing in dPsi_dx2_full or term1, immediately compute contribution and add to sigma_ne_sigma_te_dx2
            # term1[:, :, var_idx] = dPsi_dx2_var @ sigma_e would be (2, X)
            # After transpose: (X, 2) which should be added to sigma_ne_sigma_te_dx2[:, var_idx, :]
            # dPsi_dx2_var is (2, 3, X), sigma_e is (3, X)
            # We need: (2, 3, X) @ (3, X) -> (2, X), then transpose to (X, 2)
            # Using einsum: 'ijk,jk->ik' where i=2, j=3, k=X
            term1_contribution = np.einsum('ijk,jk->ik', dPsi_dx2_var, sigma_e)  # (2, X)
            del dPsi_dx2_var
        
            # Store contribution directly in pre-allocated array instead of appending to list
            term1_chunk[:, var_idx_local, :] = term1_contribution.T  # (X, 2)
            del term1_contribution
        
            # Periodic cleanup every 50 variables to prevent memory buildup
            if (var_idx_local + 1) % 50 == 0:
                gc.collect()
    
        # Free nodal_sensitivity_for_vars which is only used in this chunk iteration
        del nodal_sensitivity_for_vars
        gc.collect()  # Force garbage collection before storing chunk
    
        # term1_chunk is already allocated and filled, no need to stack
        # Store this term1 chunk for later combination with term2
        term1_column_chunks.append(term1_chunk)
        term1_var_chunk_starts.append(var_chunk_start)
        del term1_chunk
        gc.collect()  # Force garbage collection after processing each variable chunk

    # Free memory immediately after computing term1 (all chunks processed)
    # Free variables that were used in all chunk iterations but are no longer needed
    del mu_x, mu_y, L, mask_squeezed
    del dPsi_dx2x_base, dPsi_dx2y_base, nodal_sensitivity_chunk
    del P, H2_T_lu, H1_T_lu, E_to_N, proj_derivative
    # Note: mu_x_expanded, mu_y_expanded, L_expanded, mask_expanded were already deleted earlier
    # term1 is now (2, X, X) and will be used later - no need to convert to float32 yet

    # Using einsum (most efficient and readable)
    sigma_ne_sigma_te = np.einsum('ijk,jk->ik', Psi, sigma_e)

    # For sigma_ne_sigma_te_dx1: derivative of [sigma_ne, sigma_te] with respect to x1
    # Psi (2, 3, num_elements) * dsigma_e_dx1 (num_elements, num_elements, 3) -> (num_elements, num_elements, 2)
    # Compute dsigma_e_dx1 in chunks to avoid storing full (num_elements, num_elements, 3) array
    # This avoids the 7.59 GiB memory allocation while maintaining performance
    chunk_size = 2000  # Process 2000 rows at a time - adjust based on available memory
    # Recreate EL_all (needed for dUdx_full[EL_all[i], :] in chunked computation)
    # It may have been deleted earlier, so recreate it here
    EL_all = np.array([
        [2 * ele[0], 2 * ele[0] + 1, 2 * ele[1], 2 * ele[1] + 1,
         2 * ele[2], 2 * ele[2] + 1, 2 * ele[3], 2 * ele[3] + 1]
        for ele in elements_arr
    ])  # shape: (num_elements, 8)
    # Ensure B_numeric is available (needed for B @ dudx_element computation)
    B_numeric = np.array(B, dtype=np.float64)  # (3, 8)
    # Ensure D_e is float64 for computation accuracy (convert once before loop)
    D_e_compute = np.asarray(D_e, dtype=np.float64)
    # Memory-optimized: Store results in row chunks to avoid full (num_elements, num_elements, 2) = 5.06 GiB allocation
    # This follows the same pattern as sigma_ne_sigma_te_dx2
    sigma_ne_sigma_te_dx1_row_chunks = []  # List to store (chunk_size, num_elements, 2) chunks
    sigma_ne_sigma_te_dx1_row_starts = []  # Track which rows each chunk corresponds to
    for k_start in range(0, num_elements, chunk_size):
        k_end = builtins.min(k_start + chunk_size, num_elements)
        # Compute chunk of first_term: first_term[k_start:k_end, :, :]
        # Memory-optimized: Get der_solid chunk on-the-fly
        if isinstance(der_solid, LazySensitivityMatrix):
            # Compute rows k_start:k_end by getting columns and transposing
            der_solid_chunk = np.zeros((k_end - k_start, num_elements), dtype=np.float64)
            for j in range(num_elements):
                col_j = der_solid.get_column(j)  # (num_elements,)
                der_solid_chunk[:, j] = col_j[k_start:k_end]
        else:
            der_solid_chunk = der_solid[k_start:k_end, :]  # (chunk_size, num_elements)
    
        # Process in sub-chunks along num_elements dimension to avoid allocating full (chunk_size, num_elements, 3) array
        # Collect sub-chunk results instead of pre-allocating full array to avoid memory issue
        sub_chunk_size = 500  # Process num_elements in sub-chunks (reduced from 2000 to avoid memory allocation errors)
        chunk_result_subchunks = []  # List to collect sub-chunk results
        for j_start in range(0, num_elements, sub_chunk_size):
            j_end = builtins.min(j_start + sub_chunk_size, num_elements)
            # Compute sub-chunk of first_term
            chunk_first_term_sub = np.zeros((k_end - k_start, j_end - j_start, 3), dtype=np.float64)
            # Use einsum for efficient computation without intermediate arrays
            np.einsum('i,ij,il->ijl', 
                      g_sf_pow_p_minus_1[k_start:k_end], 
                      der_solid_chunk[:, j_start:j_end], 
                      D_e_Bu_T[k_start:k_end, :],
                      out=chunk_first_term_sub)
            # Compute sub-chunk of third_term on-the-fly
            chunk_third_term_sub = np.zeros((k_end - k_start, j_end - j_start, 3), dtype=np.float64)
            for i_idx, i in enumerate(range(k_start, k_end)):
                # Extract dUdx for this element's DOFs: shape (8, num_vars)
                dudx_element = dUdx_full[EL_all[i], :].astype(np.float64)  # (8, num_vars)
                # Compute B @ dudx_element: (3, 8) @ (8, num_vars) = (3, num_vars)
                Bu_element = (B_numeric @ dudx_element).T  # Transpose to (num_vars, 3)
                # Compute D_e[:,:,i] @ Bu_element.T: (3, 3) @ (3, num_vars) = (3, num_vars)
                # Then transpose to (num_vars, 3) and multiply by g_sf_pow[i]
                # Extract only the j_start:j_end slice to match chunk_third_term_sub[i_idx, :, :] shape (j_end - j_start, 3)
                chunk_third_term_sub[i_idx, :, :] = g_sf_pow[i] * (D_e_compute[:, :, i] @ Bu_element.T).T[j_start:j_end, :]  # (j_end - j_start, 3)
                del dudx_element, Bu_element  # Free immediately
            # Compute sub-chunk of dsigma_e_dx1 using in-place addition
            chunk_first_term_sub += chunk_third_term_sub  # In-place addition reuses memory
            # Use sub-chunk immediately in einsum and accumulate result
            # Psi[:, :, k_start:k_end] is (2, 3, chunk_size)
            # chunk_first_term_sub is (chunk_size, sub_chunk_size, 3)
            # Result sub-chunk is (chunk_size, sub_chunk_size, 2)
            chunk_result_sub = np.einsum('ijk,klj->kli', 
                                         Psi[:, :, k_start:k_end], 
                                         chunk_first_term_sub)  # (chunk_size, sub_chunk_size, 2)
            chunk_result_subchunks.append(chunk_result_sub)
            del chunk_first_term_sub, chunk_third_term_sub  # Free sub-chunks immediately
        del der_solid_chunk  # Free immediately before concatenation to free up memory
        gc.collect()  # Force garbage collection to free memory before large allocation
        # Concatenate all sub-chunks along the num_elements dimension (axis=1) to form full chunk_result
        chunk_result = np.concatenate(chunk_result_subchunks, axis=1)  # (chunk_size, num_elements, 2)
        del chunk_result_subchunks  # Free the list after concatenation
        # Store this chunk instead of writing to full array
        sigma_ne_sigma_te_dx1_row_chunks.append(chunk_result)
        sigma_ne_sigma_te_dx1_row_starts.append(k_start)
        del chunk_result  # Free chunk immediately
    # Free components after using them (keep g_sf_pow as it may be used elsewhere)
    # dUdx_full is no longer needed after computing sigma_ne_sigma_te_dx1
    del dUdx_full, g_sf_pow_p_minus_1, D_e_Bu_T, der_solid, EL_all, D_e_compute
    gc.collect()  # Force garbage collection after freeing large arrays

    # For sigma_ne_sigma_te_dx2: derivative of [sigma_ne, sigma_te] with respect to x2
    # This combines two terms that should both result in (X, X, 2):
    # 1. term1 contributions are stored in term1_column_chunks (computed in chunks above)
    # 2. Psi (2, 3, X) * dsigma_e_dx2 (X, X, 3) -> (X, X, 2)
    # We'll combine them in chunks to avoid storing the full array

    # Memory-optimized: Compute dsigma_e_dx2 in chunks to avoid storing full (7.59 GiB) array
    # dsigma_e_dx2 = second_term + third_term_2, both computed in chunks
    chunk_size = 2000  # Process 2000 rows at a time
    num_elements = len(elements_arr)
    # We'll process term2 in row chunks and combine with term1 contributions on-the-fly
    # Store results in a list of row chunks to avoid full array allocation
    sigma_ne_sigma_te_dx2_row_chunks = []  # List to store (chunk_size, X, 2) chunks
    sigma_ne_sigma_te_dx2_row_starts = []  # Track which rows each chunk corresponds to

    # Recreate EL_all if needed (for computing third_term_2 chunks)
    EL_all = np.array([
        [2 * ele[0], 2 * ele[0] + 1, 2 * ele[1], 2 * ele[1] + 1,
         2 * ele[2], 2 * ele[2] + 1, 2 * ele[3], 2 * ele[3] + 1]
        for ele in elements_arr
    ])  # shape: (num_elements, 8)

    # Ensure B_numeric is available
    B_numeric = np.array(B, dtype=np.float64)  # (3, 8)

    for k_start in range(0, num_elements, chunk_size):
        k_end = builtins.min(k_start + chunk_size, num_elements)
    
        # Compute chunk of temp_ij on-the-fly: temp_ij[i,j] = g_tf_pow_p2_minus_1[i] * der_type[i, j]
        # Memory-optimized: Get der_type chunk on-the-fly
        if isinstance(der_type, LazySensitivityMatrix):
            # Compute rows k_start:k_end by getting columns and extracting rows
            der_type_chunk = np.zeros((k_end - k_start, num_elements), dtype=np.float64)
            for j in range(num_elements):
                col_j = der_type.get_column(j)  # (num_elements,)
                der_type_chunk[:, j] = col_j[k_start:k_end]
        else:
            der_type_chunk = der_type[k_start:k_end, :]  # (chunk_size, num_elements)
    
        # Compute temp_ij chunk: temp_ij[i,j] = g_tf_pow_p2_minus_1[i] * der_type[i, j]
        temp_ij_chunk = g_tf_pow_p2_minus_1[k_start:k_end, np.newaxis] * der_type_chunk  # (chunk_size, num_elements)
        del der_type_chunk  # Free immediately
    
        # Compute chunk of second_term: second_term[k_start:k_end, :, :]
        # second_term[i,j,:] = g_sf_pow[i] * temp_ij[i,j] * matDT_diff_Bu[i, :]
        chunk_second_term = (g_sf_pow[k_start:k_end, np.newaxis, np.newaxis] * 
                            temp_ij_chunk[:, :, np.newaxis] * 
                            matDT_diff_Bu[k_start:k_end, np.newaxis, :])  # (chunk_size, num_elements, 3)
        del temp_ij_chunk  # Free immediately
    
        # Compute chunk of third_term_2 on-the-fly
        # third_term_2[i, :, :] = g_sf_pow[i] * (D_e[:, :, i] @ (B_numeric @ dUdx_full_2[EL_all[i], :]).T).T
        chunk_third_term_2 = np.zeros((k_end - k_start, num_elements, 3), dtype=np.float64)
        for i_idx, i in enumerate(range(k_start, k_end)):
            # Extract dUdx for this element's DOFs: shape (8, num_vars)
            dudx_element = dUdx_full_2[EL_all[i], :].astype(np.float64)  # (8, num_vars)
            # Compute B @ dudx_element: (3, 8) @ (8, num_vars) = (3, num_vars)
            Bu_element = (B_numeric @ dudx_element).T  # Transpose to (num_vars, 3)
            # Compute D_e[:,:,i] @ Bu_element.T: (3, 3) @ (3, num_vars) = (3, num_vars)
            # Then transpose to (num_vars, 3) and multiply by g_sf_pow[i]
            chunk_third_term_2[i_idx, :, :] = g_sf_pow[i] * (D_e[:, :, i] @ Bu_element.T).T  # (num_vars, 3)
            del dudx_element, Bu_element  # Free immediately
    
        # Compute chunk of dsigma_e_dx2 = second_term + third_term_2 using in-place addition to avoid allocating new array
        chunk_second_term += chunk_third_term_2  # In-place addition reuses memory (chunk_size, num_elements, 3)
        chunk_dsigma_e_dx2 = chunk_second_term  # Reference to same array for clarity
        del chunk_third_term_2  # Free third_term_2, keep chunk_second_term (now contains the sum)
    
        # Use chunk immediately in einsum: 'ijk,klj->kli' where k is chunked
        # Psi[:, :, k_start:k_end] is (2, 3, chunk_size)
        # chunk_dsigma_e_dx2 is (chunk_size, num_elements, 3)
        # Result chunk is (chunk_size, num_elements, 2)
        chunk_term2 = np.einsum('ijk,klj->kli', Psi[:, :, k_start:k_end], chunk_dsigma_e_dx2)  # (chunk_size, num_elements, 2)
    
        # Combine term2 with term1 contributions for this row chunk
        # term1 contributions are stored in column chunks, so we need to extract the corresponding rows
        chunk_result = chunk_term2.copy()  # Start with term2: (chunk_size, num_elements, 2)
    
        # Add term1 contributions from all column chunks for these rows
        for term1_chunk, var_chunk_start in zip(term1_column_chunks, term1_var_chunk_starts):
            # term1_chunk is (X, var_chunk_size, 2)
            # We need to add term1_chunk[k_start:k_end, :, :] to chunk_result
            var_chunk_size = term1_chunk.shape[1]
            var_chunk_end = var_chunk_start + var_chunk_size
            chunk_result[:, var_chunk_start:var_chunk_end, :] += term1_chunk[k_start:k_end, :, :]
    
        # Store this combined chunk
        sigma_ne_sigma_te_dx2_row_chunks.append(chunk_result)
        sigma_ne_sigma_te_dx2_row_starts.append(k_start)
    
        del chunk_second_term, chunk_dsigma_e_dx2, chunk_term2, chunk_result  # Free chunk immediately (chunk_dsigma_e_dx2 is reference to chunk_second_term)

    # Note: term1 and term2 were combined in chunks above and stored in row chunks
    # Now we need to extract sigma_ne_dx2 and sigma_te_dx2 from the row chunks
    # We can do this in chunks to avoid allocating the full array

    # Free term1 column chunks as they're no longer needed (already combined with term2)
    del term1_column_chunks, term1_var_chunk_starts

    # dUdx_full_2 is no longer needed after computing sigma_ne_sigma_te_dx2
    del dUdx_full_2

    # Free large arrays after use (sigma_e is used in both dx1 and dx2 calculations)
    # Note: dsigma_e_dx1 and dsigma_e_dx2 were computed in chunks and never stored as full arrays
    # Note: dPsi_dx2_full, term1, and sigma_ne_sigma_te_dx2 were never stored as full arrays
    # Note: temp_ij was computed on-the-fly in chunks, so no need to delete it
    del Psi, sigma_e, g_tf_pow_p2_minus_1, matDT_diff, matDT_diff_Bu, EL_all, B_numeric, D_e
    # Note: g_sf_pow is kept as it may be used elsewhere

    # Extract sigma_ne_dx1 and sigma_te_dx1 from row chunks to avoid full array allocation
    # Memory-optimized: Extract components from chunks similar to sigma_ne_sigma_te_dx2
    sigma_ne_dx1_chunks = []
    sigma_te_dx1_chunks = []
    for chunk, row_start in zip(sigma_ne_sigma_te_dx1_row_chunks, sigma_ne_sigma_te_dx1_row_starts):
        # chunk is (chunk_size, num_elements, 2)
        sigma_ne_dx1_chunks.append((row_start, chunk[:, :, 0]))  # (chunk_size, num_elements)
        sigma_te_dx1_chunks.append((row_start, chunk[:, :, 1]))  # (chunk_size, num_elements)

    # Reconstruct full 2D arrays from chunks (needed for later use)
    # Keep as float64 for precision (user requirement)
    sigma_ne_dx1 = np.zeros((num_elements, num_elements), dtype=np.float64)
    sigma_te_dx1 = np.zeros((num_elements, num_elements), dtype=np.float64)
    for row_start, chunk in sigma_ne_dx1_chunks:
        chunk_size_actual = chunk.shape[0]
        sigma_ne_dx1[row_start:row_start+chunk_size_actual, :] = chunk.astype(np.float64)
    for row_start, chunk in sigma_te_dx1_chunks:
        chunk_size_actual = chunk.shape[0]
        sigma_te_dx1[row_start:row_start+chunk_size_actual, :] = chunk.astype(np.float64)

    # Free row chunks after extracting
    del sigma_ne_sigma_te_dx1_row_chunks, sigma_ne_sigma_te_dx1_row_starts
    del sigma_ne_dx1_chunks, sigma_te_dx1_chunks

    # Extract sigma_ne_dx2 and sigma_te_dx2 from row chunks in chunks to avoid full array allocation
    # We'll process the row chunks and extract the components
    sigma_ne_dx2_chunks = []
    sigma_te_dx2_chunks = []
    for chunk, row_start in zip(sigma_ne_sigma_te_dx2_row_chunks, sigma_ne_sigma_te_dx2_row_starts):
        # chunk is (chunk_size, X, 2)
        sigma_ne_dx2_chunks.append((row_start, chunk[:, :, 0]))  # (chunk_size, X)
        sigma_te_dx2_chunks.append((row_start, chunk[:, :, 1]))  # (chunk_size, X)

    # Now we have the data in chunks, but we need full arrays for later use
    # Keep as float64 for precision (user requirement)
    sigma_ne_dx2 = np.zeros((num_elements, num_elements), dtype=np.float64)
    sigma_te_dx2 = np.zeros((num_elements, num_elements), dtype=np.float64)
    for row_start, chunk in sigma_ne_dx2_chunks:
        chunk_size_actual = chunk.shape[0]
        sigma_ne_dx2[row_start:row_start+chunk_size_actual, :] = chunk.astype(np.float64)
    for row_start, chunk in sigma_te_dx2_chunks:
        chunk_size_actual = chunk.shape[0]
        sigma_te_dx2[row_start:row_start+chunk_size_actual, :] = chunk.astype(np.float64)

    # Free row chunks after extracting
    del sigma_ne_sigma_te_dx2_row_chunks, sigma_ne_sigma_te_dx2_row_starts
    del sigma_ne_dx2_chunks, sigma_te_dx2_chunks

    # Extract components before deleting sigma_ne_sigma_te
    sigma_ne_full = sigma_ne_sigma_te[0, :]
    sigma_te_full = sigma_ne_sigma_te[1, :]

    # Free memory after extracting components
    del sigma_ne_sigma_te
    # Note: sigma_ne_sigma_te_dx1 and sigma_ne_sigma_te_dx2 were never stored as full arrays - they were processed in chunks

    Triangle = 2
    sigma_int_e = np.where(sigma_ne_full >= 0, np.sqrt(sigma_ne_full**2 + (1/Triangle**2) * sigma_te_full**2), np.sqrt((1/Triangle**2) * sigma_te_full**2))   

    p_sum = 3
    C_n = 20000 

    # Vectorized version
    g_sigma_temp = np.sum((X_int * sigma_int_e / C_n)**p_sum)
    g_sigma = g_sigma_temp**(1/p_sum) - 1

    sigma_wiggle_ne = np.where(sigma_ne_full >= 0, sigma_ne_full, 0)

    # Reshape 1D arrays to (1152, 1) for broadcasting with (1152, 1152) arrays
    sigma_wiggle_ne_2d = sigma_wiggle_ne.reshape(-1, 1)
    sigma_ne_full_2d = sigma_ne_full.reshape(-1, 1)
    sigma_te_full_2d = sigma_te_full.reshape(-1, 1)

    # Denominator: (sigma_wiggle_ne**2 + 1/Triangle**2 * sigma_te_full**2)**(-1/2)
    # Shape: (1152, 1)
    # Add small epsilon to prevent divide by zero in power operation
    denom_base = sigma_wiggle_ne_2d**2 + 1/Triangle**2 * sigma_te_full_2d**2
    # Use np.power with where parameter to avoid RuntimeWarning for zero/negative values
    denominator = np.zeros_like(denom_base)
    np.power(denom_base, -1/2, out=denominator, where=denom_base > 0)

    # Memory-optimized: Pre-allocate output array once to avoid temporary allocation in np.zeros_like
    # Shape will be (num_elements, num_elements) = (18432, 18432)
    num_elements = sigma_ne_dx1.shape[0]
    first_term_temp = np.zeros((num_elements, num_elements), dtype=np.float64)

    # Pre-compute sigma_wiggle_ne_2d squared (small array, just broadcasts)
    sigma_wiggle_ne_sq = sigma_wiggle_ne_2d * sigma_wiggle_ne_2d  # (num_elements, 1)

    # Vectorized computation
    # Avoid division by zero: when sigma_ne_full_2d is zero, the first term should be zero
    # Use np.divide with where to handle zero division safely
    # Compute numerator directly into temporary array, then divide in-place
    np.multiply(sigma_wiggle_ne_sq, sigma_ne_dx1, out=first_term_temp)
    first_term_dx1 = np.divide(first_term_temp, 
                              sigma_ne_full_2d, 
                              out=first_term_temp, 
                              where=np.abs(sigma_ne_full_2d) > 1e-10)
    dsigma_int_e_dx1 = denominator * (
        first_term_dx1 + 
        1/Triangle**2 * sigma_te_full_2d * sigma_te_dx1
    )

    # Avoid division by zero: when sigma_ne_full_2d is zero, the first term should be zero
    # Reuse the same temporary array for dx2 computation
    np.multiply(sigma_wiggle_ne_sq, sigma_ne_dx2, out=first_term_temp)
    first_term_dx2 = np.divide(first_term_temp, 
                              sigma_ne_full_2d, 
                              out=first_term_temp, 
                              where=np.abs(sigma_ne_full_2d) > 1e-10)
    dsigma_int_e_dx2 = denominator * (
        first_term_dx2 + 
        1/Triangle**2 * sigma_te_full_2d * sigma_te_dx2
    )

    # Free memory after creating dsigma_int_e derivatives
    del denominator, sigma_wiggle_ne_2d, sigma_ne_full_2d, sigma_te_full_2d

    #sensitivity analysis of g_sigma 
    sum_1 = g_sigma_temp**(1/p_sum - 1)

    # Compute intermediate arrays and free them immediately
    X_int_reshaped = X_int.reshape(-1, 1)
    sigma_int_e_reshaped = sigma_int_e.reshape(-1, 1)
    term_base = (X_int_reshaped * sigma_int_e_reshaped / C_n)**(p_sum-1)

    # Compute sum_2 in chunks to avoid storing full (18432 x 18432) intermediate array
    # This avoids the 2.53 GiB memory allocation while maintaining performance
    num_elements = len(X_int_reshaped)
    chunk_size = 2000  # Process 2000 rows at a time - adjust based on available memory
    sum_2 = np.zeros(num_elements, dtype=np.float64)
    for i in range(0, num_elements, chunk_size):
        end_idx = builtins.min(i + chunk_size, num_elements)
    
        # Get chunks of arrays
        term_base_chunk = term_base[i:end_idx]  # (chunk_size, 1)
        dX_dg_chunk = dX_dg[i:end_idx, :]  # (chunk_size, num_elements)
        dsigma_int_e_dx1_chunk = dsigma_int_e_dx1[i:end_idx, :]  # (chunk_size, num_elements)
        X_int_chunk = X_int_reshaped[i:end_idx]  # (chunk_size, 1)
        sigma_int_e_chunk = sigma_int_e_reshaped[i:end_idx]  # (chunk_size, 1)
    
        # Compute chunk contribution without creating full intermediate array
        chunk_result = term_base_chunk * (dX_dg_chunk * sigma_int_e_chunk/C_n + X_int_chunk * dsigma_int_e_dx1_chunk/C_n)
        sum_2 += np.sum(chunk_result, axis=0)
    
        # Free chunk immediately to minimize peak memory
        del chunk_result, dX_dg_chunk, dsigma_int_e_dx1_chunk

    # Compute sum_3 in chunks to avoid storing full dX_dg_type array (19208 x 19208)
    # This avoids the 2.75 GiB memory allocation while maintaining performance
    chunk_size = 2000  # Process 2000 rows at a time - adjust based on available memory
    num_elements = len(g_sf_1_minus_g_tf)
    sum_3 = np.zeros(num_elements, dtype=np.float64)
    for i in range(0, num_elements, chunk_size):
        end_idx = builtins.min(i + chunk_size, num_elements)
    
        # Memory-optimized: Get g_tsf_sens chunk on-the-fly
        if isinstance(g_tsf_sens, LazyMultiStepSensitivityMatrix):
            # Compute rows i:end_idx by getting columns and extracting rows
            g_tsf_sens_chunk = np.zeros((end_idx - i, num_elements), dtype=np.float64)
            for j in range(num_elements):
                col_j = g_tsf_sens.get_column(j)  # (num_elements,)
                g_tsf_sens_chunk[:, j] = col_j[i:end_idx]
        else:
            g_tsf_sens_chunk = g_tsf_sens[i:end_idx, :]  # (chunk_size, num_elements)
    
        # Get der_type chunk
        if isinstance(der_type, LazySensitivityMatrix):
            der_type_chunk = np.zeros((end_idx - i, num_elements), dtype=np.float64)
            for j in range(num_elements):
                col_j = der_type.get_column(j)  # (num_elements,)
                der_type_chunk[:, j] = col_j[i:end_idx]
        else:
            der_type_chunk = der_type[i:end_idx, :]  # (chunk_size, num_elements)
    
        # Compute chunk of dX_dg_type without storing full array
        chunk_dX_dg_type = (g_tsf_sens_chunk * g_sf_1_minus_g_tf[i:end_idx, np.newaxis] - 
                            der_type_chunk * g_sf_g_tsf[i:end_idx, np.newaxis])
    
        # Free chunks immediately
        del g_tsf_sens_chunk, der_type_chunk
    
        # Accumulate sum for this chunk
        sum_3 += np.sum(term_base[i:end_idx] * (chunk_dX_dg_type * sigma_int_e_reshaped[i:end_idx]/C_n + 
                                                 X_int_reshaped[i:end_idx] * dsigma_int_e_dx2[i:end_idx]/C_n), axis=0)
        del chunk_dX_dg_type  # Free chunk immediately to minimize peak memory
    # Free g_tsf_sens and intermediate arrays after using them
    del g_tsf_sens, g_sf_1_minus_g_tf, g_sf_g_tsf
    gc.collect()  # Force garbage collection after freeing large arrays

    # Free intermediate arrays
    del X_int_reshaped, sigma_int_e_reshaped, term_base
    # Free large derivative arrays after using them
    del dsigma_int_e_dx1, dsigma_int_e_dx2, sigma_ne_dx1, sigma_te_dx1, sigma_ne_dx2, sigma_te_dx2

    dg_sigma_dx1 = sum_1 * sum_2
    dg_sigma_dx2 = sum_1 * sum_3

    # Free after computing final results
    del sum_1, sum_2, sum_3
    return float(g_sigma), dg_sigma_dx1, dg_sigma_dx2

def myconstraintstress(g, grad):
    beta = continuation_beta(opt.get_numevals(), max_eval)

    g_sigma, dg_sigma_dx1, dg_sigma_dx2 = mystressconstraint(g, beta)
    if grad.size > 0:
        grad[:mid_index] = dg_sigma_dx1
        grad[mid_index:] = dg_sigma_dx2
    return g_sigma

opt = nlopt.opt(nlopt.LD_MMA, 2*len(elements))
listofzeros = [0.0000001] * 2 * len(elements)
opt.set_lower_bounds(listofzeros)
listofones = [1] * 2 * len(elements)
opt.set_upper_bounds(listofones)
opt.set_min_objective(myfunc)
# opt.set_max_objective(myfunc)
# opt.add_inequality_constraint(lambda g, grad: myconstraint(g, grad), 1e-8)
# opt.add_inequality_constraint(lambda g, grad: myconstraint2(g, grad), 1e-8)
# opt.add_inequality_constraint(lambda g, grad: myconstraint3(g, grad), 1e-8)

# opt.add_inequality_constraint(lambda g, grad: myconstraint3(g, grad), 1e-8)
opt.add_inequality_constraint(lambda g, grad: myconstraint7(g, grad), 1e-8)
opt.add_inequality_constraint(lambda g, grad: myconstraint8(g, grad), 1e-10)      # ≤ 0.25
opt.add_inequality_constraint(lambda g, grad: myconstraintstress(g, grad), 1e-8)
#opt.add_inequality_constraint(lambda g, grad: myconstraint9(g, grad), 1e-6)      # ≤ 0.25
#opt.add_inequality_constraint(lambda g, grad: -myconstraint10(g, grad), 1e-6)      # >= 0.22
#opt.add_inequality_constraint(lambda g, grad: -myconstraint8(g, grad), 1e-8)     # ≥ 0.25
opt.set_xtol_rel(1e-8)  # Relative tolerance for convergence
opt.set_ftol_rel(1e-8)  # Function value tolerance
opt.set_ftol_abs(1e-10)  # Absolute function tolerance
max_eval = 100
opt.set_maxeval(max_eval)
start_void = [0.45] * len(elements)
start_type = [0.20] * len(elements)
for x in start_type:
  start_void.append(x)
#starting_point = [0.5] * 2 * len(new_elements)
starting_point = start_void

# Add this after your myfunc function definition (around line 576)
# Create a LineProfiler instance
profiler = LineProfiler()

# Add the myfunc function to be profiled
profiler.add_function(myfunc)

# Instead, run the optimization with profiling
print("Starting optimization with line profiling...")
profiler.enable_by_count()
g = opt.optimize(starting_point)
profiler.disable_by_count()

# Print the profiling results
print("\n" + "="*60)
print("LINE PROFILING RESULTS FOR myfunc")
print("="*60)
profiler.print_stats()

# ... rest of existing code ...
# Print final optimization results
print("\n" + "="*50)
print("OPTIMIZATION COMPLETED")
print("="*50)

print(f"Constraint 8 call count: {constraint8_call_count}")
# Add this test function to verify gradients
def test_constraint8_gradient():
    print("="*60)
    print("TESTING CONSTRAINT8 GRADIENT")
    print("="*60)
    
    # Create a simple test case
    test_g = np.full(2 * len(elements), 0.6)
    
    # Finite difference step
    eps = 1e-6
    
    # Calculate constraint value at base point
    base_value = myconstraint8(test_g, np.array([]))
    
    # Test gradient w.r.t. first solid variable
    test_g_pert = test_g.copy()
    test_g_pert[0] += eps
    pert_value = myconstraint8(test_g_pert, np.array([]))
    fd_grad_solid_0 = (pert_value - base_value) / eps
    
    # Test gradient w.r.t. first type variable  
    test_g_pert = test_g.copy()
    test_g_pert[mid_index] += eps
    pert_value = myconstraint8(test_g_pert, np.array([]))
    fd_grad_type_0 = (pert_value - base_value) / eps
    
    # Get analytical gradient
    grad = np.zeros(len(test_g))
    myconstraint8(test_g, grad)
    
    print(f"Finite difference gradient [0] (solid): {fd_grad_solid_0:.8f}")
    print(f"Analytical gradient [0] (solid):       {grad[0]:.8f}")
    print(f"Error solid [0]: {abs(fd_grad_solid_0 - grad[0]):.8f}")
    print()
    print(f"Finite difference gradient [mid] (type): {fd_grad_type_0:.8f}")
    print(f"Analytical gradient [mid] (type):        {grad[mid_index]:.8f}")
    print(f"Error type [mid]: {abs(fd_grad_type_0 - grad[mid_index]):.8f}")
    
    # Check if errors are reasonable (< 1e-5)
    if abs(fd_grad_solid_0 - grad[0]) < 1e-5 and abs(fd_grad_type_0 - grad[mid_index]) < 1e-5:
        print("✓ Gradients appear correct!")
    else:
        print("✗ Gradients are WRONG!")
    
    print("="*60)

# Call this test before optimization
test_constraint8_gradient()
# End timing and print results
optimization_end_time = time.time()
total_optimization_time = optimization_end_time - optimization_start_time
print(f"\nTotal optimization time: {total_optimization_time:.2f} seconds")
print(f"Total optimization time: {total_optimization_time/60:.2f} minutes")
print(f"Total optimization time: {total_optimization_time/3600:.2f} hours")

# Split the final design variables
g_solid_final = g[:mid_index]
g_type_final = g[mid_index:]

print(f"\nFinal design variables:")
print(f"Total variables: {len(g)}")
print(f"Solid variables: {len(g_solid_final)}")
print(f"Type variables: {len(g_type_final)}")

print(f"\nSolid density variables (first 10): {g_solid_final[:10]}")
print(f"Type density variables (first 10): {g_type_final[:10]}")

beta = continuation_beta(max_eval, max_eval)
# Apply filtering to get actual final densities
new_g_final = step_filter_element(g_solid_final, r, Nel_x, Nel_y, Delta_x, beta)
new_g_2_final = step_filter_element(g_type_final, r, Nel_x, Nel_y, Delta_x, beta)

print(f"\nFiltered solid densities (first 10): {new_g_final[:10]}")
print(f"Filtered type densities (first 10): {new_g_2_final[:10]}")

# Calculate volume fractions
total_solid_volume = np.sum(new_g_final)
total_type2_volume = np.sum(new_g_final * new_g_2_final)
total_elements = len(elements)

print(f"\nVolume fractions:")
print(f"Solid volume fraction: {total_solid_volume/total_elements:.4f} (target: 0.5000)")
print(f"Type 2 volume fraction: {total_type2_volume/total_elements:.4f} (target: 0.2500)")

print(f"\nConstraint satisfaction:")
print(f"Solid constraint: {total_solid_volume - 0.5*total_elements:.6f} (should be ≤ 0)")
print(f"Type 2 constraint: {total_type2_volume - 0.25*total_elements:.6f} (should be ≤ 0)")

# Print statistics
print(f"\nStatistics:")
print(f"Min solid density: {np.min(new_g_final):.6f}")
print(f"Max solid density: {np.max(new_g_final):.6f}")
print(f"Mean solid density: {np.mean(new_g_final):.6f}")
print(f"Min type density: {np.min(new_g_2_final):.6f}")
print(f"Max type density: {np.max(new_g_2_final):.6f}")

# Calculate mean type density weighted by solid density (matches myconstraint8)
total_solid_type_volume = np.sum(new_g_final * new_g_2_final)
total_solid_volume = np.sum(new_g_final)
if total_solid_volume > 0:
    mean_type_density_weighted = total_solid_type_volume / total_solid_volume
    print(f"Mean type density (weighted by solid density): {mean_type_density_weighted:.6f}")
else:
    print("Mean type density (weighted by solid density): No solid elements found")

print("="*50)

import matplotlib.pyplot as plt

# Create density bins
bins = np.arange(0, 1.1, 0.1)  # [0, 0.1, 0.2, ..., 1.0]
bin_centers = (bins[:-1] + bins[1:]) / 2  # [0.05, 0.15, 0.25, ..., 0.95]

# Calculate histograms
hist_solid, _ = np.histogram(new_g_final, bins=bins)
hist_solid_type2, _ = np.histogram(new_g_final * new_g_2_final, bins=bins)

# Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: new_g_final (solid densities)
ax1.bar(bin_centers, hist_solid, width=0.08, alpha=0.7, color='blue', edgecolor='black')
ax1.set_xlabel('Density Range')
ax1.set_ylabel('Number of Elements')
ax1.set_title('Distribution of Solid Densities (new_g_final)')
ax1.set_xticks(bin_centers)
ax1.set_xticklabels([f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)], rotation=45)
ax1.grid(True, alpha=0.3)

# Plot 2: new_g_final * new_g_2_final (solid AND type 2 densities)
ax2.bar(bin_centers, hist_solid_type2, width=0.08, alpha=0.7, color='red', edgecolor='black')
ax2.set_xlabel('Density Range')
ax2.set_ylabel('Number of Elements')
ax2.set_title('Distribution of Solid × Type2 Densities (new_g_final × new_g_2_final)')
ax2.set_xticks(bin_centers)
ax2.set_xticklabels([f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)], rotation=45)
ax2.grid(True, alpha=0.3)

# Add statistics text
ax1.text(0.02, 0.98, f'Total elements: {len(new_g_final)}\nMean: {np.mean(new_g_final):.3f}\nStd: {np.std(new_g_final):.3f}', 
         transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax2.text(0.02, 0.98, f'Total elements: {len(new_g_final)}\nMean: {np.mean(new_g_final * new_g_2_final):.3f}\nStd: {np.std(new_g_final * new_g_2_final):.3f}', 
         transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\nDensity Distribution Analysis:")
print(f"Solid densities (new_g_final):")
print(f"  Elements in 0.0-0.1: {np.sum((new_g_final >= 0.0) & (new_g_final < 0.1))}")
print(f"  Elements in 0.9-1.0: {np.sum((new_g_final >= 0.9) & (new_g_final <= 1.0))}")
print(f"  Elements in 0.1-0.9: {np.sum((new_g_final >= 0.1) & (new_g_final < 0.9))}")

print(f"\nSolid × Type2 densities (new_g_final × new_g_2_final):")
print(f"  Elements in 0.0-0.1: {np.sum(((new_g_final * new_g_2_final) >= 0.0) & ((new_g_final * new_g_2_final) < 0.1))}")
print(f"  Elements in 0.9-1.0: {np.sum(((new_g_final * new_g_2_final) >= 0.9) & ((new_g_final * new_g_2_final) <= 1.0))}")
print(f"  Elements in 0.1-0.9: {np.sum(((new_g_final * new_g_2_final) >= 0.1) & ((new_g_final * new_g_2_final) < 0.9))}")

percentage = np.sum(((new_g_final * new_g_2_final) >= 0.9) & ((new_g_final * new_g_2_final) <= 1.0)) / len(new_g_final)
print(f"Percentage of elements in 0.9-1.0: {percentage:.4f}")


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_design(new_g_final, new_g_2_final, Nel_x, Nel_y):
    """
    Plot the optimized design:
    - Blue: Solid + Material 1 (new_g > 0.5 and new_g_2 < 0.5)
    - Red: Solid + Material 2 (new_g > 0.5 and new_g_2 > 0.5)  
    - White: Void (new_g < 0.5)
    """
    
    # Create design matrix
    design_matrix = np.zeros((Nel_y, Nel_x))
    
    # Threshold for binary classification
    solid_threshold = ETA_1
    material_threshold = ETA_1
    
    # CORRECTED: Use the same element numbering as your code
    for i in range(Nel_y):  # row (y direction)
        for j in range(Nel_x):  # column (x direction)
            # Use the SAME numbering as your element creation: j + i * (grid_size_x - 1)
            element_idx = j + i * (grid_size_x - 1)
            
            if element_idx < len(new_g_final):
                solid_density = new_g_final[element_idx]
                type_density = new_g_2_final[element_idx]
                
                if solid_density > solid_threshold:
                    # Element is solid
                    if type_density > material_threshold:
                        design_matrix[i, j] = 2  # Red: Solid + Material 2
                    else:
                        design_matrix[i, j] = 1  # Blue: Solid + Material 1
                else:
                    # Element is void
                    design_matrix[i, j] = 0  # White: Void
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Define colors: White (void), Blue (Material 1), Red (Material 2)
    colors = ['white', 'blue', 'red']
    cmap = mcolors.ListedColormap(colors)
    
    # Plot with correct orientation
    im = ax.imshow(design_matrix, cmap=cmap, aspect='equal', origin='upper')
    
    # Add colorbar with labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Void', 'Material 1', 'Material 2'])
    cbar.set_label('Material Type')
    
    # Set labels and title
    ax.set_xlabel('X Direction')
    ax.set_ylabel('Y Direction')
    ax.set_title('Optimized Design\nBlue: Material 1, Red: Material 2, White: Void')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # FIXED: Set axis labels to show physical coordinates
    # X-axis: 0 to 2 (length_x)
    x_ticks = np.linspace(0, length_x, 5)  # 0, 0.5, 1.0, 1.5, 2.0
    x_tick_positions = np.linspace(0, Nel_x-1, len(x_ticks))
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([f'{x:.1f}' for x in x_ticks])
    
    # Y-axis: 0 to 1 (length_y) - REVERSED because origin='upper'
    y_ticks = np.linspace(length_y, 0, 5)  # 1.0, 0.75, 0.5, 0.25, 0.0 (reversed)
    y_tick_positions = np.linspace(0, Nel_y-1, len(y_ticks))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels([f'{y:.2f}' for y in y_ticks])
    
    # Count elements
    void_count = np.sum(design_matrix == 0)
    material1_count = np.sum(design_matrix == 1)
    material2_count = np.sum(design_matrix == 2)
    total_count = void_count + material1_count + material2_count
    
    # Add statistics text
    stats_text = f'Total Elements: {total_count}\n'
    stats_text += f'Void: {void_count} ({void_count/total_count*100:.1f}%)\n'
    stats_text += f'Material 1: {material1_count} ({material1_count/total_count*100:.1f}%)\n'
    stats_text += f'Material 2: {material2_count} ({material2_count/total_count*100:.1f}%)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add verification: Check specific elements
    print(f"\nPlot verification:")
    print(f"Element 0: solid={new_g_final[0]:.3f}, type={new_g_2_final[0]:.3f}")
    print(f"Element 1151: solid={new_g_final[1151]:.3f}, type={new_g_2_final[1151]:.3f}")
    
    # Calculate coordinates for verification
    elem_0_row = 0 // (grid_size_x - 1)
    elem_0_col = 0 % (grid_size_x - 1)
    elem_1151_row = 1151 // (grid_size_x - 1)
    elem_1151_col = 1151 % (grid_size_x - 1)
    
    print(f"Element 0 plot coordinates: row={elem_0_row}, col={elem_0_col}")
    print(f"Element 1151 plot coordinates: row={elem_1151_row}, col={elem_1151_col}")
    
    if elem_0_row < Nel_y and elem_0_col < Nel_x:
        print(f"Plot matrix at element 0 ({elem_0_row}, {elem_0_col}): {design_matrix[elem_0_row, elem_0_col]}")
    if elem_1151_row < Nel_y and elem_1151_col < Nel_x:
        print(f"Plot matrix at element 1151 ({elem_1151_row}, {elem_1151_col}): {design_matrix[elem_1151_row, elem_1151_col]}")
    
    plt.tight_layout()
    plt.show()
    
    return design_matrix

# Create the plot
design_matrix = plot_design(new_g_final, new_g_2_final, Nel_x, Nel_y)

# Create 3D visualization of the design
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_3d_design(new_g_final, new_g_2_final, Nel_x, Nel_y, Delta_x):
    """
    Create a 3D visualization of the optimized design.
    Width is set to 1/3 of the total height of the component.
    Uses the same coordinate system as the 2D plot.
    """
    
    # Calculate component dimensions using the same system as 2D plot
    component_width = Nel_x * Delta_x  # Total width in x direction
    component_height = Nel_y * Delta_x  # Total height in y direction  
    component_depth = component_height / 3.0  # Depth is 1/3 of height
    
    print(f"Component dimensions:")
    print(f"  Height (Y): {component_height:.3f}")
    print(f"  Width (X): {component_width:.3f}")
    print(f"  Depth (Z): {component_depth:.3f}")
    
    # Threshold for binary classification
    solid_threshold = ETA_1
    material_threshold = ETA_1
    
    # Create 3D figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for different materials
    colors = {
        'void': (0.9, 0.9, 0.9, 0.1),      # Light gray, very transparent
        'material1': (0.2, 0.4, 0.8, 0.8), # Blue
        'material2': (0.8, 0.2, 0.2, 0.8)  # Red
    }
    
    # Create 3D blocks for each SOLID element only (skip void elements)
    blocks = []
    block_colors = []
    
    print(f"Processing {Nel_x * Nel_y} elements for 3D visualization...")
    
    # Count elements for statistics
    void_count = 0
    material1_count = 0
    material2_count = 0
    
    # Use the SAME coordinate system as the 2D plot
    for i in range(Nel_y):  # row (y direction)
        for j in range(Nel_x):  # column (x direction)
            # Use the SAME numbering as the 2D plot: j + i * (grid_size_x - 1)
            element_idx = j + i * (grid_size_x - 1)
            
            if element_idx < len(new_g_final):
                solid_density = new_g_final[element_idx]
                type_density = new_g_2_final[element_idx]
                
                # Calculate element center coordinates using the same system as 2D plot
                # Element center is at (j + 0.5) * Delta_x, (i + 0.5) * Delta_x
                center_x = (j + 0.5) * Delta_x
                center_y = (i + 0.5) * Delta_x
            
                # Determine material type
                if solid_density > solid_threshold:
                    if type_density > material_threshold:
                        material_type = 'material2'
                        material2_count += 1
                    else:
                        material_type = 'material1'
                        material1_count += 1
                else:
                    material_type = 'void'
                    void_count += 1
                    # Skip void elements entirely - don't create blocks for them
                    continue
                
                # Only create 3D blocks for solid elements
                # Create a larger, smoother block by extending slightly beyond element boundaries
                try:
                    # Extend the block slightly to create smoother appearance (reduce gaps)
                    extension = Delta_x * 0.1  # 10% extension for smoother look
                    
                    # Define the 8 corners of the extended cube
                    # Use the same coordinate system as 2D plot
                    corners = [
                        [-component_depth/2, center_x - Delta_x/2 - extension, -(center_y - Delta_x/2 - extension)],  # 0: Bottom-left-back
                        [-component_depth/2, center_x + Delta_x/2 + extension, -(center_y - Delta_x/2 - extension)],  # 1: Bottom-right-back
                        [-component_depth/2, center_x + Delta_x/2 + extension, -(center_y + Delta_x/2 + extension)],  # 2: Bottom-right-front
                        [-component_depth/2, center_x - Delta_x/2 - extension, -(center_y + Delta_x/2 + extension)],  # 3: Bottom-left-front
                        [component_depth/2, center_x - Delta_x/2 - extension, -(center_y - Delta_x/2 - extension)],   # 4: Top-left-back
                        [component_depth/2, center_x + Delta_x/2 + extension, -(center_y - Delta_x/2 - extension)],   # 5: Top-right-back
                        [component_depth/2, center_x + Delta_x/2 + extension, -(center_y + Delta_x/2 + extension)],   # 6: Top-right-front
                        [component_depth/2, center_x - Delta_x/2 - extension, -(center_y + Delta_x/2 + extension)]    # 7: Top-left-front
                    ]
                    
                    # Define the 6 faces of the cube - each face as a list of 4 corner indices
                    face_indices = [
                        [0, 1, 2, 3],  # Bottom face
                        [4, 7, 6, 5],  # Top face
                        [0, 4, 5, 1],  # Front face
                        [2, 6, 7, 3],  # Back face
                        [0, 3, 7, 4],  # Left face
                        [1, 5, 6, 2]   # Right face
                    ]
                    
                    # Create faces using the corner indices
                    faces = []
                    for face_idx in face_indices:
                        face = [corners[face_idx[0]], corners[face_idx[1]], corners[face_idx[2]], corners[face_idx[3]]]
                        faces.append(face)
                    
                    # Convert faces to numpy array to ensure consistent shape
                    faces_array = np.array(faces, dtype=object)
                    
                    blocks.append(faces_array)
                    block_colors.append(colors[material_type])
                    
                except Exception as e:
                    print(f"Warning: Skipping element {element_idx} due to block creation issue: {e}")
                    continue
        
        # Progress indicator for large meshes
        if (i * Nel_x + j + 1) % 1000 == 0:
            print(f"  Processed {i * Nel_x + j + 1}/{Nel_x * Nel_y} elements...")
    
    print(f"Created {len(blocks)} 3D blocks for visualization")
    
    if len(blocks) == 0:
        print("Warning: No valid blocks created for 3D visualization!")
        return None
    
    # Create the 3D collection
    try:
        # Flatten the blocks list to create a single collection
        all_faces = []
        all_face_colors = []
        
        for block_faces, block_color in zip(blocks, block_colors):
            for face in block_faces:
                all_faces.append(face)
                all_face_colors.append(block_color)
        
        print(f"Creating Poly3DCollection with {len(all_faces)} faces...")
        
        # Create the collection with flattened data
        collection = Poly3DCollection(all_faces, facecolors=all_face_colors, edgecolors='black', linewidths=0.1, alpha=0.8)
        ax.add_collection3d(collection)
        
        print("3D collection created successfully!")
        
    except Exception as e:
        print(f"Error creating 3D collection: {e}")
        print(f"Number of blocks: {len(blocks)}")
        print(f"Number of block colors: {len(block_colors)}")
        if len(blocks) > 0:
            print(f"First block shape: {np.array(blocks[0]).shape if hasattr(blocks[0], 'shape') else 'No shape attribute'}")
        return None
    
    # Set axis limits and labels (using the same coordinate system as 2D plot)
    ax.set_xlim(-component_depth/2 - Delta_x, component_depth/2 + Delta_x)  # X is depth
    ax.set_ylim(0 - Delta_x, component_width + Delta_x)  # Y is width (0 to component_width)
    ax.set_zlim(-(component_height + Delta_x), -(-Delta_x))  # Z is height (inverted, 0 to component_height)
    
    ax.set_xlabel('X Coordinate (Depth)', fontsize=12)
    ax.set_ylabel('Y Coordinate (Width)', fontsize=12)
    ax.set_zlabel('Z Coordinate (Height)', fontsize=12)
    ax.set_title('3D Visualization of Optimized Design\nBlue: Material 1, Red: Material 2, Gray: Void', fontsize=14)
    
    # Set equal aspect ratio (rotated coordinate system)
    try:
        # After 90-degree rotation around Z, then 90-degree rotation around Y: X becomes depth, Y stays width, Z becomes height
        ax.set_box_aspect([component_depth, component_width, component_height])
    except Exception as e:
        print(f"Warning: Could not set box aspect ratio: {e}")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['void'], label='Void'),
        Patch(facecolor=colors['material1'], label='Material 1'),
        Patch(facecolor=colors['material2'], label='Material 2')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Statistics are already counted in the main loop above
    total_count = void_count + material1_count + material2_count
    
    # Add statistics text
    stats_text = f'3D Component Statistics:\n'
    stats_text += f'Dimensions: {component_width:.1f} × {component_height:.1f} × {component_depth:.1f}\n'
    stats_text += f'Total Elements: {total_count}\n'
    stats_text += f'Void: {void_count} ({void_count/total_count*100:.1f}%)\n'
    stats_text += f'Material 1: {material1_count} ({material1_count/total_count*100:.1f}%)\n'
    stats_text += f'Material 2: {material2_count} ({material2_count/total_count*100:.1f}%)'
    
    # Position text box
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
              fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return None

# Create the 3D visualization
print("\n" + "="*50)
print("CREATING 3D VISUALIZATION")
print("="*50)
create_3d_design(new_g_final, new_g_2_final, Nel_x, Nel_y, Delta_x)

# Create compliance evolution plot
def plot_compliance_evolution(compliance_history):
    """
    Plot the evolution of compliance during optimization iterations.
    """
    if not compliance_history:
        print("No compliance history available for plotting.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create iteration numbers (1-indexed)
    iterations = range(1, len(compliance_history) + 1)
    
    # Plot compliance vs iteration
    plt.plot(iterations, compliance_history, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Optimization Iteration', fontsize=14)
    plt.ylabel('Compliance', fontsize=14)
    plt.title('Compliance Evolution During Optimization', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    min_compliance = min(compliance_history)
    max_compliance = max(compliance_history)
    final_compliance = compliance_history[-1]
    initial_compliance = compliance_history[0]
    
    # Add text box with statistics
    stats_text = f'Optimization Statistics:\n'
    stats_text += f'Total iterations: {len(compliance_history)}\n'
    stats_text += f'Initial compliance: {initial_compliance:.6f}\n'
    stats_text += f'Final compliance: {final_compliance:.6f}\n'
    stats_text += f'Min compliance: {min_compliance:.6f}\n'
    stats_text += f'Max compliance: {max_compliance:.6f}\n'
    stats_text += f'Improvement: {((initial_compliance - final_compliance) / initial_compliance * 100):.2f}%'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # Highlight minimum point
    min_iteration = compliance_history.index(min_compliance) + 1
    plt.plot(min_iteration, min_compliance, 'ro', markersize=10, label=f'Minimum at iteration {min_iteration}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Create log plot for better visualization of compliance changes
    plt.figure(figsize=(12, 8))
    
    # Plot compliance vs iteration with log y-axis
    plt.semilogy(iterations, compliance_history, 'r-', linewidth=2, marker='s', markersize=4, alpha=0.7)
    
    # Customize the log plot
    plt.xlabel('Optimization Iteration', fontsize=14)
    plt.ylabel('Compliance (log scale)', fontsize=14)
    plt.title('Compliance Evolution During Optimization (Log Scale)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics for log plot
    stats_text_log = f'Log Scale Analysis:\n'
    stats_text_log += f'Total iterations: {len(compliance_history)}\n'
    stats_text_log += f'Initial compliance: {initial_compliance:.2e}\n'
    stats_text_log += f'Final compliance: {final_compliance:.2e}\n'
    stats_text_log += f'Min compliance: {min_compliance:.2e}\n'
    stats_text_log += f'Max compliance: {max_compliance:.2e}\n'
    stats_text_log += f'Improvement: {((initial_compliance - final_compliance) / initial_compliance * 100):.2f}%'
    
    plt.text(0.02, 0.98, stats_text_log, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             fontsize=10)
    
    # Highlight minimum point
    plt.semilogy(min_iteration, min_compliance, 'ro', markersize=10, label=f'Minimum at iteration {min_iteration}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*50)
    print("COMPLIANCE EVOLUTION ANALYSIS")
    print("="*50)
    print(f"Total optimization iterations: {len(compliance_history)}")
    print(f"Initial compliance: {initial_compliance:.8f}")
    print(f"Final compliance: {final_compliance:.8f}")
    print(f"Minimum compliance: {min_compliance:.8f} (at iteration {min_iteration})")
    print(f"Maximum compliance: {max_compliance:.8f}")
    print(f"Absolute improvement: {initial_compliance - final_compliance:.8f}")
    print(f"Relative improvement: {((initial_compliance - final_compliance) / initial_compliance * 100):.2f}%")
    
    # Check for convergence
    if len(compliance_history) >= 10:
        recent_improvement = compliance_history[-10:]
        improvement_rate = (recent_improvement[0] - recent_improvement[-1]) / recent_improvement[0] * 100
        print(f"Improvement in last 10 iterations: {improvement_rate:.4f}%")
        
        if improvement_rate < 0.1:
            print("✓ Optimization appears to have converged (very small improvement in recent iterations)")
        else:
            print("→ Optimization may still be improving (significant improvement in recent iterations)")

# Create compliance evolution plot
print("\n" + "="*50)
print("CREATING COMPLIANCE EVOLUTION PLOT")
print("="*50)
if compliance_history:
    plot_compliance_evolution(compliance_history)
else:
    print("No compliance history available for plotting.")

# Plot final material interface front (based on X_int construction)
print("\n" + "="*50)
print("PLOTTING FINAL MATERIAL INTERFACE FRONT (X_int)")
print("="*50)
try:
    # Mirror the construction used at line ~2393: X_int = g_sf * g_tsf * (1 - g_tf)
    g_sf_final_front = step_filter_element(g_solid_final, r, Nel_x, Nel_y, Delta_x, beta)
    g_tf_final_front = step_filter_element(g_type_final, r, Nel_x, Nel_y, Delta_x, beta)

    # Reproduce the "second filter scheme" used by multi_step_filter_element(), inline here
    # (cannot call it yet at this point in the file because it's defined later).
    eta_local = ETA_1 if 'ETA_1' in globals() else 0.5
    # First Helmholtz filter with radius r
    dens_1 = helmholtz_filter(g_type_final, r, Nel_x, Nel_y, Delta_x)
    # Projection
    dens_2 = (np.tanh(beta * eta_local) + np.tanh(beta * (dens_1 - eta_local))) / (
        np.tanh(beta * eta_local) + np.tanh(beta * (1.0 - eta_local))
    )
    # Second Helmholtz filter with radius 0.8*r
    dens_3 = helmholtz_filter(dens_2, 0.8 * r, Nel_x, Nel_y, Delta_x)
    # Second projection (match multi_step_filter_element): ETA_2 = 0.05
    ETA_2_local = 0.05
    dens_4 = (np.tanh(beta * ETA_2_local) + np.tanh(beta * (dens_3 - ETA_2_local))) / (
        np.tanh(beta * ETA_2_local) + np.tanh(beta * (1.0 - ETA_2_local))
    )
    g_tsf_final_front = dens_4

    X_int_final = g_sf_final_front * g_tsf_final_front * (1.0 - g_tf_final_front)

    # Map element vector -> (Nel_y, Nel_x) using the same element numbering as plot_design()
    X_int_matrix = np.zeros((Nel_y, Nel_x), dtype=np.float64)
    for i in range(Nel_y):
        for j in range(Nel_x):
            element_idx = j + i * (grid_size_x - 1)
            if element_idx < len(X_int_final):
                X_int_matrix[i, j] = X_int_final[element_idx]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    im = ax.imshow(X_int_matrix, cmap='inferno', aspect='equal', origin='upper')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('X_int (final)')
    ax.set_xlabel('X Direction')
    ax.set_ylabel('Y Direction')
    ax.set_title('Final material interface front (X_int)\nX_int = g_sf · g_tsf · (1 − g_tf)')
    ax.grid(True, alpha=0.2)

    # Optional contour to emphasize the front
    xint_max = float(np.max(X_int_matrix)) if X_int_matrix.size else 0.0
    if xint_max > 0:
        level = 0.5 * xint_max
        ax.contour(X_int_matrix, levels=[level], colors='cyan', linewidths=1.0, alpha=0.9)

    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not plot X_int front due to: {e}")


# Interface stress verification (same pipeline as mystressconstraint)
#_stress_verify_t0 = time.time()
#beta_verify = 12 + (beta_max - 8) * (1 - np.exp(-3 * 1))
#g_sigma, dg_sigma_dx1, dg_sigma_dx2 = mystressconstraint(g, beta_verify)
#print('dg_sigma_dx1[0]', dg_sigma_dx1[0])
#print('dg_sigma_dx2[0]', dg_sigma_dx2[0])
#print(f"Time taken for dsigma_dx: {time.time() - _stress_verify_t0:.2f} seconds")
