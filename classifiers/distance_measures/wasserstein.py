import numpy as np
import scipy.optimize
# from distance.wasserstein_GitHub import getEMD

# Constraints
def positivity(f):
    '''
    Constraint 1:
    Ensures flow moves from source to target
    '''
    return f


def fromSrc(f, wp, i, shape):
    """
    Constraint 2:
    Limits supply for source according to weight
    """
    fr = np.reshape(f, shape)
    f_sumColi = np.sum(fr[i, :])
    return wp[i] - f_sumColi


def toTgt(f, wq, j, shape):
    """
    Constraint 3:
    Limits demand for target according to weight
    """
    fr = np.reshape(f, shape)
    f_sumRowj = np.sum(fr[:, j])
    return wq[j] - f_sumRowj


def maximiseTotalFlow(f, wp, wq):
    """
    Constraint 4:
    Forces maximum supply to move from source to target
    """
    return f.sum() - np.minimum(wp.sum(), wq.sum())


# Objective function
def flow(f, D):
    """
    The objective function
    The flow represents the amount of goods to be moved
    from source to target
    """
    f = np.reshape(f, D.shape)
    return (f * D).sum()


# Distance
def groundDistance(x1, x2, norm=2):
    """
    L-norm distance
    Default norm = 2
    """
    return np.linalg.norm(x1 - x2, norm)


# Distance matrix
def getDistMatrix(s1, s2, norm=2):
    """
    Computes the distance matrix between the source
    and target distributions.
    The ground distance is using the L-norm (default L2 norm)
    """
    # Slow method
    # rows = s1 feature length
    # cols = s2 feature length
    numFeats1 = s1.shape[0]
    numFeats2 = s2.shape[0]
    distMatrix = np.zeros((numFeats1, numFeats2))

    for i in range(0, numFeats1):
        for j in range(0, numFeats2):
            distMatrix[i, j] = groundDistance(s1[i], s2[j], norm)

    # Fast method (requires scipy.spatial)
    # import scipy.spatial
    # distMatrix = scipy.spatial.distance.cdist(s1, s2)

    return distMatrix


# Flow matrix
def getFlowMatrix(P, Q, D):
    """
    Computes the flow matrix between P and Q
    """
    numFeats1 = P[0].shape[0]
    numFeats2 = Q[0].shape[0]
    shape = (numFeats1, numFeats2)

    # Constraints
    cons1 = [{'type': 'ineq', 'fun': positivity},
             {'type': 'eq', 'fun': maximiseTotalFlow, 'args': (P[1], Q[1],)}]

    cons2 = [{'type': 'ineq', 'fun': fromSrc, 'args': (P[1], i, shape,)} for i in range(numFeats1)]
    cons3 = [{'type': 'ineq', 'fun': toTgt, 'args': (Q[1], j, shape,)} for j in range(numFeats2)]

    cons = cons1 + cons2 + cons3

    # Solve for F (solve transportation problem)
    F_guess = np.zeros(D.shape)
    F = scipy.optimize.minimize(flow, F_guess, args=(D,), constraints=cons)
    F = np.reshape(F.x, (numFeats1, numFeats2))

    return F


# Normalised EMD
def EMD(F, D):
    """
    EMD formula, normalised by the flow
    """
    return (F * D).sum() / F.sum()


# Runs EMD program
def getEMD(P, Q, norm=2):
    """
    EMD computes the Earth Mover's Distance between
    the distributions P and Q

    P and Q are of shape (2,N)

    Where the first row are the set of N features
    The second row are the corresponding set of N weights

    The norm defines the L-norm for the ground distance
    Default is the Euclidean norm (norm = 2)
    """

    D = getDistMatrix(P[0], Q[0], norm)
    F = getFlowMatrix(P, Q, D)

    return EMD(F, D)

class Distance_Wasserstein :
    def __init__(self, vec1, vec2):
        vec1=np.array([vec1])
        vec2=np.array([vec2])
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            raise ValueError("vec1 and vec2 must be of the np.ndarray type!")

        if vec1.shape[0] != vec2.shape[0]:
            raise ValueError("vec1 and vec2 must be of the same length!")

        self.vec1 = vec1
        self.vec2 = vec2

    def get_distance_between_two_vecs(self):
        weights1=np.array([1])
        weights2=np.array([1])
        featureVec1=(self.vec1,weights1)
        featureVec2=(self.vec2,weights2)
        distance=getEMD(featureVec1,featureVec2)
        print(distance)
        return distance

if __name__=='__main__':
    sample1=Distance_Wasserstein([1,2,3],[3,4,5])
    sample1.get_distance_between_two_vecs()
