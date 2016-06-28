"""
Implements algorithms detailed in [1]

References:
[1]: Bubeck, Sebastien, et al. "X-armed bandits." Journal of Machine Learning Research 12.May (2011): 1655-1695.
"""

import math
import random
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Hoo:
    """ Basic HOO descrbed in X-armed bandits [1]
    Assume that search space is in to D-dimensional hypercube [0,1]^D
    Refer to Example 1 of [1] for details about the parameter selection and tree-splitting
    """
    def __init__(self, D, v1, p, fun):
        """ Constructs Hoo empty search tree 
        @param D - dimension of hypercube
        @param v1 - tunable parameters --> will affect regret bounds (if it even converges to 0!) 
        @param p - tunable parameters --> will affect regret bounds (if it even converges to 0!)
        """
        self.D = D
        self.v1 = v1
        self.p = p
        self.fun = fun
        self.root = Node(np.tile([0.0,1.0],[D,1]), None, h=0,nextCut=0)
        self.n = 0

    def Pull(self):
        """ Pull arm
        """
        self.n += 1
        
        # Find node to pull
        curNode = self.root
        while True:
            # Select best arm
            if curNode.children_B[0] >= curNode.children_B[1]: branch = 0
            else: branch = 1
            
            nextNode = curNode.children[branch]
                        
            if nextNode == None: break
            curNode = nextNode

        # Play arm and get reward
        pulled =  random.uniform(curNode.rng[0,0], curNode.rng[0,1])
        reward = self.fun(pulled)
        print curNode.rng
        print pulled, reward
        
        # Create child node 
        curNode.children[branch] = curNode.MakeNextNode(branch, self.D)
 
        # Update all counts and means [T and mu]
        c = curNode.children[branch]
        while not c == None:
            c.T += 1
            c.mu = (1.0-1.0/c.T)*c.mu + reward/c.T
            c=c.parent
        
        # Update all upper bounds [U]
        stack = [self.root]
        opposite = []
        while len(stack) > 0:
            c = stack.pop(0)
            opposite.append(c)
            c.U = c.mu + math.sqrt((2 * math.log(self.n)/c.T)) + self.v1 * self.p ** c.h
            for z in c.children:
                if z == None: continue
                stack.append(z)

        # Bottom up 
        while len(opposite) > 0:
            c = opposite.pop()
            for g in xrange(2):
                if not c.children[g] == None: c.children_B[g] = c.children[g].B
            c.B = min(c.U, max(c.children_B))
        
        return pulled, reward


class Node:
    def __init__(self, rng, parent, h=0,nextCut=0):
        self.B = float('inf')
        self.rng = rng
        self.nextCut = nextCut
        self.children_B = [float('inf'), float('inf')]
        self.children = [None, None]
        self.parent = parent
        self.T = 0
        self.mu = 0
        self.U = 0
        self.h = h
        
    def MakeNextNode(self, pos, D):
        nextRng = copy.deepcopy(self.rng)
        if pos == 0:
            #print self.nextCut
            nextRng[self.nextCut, 0] = (nextRng[self.nextCut, 0] + nextRng[self.nextCut, 1])/2.0
        else:       

            #print self.nextCut
            nextRng[self.nextCut, 1] = (nextRng[self.nextCut, 0] + nextRng[self.nextCut, 1])/2.0
        nextNode = Node(nextRng, self, h=self.h+1, nextCut = (self.nextCut+1)%D)
        self.children[pos] = nextNode
        return nextNode
        
def basic_params():
    a = 2.0 # square error norm
    D = 1 # 
    p = 2.0**(-a/D)
    b = 1.0
    v1 = b * (2 * math.sqrt(D)) ** a 
    objfun = lambda x: (1.0/2.0) * (math.sin(13*x)*math.sin(27*x)+1)
    noise = lambda : np.random.binomial(1,0.5)
    fullfun = lambda x: objfun(x) * noise()
    xMaxEReward=0.867526/2.0
    xMax=0.975599
    
    return a,D,p,b,v1,fullfun, xMaxEReward, xMax



if __name__ == '__main__': 

    [a,D,p,b,v1,fullfun, xMaxEReward, xMax] = basic_params()

    #plt.plot(np.arange(0,1,0.01), [fun(x) for x in np.arange(0,1,0.01)])
    #plt.show()
    h = Hoo(D, v1, p, fullfun)
    arms_pulled = []
    rewards_recieved = []
    average_regret = []
    for k in xrange(5000):
        pulled, reward = h.Pull()
        arms_pulled.append(pulled)
        regret = xMaxEReward-reward
        if k == 0: average_regret.append(regret)
        else: average_regret.append((average_regret[-1] * (k-1) + regret)/k)
    plt.plot(average_regret)
    plt.show()
