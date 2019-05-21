import numpy as np
import gym
import random
import math

def buildArchitecture(networkArchitecture):
    network = []
    for layers in range(0,len(networkArchitecture)):
        network.append(np.zeros(networkArchitecture[layers]))
    return network

def setWeights(networkArchitecture):
    weights = []
    for layers in range(1,len(networkArchitecture)):
        weights.append(np.random.rand(networkArchitecture[layers],networkArchitecture[layers-1]))
    return weights
    
def feedForward(networkArchitecture, network, weights, inputs):
    network[0] = inputs
    for layer in range(0,len(networkArchitecture)-1):
        result = np.dot(weights[layer],network[layer])
        network[layer+1] = activationFunction(result)
    return(network)
    
def activationFunction(x):
    inter = np.exp(-(x))
    y = np.divide(1,np.add(1, inter))
    return y
    
 
def computeFitness(arch, weights, render = "off"):
    fit = []
    env = gym.make('CartPole-v0')
    for i in range(0,len(weights)):
        observation = env.reset()
        episodeFitness = 0
        for t in range(50):

            env.render()
                
            inputs = np.array(observation)
            observation, reward, done, info = env.step(getOutput(arch, weights[i], inputs))
            episodeFitness = episodeFitness + reward
            if done:
                fit.append(episodeFitness)
                
#                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    return(np.array(fit))
    

def getOutput(architecture,weights,inputs):
    model = buildArchitecture(architecture)
    network = feedForward(architecture, model, weights, inputs)
    output = network[len(architecture)-1]
    action = np.where(output == max(output))
    return np.asscalar(action[0])
 

# =============================================================================
# def mutation(arch, weights, rate = 0.2):
#     
#     for i in range(0,len(weights)):
#         for j in range(0,len(arch)-1):
#             product = np.prod(weights[i][j].shape)
#             for mutationProb in (0, round(product * rate)):
#                 row = random.randint(0, (weights[i][j].shape[0]-1))
#                 col = random.randint(0, (weights[i][j].shape[1]-1))
# #                print(row,col)
#                 weights[i][j][row][col] = random.random()
#     return weights
# =============================================================================

# =============================================================================
# def mutation(arch, weights, rate = 0.2):
#     for i in range(0,len(weights)):
#         for j in range(0,len(arch)-1):
#             layerShape = weights[i][j].shape
#             inter_w = np.reshape(weights[i][j],np.prod(layerShape))
#             for times in range(0,math.floor(np.prod(layerShape)*rate)):
#                 r = random.randint(0,np.prod(layerShape)-1)
#                 inter_w[r] = random.random()
#                 
#             inter_w = np.reshape(inter_w, layerShape)
#             weights[i][j] = inter_w
#     return weights
# 
# =============================================================================
def mutation(arch, fitness,weights, rate = 0.2):
   for times in range(0,math.floor(len(weights)*rate)):
       r1 = rws(fitness)
       p1 = weights[r1]
       for i in range(0,len(arch)-1):
           layerShape = p1[i].shape
           interLayer_1 = np.reshape(p1[i],np.prod(layerShape))
           r = random.randint(0,len(p1)-1)
           interLayer_1[r] = random.random()
           weights[r1][i] = np.reshape(interLayer_1, layerShape)
           
           return weights

def crossover(arch, weights, fitness, rate = 0.2):
    for times in range(0,math.floor(len(weights)*rate)):
        r1 = rws(fitness)
        r2 = rws(fitness)
        p1 = weights[r1]
        p2 = weights[r2]
        
        for i in range(0,len(arch)-1):
            layerShape = p1[i].shape
            interLayer_1 = np.reshape(p1[i],np.prod(layerShape))
            interLayer_2 = np.reshape(p2[i],np.prod(layerShape))
            r = random.randint(0,len(p1)-1)
            inter_3 = np.append(interLayer_1[0:r], interLayer_2[r:np.prod(layerShape)])
            inter_4 = np.append(interLayer_2[0:r], interLayer_1[r:np.prod(layerShape)])
            
            weights[r1][i] = np.reshape(inter_3, layerShape)
            weights[r2][i] = np.reshape(inter_4, layerShape)
                        
    return weights
    
    
def rws(fitness):
    sumFitness = sum(fitness)
    fitness = [x / sumFitness for x in fitness]
    cumFitness = np.cumsum(np.array(fitness))
    r = random.random()
    index = np.array(np.where(cumFitness<=r))
    if(index[0].size==0):
        index = random.randint(0,len(fitness)-1)
    return(np.max(index))
    
def getNewWeights(weights,fit):
    fit = np.array(fit)
    fitIndices = np.flip(np.argsort(fit))
    newWeights = [weights[i] for i in fitIndices]
    
    return(newWeights[0:popSize], fit[fitIndices])
    
arch = [4,2]
popSize = 10

weights = []
for i in range(0,popSize):
    weights.append(setWeights(arch))



for i in range(1,100):
    print(i)
    fit = computeFitness(arch, weights)
    mut_weights = mutation(arch,fit, weights,0.4)
    cv_weights= crossover(arch,weights,fit,0.2)
    weights = weights + mut_weights + cv_weights
    newFit = computeFitness(arch,weights)
    weights, fittest = getNewWeights(weights, newFit)
    print(fittest)
#    ind = rws(fit)
    
    

# write crossover and mutation function
# merge all function to create one function that will input weight and return error