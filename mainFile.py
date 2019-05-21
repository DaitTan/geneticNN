import numpy as np
import gym
import random
import math
import matplotlib as plt

def buildArchitecture(networkArchitecture):
    network = []
    for layers in range(0,len(networkArchitecture)):
        network.append(np.zeros(networkArchitecture[layers]))
    return network

def setWeights(networkArchitecture):
    weights = []
    for layers in range(1,len(networkArchitecture)):
        weights.append(np.random.rand(networkArchitecture[layers],networkArchitecture[layers-1]) - 0.5)
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
        for t in range(10000):

#            env.render()
                
            inputs = np.array(observation)
            observation, reward, done, info = env.step(getOutput(arch, weights[i], inputs))
            episodeFitness = episodeFitness + reward
            if done:
#                fit.append(episodeFitness)
#                print("Episode finished after {} timesteps".format(t+1))
                break
        fit.append(episodeFitness)
    env.close()
    return(np.array(fit))
    

def getOutput(architecture,weights,inputs):
    model = buildArchitecture(architecture)
    network = feedForward(architecture, model, weights, inputs)
    output = network[len(architecture)-1]
    action = np.where(output == max(output))
    return np.asscalar(action[0][0])
  
  
def mutation(arch, fitness,weights, rate = 0.2):
  newWeights = []
  for times in range(0,math.floor(len(weights)*rate)):
    r1 = random.randint(0,len(fitness)-1)
    p = weights[r1]
    for i in range(0,len(arch)-1):
      layerShape = p[i].shape
      interLayer_1 = np.reshape(p[i],np.prod(layerShape))
      r = random.randint(0,len(interLayer_1)-1)
      interLayer_1[r] = interLayer_1[r] + (random.random() - 0.5)
      #           interLayer_1[r] = interLayer_1[r] + 100
      p[i] = np.reshape(interLayer_1, layerShape)        
    newWeights.append(p)
  return newWeights

def crossover(arch, weights, fitness, rate = 0.2):
    newWeights = []
    for times in range(0,math.floor(len(weights)*rate)):
        r1 = random.randint(0,len(fitness)-1)
        r2 = random.randint(0,len(fitness)-1)
        p1 = weights[r1]
        p2 = weights[r2]
        
        for i in range(0,len(arch)-1):
            layerShape = p1[i].shape
            interLayer_1 = np.reshape(p1[i],np.prod(layerShape))
            interLayer_2 = np.reshape(p2[i],np.prod(layerShape))
            r = random.randint(0,len(p1)-1)
            inter_3 = np.append(interLayer_1[0:r], interLayer_2[r:np.prod(layerShape)])
            inter_4 = np.append(interLayer_2[0:r], interLayer_1[r:np.prod(layerShape)])
            p1[i] = np.reshape(inter_3, layerShape)
            p2[i] = np.reshape(inter_4, layerShape)
            
        newWeights.append(p1) 
        newWeights.append(p2)
        
    return newWeights

def selection(arch, weights, fitness, rate = 0.2):
    newWeights = []
    for times in range(0,math.floor(len(weights)*rate)):
        r1 = random.randint(0,len(fitness)-1)
        r2 = random.randint(0,len(fitness)-1)
        p1 = weights[r1]
        p2 = weights[r2]
        fit1 = computeFitness(arch, [p1])
        fit2 = computeFitness(arch, [p2])
        
        if fit1 >= fit2:
            newWeights.append(p1)
        else:
            newWeights.append(p2)
            
    return newWeights
    
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
    newWeights = [weights[fitIndices[i]] for i in range(0, len(fitIndices))] 
#    newWeights = inter_weights[]
    return(newWeights[0:popSize], fit[fitIndices])

def getBestWeights(arch, weights):
  fitness = computeFitness(arch, weights)
  weightIndex = np.where(fitness == max(fitness))
  newWeights = [weights[weightIndex[0][i]] for i in range(0, len(weightIndex[0]))]
  optimalWeight = []
  for layers in range(0, len(arch)-1):
    inter = [newWeights[i][layers] for i in range(0,len(newWeights))]
    optimalWeight.append(sum(inter) / len(newWeights))
  return optimalWeight
    
  
def getOptimalWeight(arch, popSize, maxIterations, maxFitness):  
    
    weights = []
    for i in range(0,popSize):
        weights.append(setWeights(arch))
    
    
    max_fitness = []
    flag = 0
    index = 0
    for i in range(1,maxIterations):
        print(i)
        fit = computeFitness(arch, weights)
        sel_weights = selection(arch, weights, fit, 0.5)
        mut_weights = mutation(arch,fit, weights,0.5)
        cv_weights= crossover(arch,weights,fit,0.5)
        newWeights = sel_weights + mut_weights + cv_weights
        newFit = computeFitness(arch,newWeights)
        weights = 0
        weights, fittest = getNewWeights(newWeights, newFit)
        newWeights = 0
        print(max(fittest))
        max_fitness.append(max(fittest))
        
        if max(fittest)==maxFitness:
          if flag == 0:
            flag = 1
            index = i
          elif flag == 1 and i == index+1:
            print("Done")
            break
        else:
          flag = 0
          index = 0
    
    
    plt.plot(range(1, len(max_fitness)+1), max_fitness, 'r--')
    
    plt.show()
    
    optimalWeight = getBestWeights(arch, weights)
    return optimalWeight


if __name__ == "__main__":
    popSize = 2000
    arch = [4,5,3,2]
    maxIterations = 1000
    maxFitness = 200.0
    optimalWeight = getOptimalWeight(arch, popSize, maxIterations, maxFitness)
    np.save("optimalWeights", optimalWeight)

