import numpy as np
import gym

def buildArchitecture(networkArchitecture):
    network = []
    for layers in range(0,len(networkArchitecture)):
        network.append(np.zeros(networkArchitecture[layers]))
    return network

  
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

def getOutput(architecture,weights,inputs):
    model = buildArchitecture(architecture)
    network = feedForward(architecture, model, weights, inputs)
    output = network[len(architecture)-1]
    action = np.where(output == max(output))
    return np.asscalar(action[0])
 
    
def computeFitness(arch, weights, render = "off"):
    fit = []
    env = gym.make('CartPole-v0')
    for i in range(0,len(weights)):
        observation = env.reset()
        episodeFitness = 0
        for t in range(10000):
            if render == "on":
                env.render()
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



if __name__ == "__main__":
    
    arch = [4,5,3,2]
    
    weight = np.load("optimalWeights.npy", allow_pickle=True)
    optimalWeight = [weight[i] for i in range(0, len(arch)-1)]
    fitness = computeFitness(arch, [optimalWeight], "on")
    print(fitness)
    
