# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currentPos = currentGameState.getPacmanPosition()
        GhostPos = successorGameState.getGhostPositions()
        curFood = currentGameState.getFood()
        "*** YOUR CODE HERE ***"

        GhostFactor = 0

        for ghost in GhostPos:
          if(manhattanDistance(ghost, newPos) <= 1):
            GhostFactor = -3
          elif(GhostFactor != -3 and -1/manhattanDistance(ghost, newPos) < GhostFactor):
            GhostFactor = -1/manhattanDistance(ghost, newPos)

        FoodFactor = 0

        maxDis = (newFood.height**2 + newFood.width**2)**(0.5)
        minDis = maxDis
        for y in range(newFood.height):
          for x in range(newFood.width):
            if curFood[x][y] == True:
              dis = ((x-newPos[0])**2 + (y-newPos[1])**2)**(0.5)
              if(dis<minDis):
                minDis = dis

        if(minDis) == 0:
          FoodFactor = 2.5
        else:
          FoodFactor = 1/minDis

        ActionFactor = 0
        if(action == 'Stop'):
          ActionFactor = -0.1

        return FoodFactor + GhostFactor + ActionFactor

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def maxValue(self, depth, state, agent):
      if ( state.isWin() or state.isLose()):
        return  self.evaluationFunction(state)
      Actions = state.getLegalActions(agent)
      successorGameStates = [state.generateSuccessor(agent, action) for action in Actions]      
      nextAgent = (agent + 1) % state.getNumAgents()
      scores = [self.minValue(depth, nextState, nextAgent) for nextState in successorGameStates]
      return max(scores)

    def minValue(self, depth, state, agent):
      if ( state.isWin() or state.isLose()):
        return  self.evaluationFunction(state)
      Actions = state.getLegalActions(agent)
      successorGameStates = [ state.generateSuccessor(agent, action) for action in Actions ]
      nextAgent = (agent + 1) % state.getNumAgents()
      if (agent == state.getNumAgents() - 1 ):
          if (depth == self.depth): 
              scores = [self.evaluationFunction(nextState) for nextState in successorGameStates]
              return min(scores)
          else:
              scores = [self.maxValue(depth + 1, nextState, nextAgent) for nextState in successorGameStates]
              return min(scores)
      else:
          scores = [self.minValue(depth, nextState, nextAgent) for nextState in successorGameStates] 
          return min(scores)


    def minimax(self, state):
        NumAgents = state.getNumAgents()
        depth = 1
        # Collect legal moves and successor states
        Actions = state.getLegalActions()
        successorGameStates = [state.generateSuccessor(0, action) for action in Actions ]
        # Choose one of the best actions
        scores = [self.minValue(depth, state, 1) for state in successorGameStates]
        bestScore = max(scores)
        
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0]

        return Actions[chosenIndex] 

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState)





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.AlphaBeta(gameState)
        
    def minValue(self, depth, state, agent, alpha, beta):
        if(state.isLose() or state.isWin()):
          return self.evaluationFunction(state)
        Actions = state.getLegalActions(agent)
        nextAgent = (agent + 1) % state.getNumAgents()
        v = 1e50
        if(agent == state.getNumAgents()-1):
          if(depth == self.depth): 
            for action in Actions:
              nextState = state.generateSuccessor(agent, action)
              score = self.evaluationFunction(nextState)
              v = min(v, score)

              if(v < alpha):
                return v
              beta = min(beta, v)
          else:
            for action in Actions:
              nextState = state.generateSuccessor(agent, action)
              score = self.maxValue(depth+1, nextState, nextAgent, alpha, beta)
              v = min(v, score)
              if(v < alpha):
                return v
              beta = min([beta, v])
        else:
          for action in Actions:
            nextState = state.generateSuccessor(agent, action)
            score = self.minValue(depth, nextState, nextAgent, alpha, beta)
            v = min(v, score)
            if(v < alpha):
              return v
            beta = min([beta, v])
        return v

    def maxValue(self, depth, state, agent, alpha, beta):
      if(state.isLose() or state.isWin()):
        return self.evaluationFunction(state)

      Actions = state.getLegalActions(agent)
      nextAgent = (agent + 1) % state.getNumAgents()
      v = -1e50
      for action in Actions:
        nextState = state.generateSuccessor(agent, action)
        score = self.minValue(depth, nextState, nextAgent, alpha, beta)
        v = max( v, score )
        if(v > beta):
          return v
        alpha = max(alpha, v)
      return v

    def AlphaBeta(self, state):
      NumAgents = state.getNumAgents()
      depth = 1
      alpha = -1e50
      beta = 1e50
      # Collect legal moves and successor states
      Actions = state.getLegalActions()
      # Choose one of the best actions
      v = -1e50
      realAction = Actions[0]
      for action in Actions:
        nextState = state.generateSuccessor(0, action)
        if self.minValue(depth, nextState, 1, alpha, beta) > v :
          v = self.minValue(depth, nextState, 1, alpha, beta)
          realAction = action
        alpha = max(alpha, v)
        
      return realAction 

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxValue(self, depth, state):
      if depth == 0 or state.isWin() or state.isLose():
        return  self.evaluationFunction(state)
      else:
        successors = [state.generateSuccessor(0, action) for action in state.getLegalActions(0)]
        scores = [self.expectValue(depth-1, successor, 1) for successor in successors]
        return None if len(scores)==0 else max(scores)

    def expectValue(self, depth, state, agent):
      if depth == 0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      else:
        successors = [state.generateSuccessor(agent, action) for action in state.getLegalActions(agent)]
        scores = []
        if agent == state.getNumAgents()-1:
          scores = [self.maxValue(depth-1, successor) for successor in successors]
        else:
          scores = [self.expectValue(depth, successor, agent+1) for successor in successors]        
        expScore = (sum(scores)+0.0)/len(scores)
        return expScore

    def expectimax(self, depth, state):
      actions = state.getLegalActions(0)
      successors = [state.generateSuccessor(0, action) for action in actions]
      scores = [self.expectValue(depth-1, successor, 1) for successor in successors]
      bestScore = max(scores)
      bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
      chosenIndex = random.choice(bestIndices)
      
      return actions[chosenIndex]


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(self.depth*2, gameState)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

