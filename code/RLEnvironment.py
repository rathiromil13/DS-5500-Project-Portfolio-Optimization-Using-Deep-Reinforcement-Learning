import numpy as np
import pandas as pd

class RLEnv():
    
    """
    Using this class we will render an RL trading environment
    
    PortfolioValue: value of the finance portfolio
    TransCost: Transaction cost that has to be paid by the agent to execute the action
    ReturnRate: Percentage change in portfolio value
    WindowSize: Number of trading periods to be considered
    SplitSize: % of data to be used for training dataset, rest will be used for test dataset
    """
 
    def __init__(self, Path, PortfolioValue, TransCost, ReturnRate, WindowSize, TrainTestSplit):
        # Here, we need to initialize values like portfolio values, transaction costs etc.
        
        # Loading dataset in Path to Dataset variable
        self.Dataset = np.load(Path)
        
        # Number of stocks and associated values like Close, High, Low
        self.NumStocks = self.Dataset.shape[1]
        self.NumValues = self.Dataset.shape[0]
        
        # Initializing parameter
        self.PortfolioValue = PortfolioValue
        self.TransCost = TransCost
        self.ReturnRate = ReturnRate
        self.WindowSize = WindowSize
        self.Done = False
        
        # Initiate state, action
        self.state = None
        self.TimeLength = None
        self.Terminate = False
        
        # Termination cutoff
        self.TerminateRows = int((self.Dataset.shape[2] - self.WindowSize) * TrainTestSplit)
        
    def UpdatedOpenValues(self, T):
        # This function provides the 
        return np.array([1+self.ReturnRate]+self.Dataset[-1,:,T].tolist())
    
    def InputTensor(self, Tensor, T):
        return Tensor[: , : , T - self.WindowSize:T]
    
    def ResetEnvironment(self, InitWeight, InitPortfolio, T):
        self.state= (self.InputTensor(self.Dataset, self.WindowSize) , InitWeight , InitPortfolio)
        self.TimeLength = self.WindowSize + T
        self.Done = False
        
        return self.state, self.Done
    
    def Step(self, Action):
        """
        Here, we get the action that needs to be performed at time step t, so, we get new weight vector,
        reward function, updated value of portfolio
        
        We get the input tensor values for timestep t and for a given window size for each of the stocks 
        
        State usually contains a input tensor, weight vector, portfolio vector
        """
        
        # Obtain input tensor
        Dataset = self.InputTensor(self.Dataset, int(self.TimeLength))
    
    
        # Current state values - current weight vector and portfolio vector
        weight_vector_old = self.state[1]
        portfolio_value_old = self.state[2]
        
        # Update the vector with opening values
        NewOpenValues = self.UpdatedOpenValues(int(self.TimeLength))
        
        # Trading agent here provides new actions, that is new weight vector using which new 
        # allocations have to be done
        WeightAllocation = Action
        PortfolioAllocation = portfolio_value_old
        
        # While reallocating portfolios using weights we will have to account for transaction or 
        # commision rates
        TransactionCost = PortfolioAllocation * self.TransCost * np.linalg.norm((WeightAllocation-weight_vector_old),ord = 1)
        
        # Inorder to find the new weight vector we need to obtain the value of present portfolio
        # So as to obtain the value vector for each stock we need to multiply the portfolio value with the weight vector        
        # Every time a stock portfolio is updated there is an additional transaction cost that incurs on the portfolio 
        # value
        ValueAfterTransaction = (PortfolioAllocation * WeightAllocation) - np.array([TransactionCost]+ [0] * self.NumStocks)
        
        # So the valueaftertransaction has cost deducted stock values for the previous day, when we multiply this vector
        # with the latest open values
        NewValueofStocks = ValueAfterTransaction * NewOpenValues
        
        # When we sum the stock prices of individual stock prices, we get the value of portfolio
        NewPortfolioValue = np.sum(NewValueofStocks)
        
        # Inorder to obtain the new weight vector, we divide individual stock prices with total portfolio value
        NewWeightVector = NewValueofStocks / NewPortfolioValue
        
        # After each timestep, value of the portfolio either decreases or increases depending on how the agent 
        # performs
        RewardValue = (NewPortfolioValue - portfolio_value_old) / (portfolio_value_old)

        self.TimeLength = self.TimeLength + 1
        
        # Using the computed values till now we can create new state
        self.state = (self.InputTensor(self.Dataset, int(self.TimeLength)), NewWeightVector, NewPortfolioValue)
        
        # Here, we have to compute termination criteria, when to terminate the step process
        if self.TimeLength >= self.TerminateRows:
            self.Done = True
            
        return self.state, RewardValue, self.Done