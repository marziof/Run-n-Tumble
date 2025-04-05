import numpy as np
from tqdm import tqdm

from Agent import Agent
from loss_functions import loss_diff

class Simulator:
    def __init__(self, agent, loss_function, update_tumble_prob, num_steps, X, y):
        self.agent = agent  # The agent being simulated
        self.loss_function = loss_function  # The loss function used for evaluation
        self.prob_update_rule = update_tumble_prob  # The rule to update the tumble probability
        self.num_steps = num_steps  # Number of steps in the simulation
        self.X = X  # Features for loss calculation
        self.y = y  # Target values for loss calculation

    def simu(self):
        for i in range(self.num_steps):
            self.update()  # Perform one step of the simulation

    def update(self):
        """This method will be overridden in subclasses with specific update rules."""
        raise NotImplementedError("Subclasses should implement this method.")


class BasicSimulator(Simulator):
    def __init__(self, agent, loss_function, prob_update_rule):
        super().__init__(agent, loss_function, prob_update_rule, num_steps=1000, X=None, y=None)

    def update(self):
        """Simulate run-and-tumble motion in N dimensions."""
        # Calculate loss before update
        loss_before = self.loss_function(self.agent.position, self.X, self.y)

        # Let the agent decide to run or tumble
        self.agent.update_position()

        # Calculate loss after proposed move
        loss_after = self.loss_function(self.agent.position, self.X, self.y)

        # Calculate the difference in loss
        delta_loss = loss_after - loss_before

        # Update tumble probability based on the loss difference
        self.agent.tumble_prob = self.prob_update_rule(self.agent.tumble_prob, delta_loss)

class CorrectStepsSimulator(Simulator):
    def __init__(self, agent, loss_function, prob_update_rule):
        super().__init__(agent, loss_function, prob_update_rule, num_steps=1000, X=None, y=None)

    def update(self):
        """Simulate run-and-tumble motion in N dimensions."""
        # Calculate loss before update
        loss_before = self.loss_function(self.agent.position, self.X, self.y)

        # Let the agent decide to run or tumble
        proposed_position, new_step_vec = self.agent.propose_step()

        # Calculate loss after proposed move
        loss_after = self.loss_function(proposed_position, self.X, self.y)

        # Accept or reject the step based on the loss
        if loss_after < loss_before:  # Accept step if it reduces loss
            self.agent.position = proposed_position
            self.agent.step_vec = new_step_vec

        # Calculate the difference in loss
        delta_loss = loss_after - loss_before

        # Update tumble probability based on the loss difference
        self.agent.tumble_prob = self.prob_update_rule(self.agent.tumble_prob, delta_loss)

