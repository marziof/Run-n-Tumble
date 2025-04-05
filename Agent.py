import numpy as np

class Agent:
    def __init__(self, position, step_vec, tumble_prob):
        self.position = np.array(position)
        self.step_vec = np.array(step_vec)
        self.tumble_prob = tumble_prob
        self.positions = [self.position.copy()]

    def run(self):
        """Continue moving in the same direction."""
        return self.position + self.step_vec

    def tumble(self):
        """Change direction randomly and move."""
        new_step_vec = np.random.randn(*self.position.shape)  # New random direction
        new_position = self.position + new_step_vec
        return new_position, new_step_vec

    def update_position(self):
        """Run or tumble based on probability, updating position and step vector accordingly."""
        if np.random.rand() < self.tumble_prob:
            self.position, self.step_vec = self.tumble()  # Update both
        else:
            self.position = self.run()  # Keep same step vector
        self.positions.append(self.position)

    def propose_step(self):
        """Generate a proposed new position based on the step vector."""
        new_step_vector = self.step_vec  # Default: keep running

        if np.random.rand() < self.tumble_prob:
            # Tumble: Change direction randomly
            new_step_vector = np.random.randn(*self.position.shape)
        
        proposed_position = self.position + new_step_vector
        return proposed_position, new_step_vector  # Return both step vector & position
