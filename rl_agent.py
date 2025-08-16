import random
from typing import Optional

from kuimaze2 import Action, RLProblem, State
from kuimaze2.typing import Policy, QTable, VTable

T_MAX = 200  # Max steps in episode
MAX_EPISODES = 1000  # Max number of episodes


class RLAgent:
    """Implementation of Q-learning algorithM for fiding optimal policy in an environment"""

    def __init__(
        self,
        env: RLProblem,
        gamma: float = 0.9,
        alpha: float = 0.1,
    ):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.current_epsilon = self.epsilon
        self.init_q_table()

    def init_q_table(self) -> None:
        """Create and initialize the Q-table with zeros"""
        self.q_table = {
            state: {action: 0.0 for action in self.env.get_action_space()}
            for state in self.env.get_states()
        }

    def get_values(self) -> VTable:
        """Return the state values derived from the q-table"""
        return {
            state: max(q_values.values()) 
            for state, q_values in self.q_table.items()
        }

    def get_best_action_for_state(self, state: State) -> Action:
        """
        Get the best action for a given state based on current Q-values.
        
        State: The state to get best action for 
        
        Return: The action with the highest Q-value for given state
        """
        q_values = self.q_table[state]

        return max(q_values, key=q_values.get)
    
    def choose_action(self, state: State) -> Action:
        """
        Choose an action using epsilon-greedy strategy
        
        State: The state to choose action for
        
        Return: The chosen action
        """

        # Explore with probability epsilon
        if random.random() < self.current_epsilon:
            return random.choice(self.env.get_action_space())
        # Exploit with probability 1 - epsilon
        else:
            return self.get_best_action_for_state(state)
        
    def update_q_value(self, state: State, action: Action, reward: float, next_state: State) -> None:
        """
        Update the Q-value
        State: The state
        Action: The action
        Reward: The reward 
        Next_state: The next state
        """
        # Current Q-value
        current_q = self.q_table[state][action]

        # Max Q-value for the next state
        next_max_q = 0.0
        if next_state is not None:
            next_max_q = max(self.q_table[next_state].values())

        # Update TD target 
        td_target = reward + self.gamma * next_max_q

        td_error = td_target - current_q

        # Update Q-value
        self.q_table[state][action] = current_q + self.alpha * td_error


    def render(
        self,
        current_state: Optional[State] = None,
        action: Optional[Action] = None,
        values: Optional[VTable] = None,
        q_values: Optional[QTable] = None,
        policy: Optional[Policy] = None,
        *args,
        **kwargs,
    ) -> None:
        """Visualize the state of the algorithm"""
        values = values or self.get_values()
        q_values = q_values or self.q_table
        # State values will be displayed in the squares
        sq_texts = (
            {state: f"{value:.2f}" for state, value in values.items()} if values else {}
        )
        # State-action value will be displayed in the triangles
        tr_texts = {
            (state, action): f"{value:.2f}"
            for state, action_values in q_values.items()
            for action, value in action_values.items()
        }
        # If policy is given, it will be displayed in the middle
        # of the squares in the "triangular" view
        actions = {}
        if policy:
            actions = {state: str(action) for state, action in policy.items()}
        # The current state and chosen action will be displayed as an arrow
        state_action = (current_state, action)
        if current_state is None or action is None:
            state_action = None
        self.env.render(
            *args,
            square_texts=sq_texts,
            square_colors=values,
            triangle_texts=tr_texts,
            triangle_colors=q_values,
            middle_texts=actions,
            state_action_arrow=state_action,
            wait=True,
            **kwargs,
        )

    def extract_policy(self) -> Policy:
        """
        Extract policy from Q-values
        Returns: A policy that maps states to actions
        """
        policy = {
            state: self.get_best_action_for_state(state)
            for state in self.env.get_states()
        }
        return policy
    
    def run_episode(self, render: bool = False) -> float:
        """
        Run a single episode of Q-learning
        Render: Whether to render the episode
        Returns: The total reward of the episode
        """
        total_reward = 0
        episode_finished = False
        t = 0

        # Reset the environment and get the initial state
        state = self.env.reset()
        path = [state]
        
        while not episode_finished and t < T_MAX:
            t += 1

            # Choose an action using epsilon-greedy strategy
            action = self.choose_action(state)
            
            # Take the action and observe the next state and reward
            next_state, reward, episode_finished = self.env.step(action)
            total_reward += reward
            
            if next_state is not None:
                path.append(next_state)

            # Update Q-value using q-learning update rule
            self.update_q_value(state, action, reward, next_state)
            
            # Render if requested
            if render:
                policy = self.extract_policy()
                self.render(current_state=state, action=action, path=path, policy=policy)

            # Move to the next state
            state = next_state

        return total_reward

    def learn_policy(self) -> Policy:
        """Run Q-learning algoritm to learn a policy"""
        
        best_reward = float("-inf")
        episode_rewards = []

        # Run multiple episodes to learn 
        for episode  in range(MAX_EPISODES):
            self.current_epsilon = max(self.epsilon_min, 
            self.current_epsilon * self.epsilon_decay)

            # Show the current epsilon
            render = (episode  % 100 == 0)

            # Run a single episode of Q-learning
            reward = self.run_episode(render=render)
            episode_rewards.append(reward)

            if reward > best_reward:
                best_reward = reward

            if episode % 100 == 0:
                avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
                print(f"Episode {episode}/{MAX_EPISODES}, Epsilon: {self.current_epsilon:.3f}, "
                f"Avg Reward: {avg_reward:.2f}, Best Reward: {best_reward:.2f}")
            
            if len(episode_rewards) >= 100:
                recent_rewards = episode_rewards[-100:]
                if all(r == recent_rewards[0] for r in recent_rewards) and episode > 200:
                    print(f"Converged after {episode} episodes!")
                    break
                
            # Extract and return the learned policy
            final_policy = self.extract_policy()
            print(f"Training completed. Final policy found after {episode+1} episodes.")

        return final_policy

if __name__ == "__main__":
    from kuimaze2 import Map
    from kuimaze2.map_image import map_from_image

    MAP = """
    ...G
    .#.D
    S...
    """
    map_obj = Map.from_string(MAP)
    # map = map_from_image("./maps/normal/normal3.png")
    env = RLProblem(
        map_obj,
        action_probs=dict(forward=0.8, left=0.1, right=0.1, backward=0.0),
        graphics=True,
    )

    agent = RLAgent(env, gamma=0.9, alpha=0.1)
    policy = agent.learn_policy()
    print("Policy found:", policy)
    agent.render(policy=policy, use_keyboard=True)
