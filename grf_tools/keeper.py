import torch
import torch.nn as nn 


#list of actions
KEEPER_ACTIONS = [
     "idle", "left", "top_left", "top", "top_right", "right", "bottom_right", "bottom", "bottom_left", #movement
     "sprint", #speed
     "short_pass", "high_pass", "long_pass", #distribution of ball
     "keeper_rush" #come off the line
     ]

#is there a shot coming at me or not? 
def shot_incoming(obs):
        #ball position from where it's shot
        x = obs['ball'][0]
        y = obs['ball'][1]

        #change in ball position from where it's shot
        dy = obs['ball_direction'][1]
        dx = obs['ball_direction'][0]

        if dx >= 0: #moving away from goal
            return False
        
        #time it takes to reach the goal line
        t = -(x+1) / dx

        #balls y position when it reaches goal line
        y_line = y + dy *t

        shot_conditions = [
                t < 15, #arriving within 1.5 s?
                abs(y_line) < 0.06 #heading between the post?
        ]

        return all(shot_conditions)

class KeeperNet(nn.Module):
     def __init__ (self):
          super().__init__()
          self.layer1 = nn.Linear(15,64)
          self.layer2 = nn.Linear(64,64)
          self.move_dir = nn.Linear(64, 17) # A: idle + 16 directions
          self.speed = nn.Linear(64, 3) # B: walk, sprint, close control
          self.ball_action = nn.Linear(64, 8) # C: none, short, high, long, shot, curl, chip, through ball
          self.defense = nn.Linear(64, 5) # D: none, pressure, team pressure, sliding, keeper rush
          self.kick_power = nn.Linear(64, 2) # E: two numbers that shape a range of powers (explained in step 2)
          self.critic = nn.Linear(64,1)
          self.bend = nn.ReLU()
     
     def forward(self, x):
          x = self.bend(self.layer1(x))
          x = self.bend(self.layer2(x))

          power = torch.nn.functional.softplus(self.kick_power(x)) + 1 #always > 1

          heads = (self.move_dir(x), self.speed(x), self.ball_action(x), self.defense(x), power)
          return heads, self.critic(x)

if __name__ == "__main__":
    #shot_incoming tests
    obs = {'ball': [-0.6, 0.0, 0.1], 'ball_direction': [-0.05, 0.0, 0.0]}
    print(shot_incoming(obs))
    obs = {'ball': [-0.6, 0.0, 0.1], 'ball_direction': [0.05, 0.0, 0.0]}
    print(shot_incoming(obs))
    obs = {'ball': [0.5, 0.0, 0.1], 'ball_direction': [-0.02, 0.0, 0.0]}
    print(shot_incoming(obs))
    obs = {'ball': [-0.6, 0.0, 0.1], 'ball_direction': [-0.05, 0.03, 0.0]}
    print(shot_incoming(obs))

    #network test
    net = KeeperNet()

    situation = torch.rand(15) #create random situation 
    heads, value = net(situation) #from the situation get the scores and value from the actor and critic

    for head in heads:
         print(len(head))

    print(value)