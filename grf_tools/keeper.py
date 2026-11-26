import torch
import torch.nn as nn 
import math

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

#"when they shoot, the keeper is within 1 m of the line from the ball to the centre of our goal."
def check_shot_position(raw_obs, slot, instruction):

     if shot_incoming(raw_obs) == True:

          #where the keeper is in game units
          keeper_x = raw_obs["left_team"][0][0] 
          keeper_y = raw_obs["left_team"][0][1]
          #where the ball is in game units
          ball_x = raw_obs["ball"][0]
          ball_y = raw_obs["ball"][1]

          #ball location conversions
          a_x = (ball_x + 1) * 54.4 
          a_y = ball_y * 83.6
          #length of ball
          a = math.hypot(a_x, a_y)
     
          #keeper location conversions 
          b_x = (keeper_x+1) * 54.4
          b_y = keeper_y * 83.6
          if a == 0:
               return False
          distance = abs(a_x*b_y - a_y*b_x) / a
               
          return distance < 1

     if shot_incoming(raw_obs) == False:
          return False

#keeper reward function
def keeper_reward(prev_obs, obs, duty_ok, memory):
     reward = 0

     keeper_x = obs["left_team"][0][0] #goal line x = -1
     keeper_y = obs["left_team"][0][1] #between goal post
     score_now = obs["score"][1] #opponent's score current
     prev_score = prev_obs["score"][1] #opponent's score one step before
     prev_ball_owner = prev_obs["ball_owned_team"] #0 = us, 1 = them, -1 = nobody
     ball_owner_now = obs["ball_owned_team"] #0 = us, 1 = them, -1 = nobody
     who_owned = prev_obs["ball_owned_player"] #0 = the keeper
     who_owns = obs["ball_owned_player"] #0 = the keeper, -1 = nobody
     
     #inside the posts
     if keeper_x < -0.95 and abs(keeper_y) < 0.044:
          reward += 0.001

     #opponents score
     if score_now > prev_score:
          reward -= 1

     #following duty from manager
     if duty_ok == True:
          reward += 0.005

     #in position when shot comes
     if shot_incoming(obs) and not shot_incoming(prev_obs) and keeper_x < -0.95 and abs(keeper_y) < 0.044:
          reward += 0.8

     #does keeper pass? 
     if ball_owner_now == -1 and who_owned == 0 and prev_ball_owner == 0:
          memory["pass_in_air"] = True

     #keeper passed started and it landed
     if memory["pass_in_air"] and ball_owner_now != -1:
          if ball_owner_now == 1: #if it goes to opposing team
               reward -= 0.8

          elif ball_owner_now == 0 and who_owns != 0: #if it goes to our team and not to keeper
               reward += 0.6

          memory["pass_in_air"] = False

     return reward

class KeeperNet(nn.Module):
     def __init__ (self):
          super().__init__()
          self.layer1 = nn.Linear(960,256)
          self.layer2 = nn.Linear(256,256)
          self.move_dir = nn.Linear(256, 17) # A: idle + 16 directions
          self.speed = nn.Linear(256, 3) # B: walk, sprint, close control
          self.ball_action = nn.Linear(256, 8) # C: none, short, high, long, shot, curl, chip, through ball
          self.defense = nn.Linear(256, 5) # D: none, pressure, team pressure, sliding, keeper rush
          self.kick_power = nn.Linear(256, 2) # E: two numbers that shape a range of powers (explained in step 2)
          self.critic = nn.Linear(256,1)
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

    #check_shot_position (B4) tests
    obs = {'left_team': [[-1.0, 0.0]], 'ball': [-0.8, 0.0, 0.1], 'ball_direction': [-0.02, 0.0, 0.0]}
    print("B4 on the line (expect True):", check_shot_position(obs, 0, None))
    obs = {'left_team': [[-1.0, 0.03]], 'ball': [-0.8, 0.0, 0.1], 'ball_direction': [-0.02, 0.0, 0.0]}
    print("B4 2.5 m off the line (expect False):", check_shot_position(obs, 0, None))
    obs = {'left_team': [[-1.0, 0.0]], 'ball': [-0.8, 0.0, 0.1], 'ball_direction': [0.02, 0.0, 0.0]}
    print("B4 no shot coming (expect False):", check_shot_position(obs, 0, None))

    #network test
    net = KeeperNet()

    situation = torch.rand(960) #create random situation 
    heads, value = net(situation) #from the situation get the scores and value from the actor and critic

    for head in heads:
         print(len(head))

    print(value)