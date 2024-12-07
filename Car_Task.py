#@title The `Button` class

from dm_control.composer.observation import observable
from dm_control import composer
from dm_control import mjcf
import numpy as np


from dm_control.locomotion.arenas import floors
from dm_control.utils import transformations

# NUM_SUBSTEPS = 25  # The number of physics substeps per control timestep.
CONTROL_TIMESTEP = 0.05
PHYSICS_TIMESTEP = 0.005

OBSTACLE_DIMS = (0.2, 0.2, 0.3)

#@title Random initialiser using `composer.variation`
from dm_control.composer import variation
from dm_control.composer.variation import distributions



#@title The `PressWithSpecificForce` task
from heightfield_arena import HeightFieldArena

class PressWithSpecificForce(composer.Task):

  def __init__(self, creature):
    
    self._arena = floors.Floor()
    self._arena.mjcf_model.option.gravity = (0, 0, -9.81)
    self._creature = creature
    self.walls = []
    self.radius = 8
    self.add_random_obstacles(num_obstacles=50, radius=6, min_radius=0.4)
    self._arena.add_free_entity(self._creature)
    # self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

    self._creature.observables.enable_all()

    self.set_timesteps(CONTROL_TIMESTEP, PHYSICS_TIMESTEP)
    self._last_positions = np.empty((500, 2), dtype=np.float32)
    self._last_positions.fill(np.inf)
    self._last_vel = np.zeros(2)

  
   

 

  @property
  def root_entity(self):
    return self._arena

  @property
  def task_observables(self):
    # return self._task_observables
    return {}
  
  
  
  def add_random_obstacles(self, num_obstacles, radius, min_radius):

    for i in range(num_obstacles):
      # Random angle and distance within the specified radius
      # angle = np.random.uniform(0, 2*np.pi)
      # distance = np.random.uniform(1, radius)

      # Calculate x, y positions based on the angle and distance
      # x = distance * np.cos(angle)
      # y = distance * np.sin(angle)

      # x = np.random.uniform(-8,8) 
      # y = np.random.uniform(-8,8)
      x,y = self.random_obstacle_position(min_radius, max_radius=radius)

      z = 0.2  # Height of the obstacle

      
      # Add the obstacle to the environment
      
      
      wall = self._arena.mjcf_model.worldbody.add(
                'geom', type="box", mass="1.2", contype="1", friction="0.4 0.005 0.00001",
                conaffinity="1", size=OBSTACLE_DIMS, rgba=(0.8, 0.3, 0.3, 1), pos=[x, y, OBSTACLE_DIMS[2] / 2]
            )
      self.walls.append(wall)


    walls = [
    {"name": "wall_pos_x", "size": (0.1, 8, 0.1), "pos": (8, 0, 0.1)},
    {"name": "wall_neg_x", "size": (0.1, 8, 0.1), "pos": (-8, 0, 0.1)},
    {"name": "wall_pos_y", "size": (8, 0.1, 0.1), "pos": (0, 8, 0.1)},
    {"name": "wall_neg_y", "size": (8, 0.1, 0.1), "pos": (0, -8, 0.1)}
    ]

    for wall in walls:
      bound_wall = self._arena.mjcf_model.worldbody.add(
          'geom',
          name=wall["name"],         # Unique name for each wall
          contype="1",
          conaffinity="1",
          friction="0.4 0.005 0.00001",
          type="box",
          size=wall["size"],         # Wall thickness and height
          pos=wall["pos"],           # Position for each wall
          rgba=(0.3, 0.3, 0.3, 1)    # Gray color for walls
      )
      self.walls.append(bound_wall)


  def random_obstacle_position(self, min_radius, max_radius):
    while True:
      # Random x and y in the square (-max_radius, max_radius)
      x = np.random.uniform(-max_radius, max_radius)
      y = np.random.uniform(-max_radius, max_radius)
      
      # Check the distance from the origin (0, 0)
      distance = int(np.sqrt(x**2 + y**2))
      
      # If the point is outside the exclusion radius, return it
      if distance >= min_radius:
          return x, y





  def initialize_episode(self, physics, random_state):

    super().initialize_episode(physics, random_state)
    self._arena.initialize_episode(physics, random_state)
    # x,y = self.random_obstacle_position(0, max_radius= self.radius)
    x = 0
    y = 0
    self._creature.set_pose(physics, np.array([x, y, 0]), transformations.euler_to_quat([0, 0, 0]))
    self._creature_geomids = set(physics.bind(self._creature.mjcf_model.find_all('geom')).element_id)
    self._last_positions.fill(np.inf)




  def should_terminate_episode(self, physics):
    return False

  def after_step(self, physics, random_state):
    car_pos, car_quat = self._creature.get_pose(physics)
    self._last_positions = np.roll(self._last_positions, -1, axis=0)
    self._last_positions[-1] = car_pos[:2]



  def get_reward(self, physics):
    check = 0

    reward = self._compute_collision_penalty(physics)
    if reward!=0:
      check = 3

    # min_depth = self.depth_penalty(physics)
    # depth_penalty = 1/(min_depth + 1e-5)
    linear_vel = np.linalg.norm(self._creature.observables.body_vel_2d(physics))

    reward +=  linear_vel 
    return reward,check 
  
  def depth_penalty(self, physics):
    image = self._creature.observables.realsense_camera(physics)
    height, width, _ = image.shape
    region = image[:height // 2, :]
    min_depth = np.min(region)

    return min_depth


  def _compute_collision_penalty(self, physics):
    car_geom_ids = set(physics.bind(self._creature.mjcf_model.find_all('geom')).element_id)
    obstacle_geom_ids = set(physics.bind(self.walls).element_id)
    for contact in physics.data.contact:
      if (contact.geom1 in car_geom_ids and contact.geom2 in obstacle_geom_ids) or \
        (contact.geom2 in car_geom_ids and contact.geom1 in obstacle_geom_ids):
        
        return -100
    return 0
  
  
  def getObstacles(self):
    obs_pos = []
    for wall in self.walls:
        obs_pos.append(wall.pos)
    return np.array(obs_pos)


     