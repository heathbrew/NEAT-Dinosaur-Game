import pygame
import os
import random
import math
import sys
import neat

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]

JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

# Initialize the font module
pygame.font.init()
FONT = pygame.font.Font('freesansbold.ttf', 20)

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    JUMP_VEL = 8.5

    def __init__(self, img=RUNNING[0]):
        self.image = img
        self.dino_run = True
        self.dino_jump = False
        self.jump_vel = self.JUMP_VEL
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.step_index = 0

    def update(self):
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.step_index >= 10:
            self.step_index = 0

    def jump(self):
        self.image = JUMPING
        if self.dino_jump:
            self.rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel <= -self.JUMP_VEL:
            self.dino_jump = False
            self.dino_run = True
            self.jump_vel = self.JUMP_VEL

    def run(self):
        self.image = RUNNING[self.step_index // 5]
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
        self.step_index += 1

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))
        pygame.draw.rect(SCREEN, self.color, (self.rect.x, self.rect.y, self.rect.width, self.rect.height), 2)
        for obstacle in obstacles:
            pygame.draw.line(SCREEN, self.color, (self.rect.x + 54, self.rect.y + 12), obstacle.rect.center, 2)

class Obstacle:
    def __init__(self, image, number_of_cacti):
        self.image = image
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 325

class LargeCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 300

def remove(index):
    dinosaurs.pop(index)
    ge.pop(index)
    nets.pop(index)

def distance(pos_a, pos_b):
    dx = pos_a[0]-pos_b[0]
    dy = pos_a[1]-pos_b[1]
    return math.sqrt(dx**2+dy**2)

def calculate_loss(obstacle_distance, output):
    target = 1 if obstacle_distance < 100 else 0
    epsilon = 1e-15  # Small epsilon value to prevent log(0) or log(1)
    output = max(epsilon, min(1 - epsilon, output))  # Clip output to be in (epsilon, 1-epsilon) range
    loss = -(target * math.log(output) + (1 - target) * math.log(1 - output))
    return loss



def eval_genomes(genomes, config, seegenomes=False):
    global game_speed, x_pos_bg, y_pos_bg, obstacles, dinosaurs, ge, nets, points
    
    def check_collision(dinosaur, obstacles):
        for obstacle in obstacles:
            if dinosaur.rect.colliderect(obstacle.rect):
                return True
        return False
    
    clock = pygame.time.Clock()
    points = 0

    obstacles = []
    dinosaurs = []
    ge = []
    nets = []

    x_pos_bg = 0
    y_pos_bg = 380
    game_speed = 20

    # Initialize the genomes counter
    genomes_count = len(genomes)

    for genome_id, genome in genomes:
        # Decrement the genomes counter
        genomes_count -= 1
        dinosaurs.append(Dinosaur())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0
        print(f"Remaining Genomes: {genomes_count}")


    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1
        text = FONT.render(f'Points:  {str(points)}', True, (0, 0, 0))
        SCREEN.blit(text, (950, 50))

    def statistics():
        global dinosaurs, game_speed, ge
        text_1 = FONT.render(f'Dinosaurs Alive:  {str(len(dinosaurs))}', True, (0, 0, 0))
        text_2 = FONT.render(f'Generation:  {pop.generation+1}', True, (0, 0, 0))
        text_3 = FONT.render(f'Game Speed:  {str(game_speed)}', True, (0, 0, 0))

        SCREEN.blit(text_1, (50, 450))
        SCREEN.blit(text_2, (50, 480))
        SCREEN.blit(text_3, (50, 510))

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= game_speed

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.fill((255, 255, 255))

        for dinosaur in dinosaurs:
            dinosaur.update()
            dinosaur.draw(SCREEN)

        if len(dinosaurs) == 0:
            break

        if len(obstacles) == 0:
            rand_int = random.randint(0, 1)
            if rand_int == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS, random.randint(0, 2)))
            elif rand_int == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS, random.randint(0, 2)))

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()

        for i, dinosaur in enumerate(dinosaurs):
            # Collision detection and handling
            if check_collision(dinosaur, obstacles):
                ge[i].fitness -= 50  # Adjust this penalty as needed
                remove(i)
                continue  # Skip the rest of the loop for this dinosaur

            for obstacle in obstacles:
                obstacle_distance = distance((dinosaur.rect.x, dinosaur.rect.y), obstacle.rect.midtop)
                output = nets[i].activate((dinosaur.rect.y, obstacle_distance))[0]
                loss = calculate_loss(obstacle_distance, output)
                ge[i].fitness += loss

                if seegenomes == True:
                    print(f"Genome ID: {ge[i].key}, Loss: {loss:.3f}, Total Loss: {ge[i].fitness:.3f}")
                    

                if output > 0.5 and dinosaur.rect.y == dinosaur.Y_POS:
                    dinosaur.dino_jump = True
                    dinosaur.dino_run = False

        statistics()
        score()
        background()
        clock.tick(30)
        pygame.display.update()


import copy
import warnings
import graphviz

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    
    # Check if graphviz is available
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # Default settings for node names and colors
    if node_names is None:
        node_names = {}
    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}
    assert type(node_colors) is dict

    # Default attributes for network nodes
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    # Create a Digraph object
    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    # Add input nodes
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    # Add output nodes
    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    # Determine which nodes are used if pruning is enabled
    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    # Add non-input, non-output nodes
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    # Add connections between nodes
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    # Save or view the resulting graph
    if filename is not None:
        # Save the Dot representation to a file
        dot.save(str(filename) + '.gv')
        # dot.render(filename, view=view)
    elif view:
        print(dot.source)
        # dot.view()

import pickle  # Import pickle at the beginning

# Function to save the best genome
def save_genome(genome, filename="best_genome.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(genome, file)
    print(f"Best genome saved to {filename}")
    
    
# Function to check if best genome file exists and load it
def check_and_load_genome(filename="best_genome.pkl"):
    if os.path.isfile(filename):
        print(f"Found saved genome: {filename}")
        with open(filename, "rb") as file:
            return True, pickle.load(file)
    else:
        print("No saved genome found. Starting fresh.")
        return False, None

# Function to create a new population with all genomes copied from a saved genome
import copy

def create_population_from_genome(config, saved_genome):
    population = neat.Population(config)
    
    # Iterate over each genome in the population
    for _, genome in population.population.items():
        # Update each attribute of the genome to match the saved genome
        # This assumes that the genome is a NEAT DefaultGenome or similar
        genome.connections = copy.deepcopy(saved_genome.connections)
        genome.nodes = copy.deepcopy(saved_genome.nodes)
        genome.key = saved_genome.key
        genome.fitness = saved_genome.fitness
        print("using last saved genome")

    return population


# Modify the run function to include saving the best genome
# Modify the run function to use the loaded genome if it exists
# Modify the run function to include using the saved genome
def run(config_path, genomes):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    genome_exists, loaded_genome = check_and_load_genome("best_genome.pkl")

    if genome_exists:
        pop = create_population_from_genome(config, loaded_genome)
    else:
        print("No saved genome found. Starting fresh with a new population.")
        pop = neat.Population(config)

    winner = pop.run(eval_genomes, genomes)

    # Save the new best genome
    save_genome(winner)

    # Create and draw the best network
    best_net = neat.nn.FeedForwardNetwork.create(winner, config)
    draw_net(config, winner, view=True, filename="network")
    
    
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path,genomes=25)

# dot -Tpng network.gv -o output.png