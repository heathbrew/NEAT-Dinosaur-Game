import math

def calculate_loss(obstacle_distance, output):
    target = 1 if obstacle_distance < 100 else 0
    loss = -(target * math.log(output) + (1 - target) * math.log(1 - output))
    return loss

class Dinosaur:

    def __init__(self):
        self.fitness = 0 
        # Initialization
        
    def evaluate(self, obstacle_distance):
        output = self.net.activate((self.rect.y, obstacle_distance))[0]
        loss = calculate_loss(obstacle_distance, output)
        self.fitness += loss
        
        print(f"Genome: {self.genome_id}, Loss: {loss:.3f}, Total Loss: {self.fitness:.3f}")

def eval_genomes(genomes, config):
    
    for obstacle in obstacles:
        for dinosaur in dinosaurs:
        
            obstacle_distance = distance(dinosaur.rect.midbottom, 
                                        obstacle.rect.midtop)
            
            dinosaur.evaluate(obstacle_distance)

    # Normalize fitness
    for dinosaur in dinosaurs:
        dinosaur.genome.fitness = max_fitness - dinosaur.fitness

def run(config):
    # Set up NEAT population
    
    pop.run(eval_genomes, 10)

if __name__ == '__main__':

    run(config)