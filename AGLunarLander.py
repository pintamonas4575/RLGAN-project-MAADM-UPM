from gymnasium.wrappers.common import TimeLimit
import random
import numpy as np
import copy
import multiprocessing
import concurrent.futures

from loky import get_reusable_executor


from MLP import MLP

class AG_Lunar_Lander():

    def __init__(self, population_size: int, num_ind_exp: int, MLP: MLP, env: TimeLimit, env_seed: int = None):
        """Create the population"""
        self.MLP = MLP
        self.env = env
        self.env_seed = env_seed
        self.population_size = population_size
        self.chromosome_length = len(self.MLP.to_chromosome()) # Chromosome length
        self.num_ind_exp = num_ind_exp # Number of experiments per individual
        self.max_fitnesses = []
        self.min_fitnesses = []
        self.mean_fitnesses = []
        self.best_global_individual = (-9999, [])
        # ---------Create the population--------- #
        self.population = np.random.uniform(-5, 5, size=(population_size, self.chromosome_length)).tolist()
        self.fitnesses = []
    # ---------------------------Fitness--------------------------- #
    def fitness_lunar_lander(self, chromosome: list) -> float:
        """Evalua un cromosoma"""
        def policy(observation) -> int:
            epsilon = 0.10
            s = self.MLP.forward(observation)
            if np.random.rand() < epsilon:
                action = np.random.randint(len(s))
            else:
                action = np.argmax(s)
            return action
        def run() -> float:
            observation, info = self.env.reset(seed=self.env_seed)
            racum = 0
            while True:
                action = policy(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                racum += reward
                if terminated or truncated:
                    return racum 
        # ---------------------------------- # 
        self.MLP.from_chromosome(chromosome)
        reward = 0
        for _ in range(self.num_ind_exp):
            reward += run()
        return reward/self.num_ind_exp
    # ---------------------------Fitness--------------------------- #
    def sort_pop(self, reverse_sort: bool) -> list[float]:
        # Paralelizar la evaluación de la población
        # with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-2) as executor:
            # fitness_scores = list(executor.map(self.fitness_lunar_lander, self.population))

        # with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-2) as executor:
            # fitness_scores = list(executor.map(self.fitness_lunar_lander, self.population))

        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()-2) as pool: # 16 CPU cores
            # fitness_scores = pool.map(self.fitness_lunar_lander, self.population) # la de antes

        # lista = sorted(zip(self.population, fitness_scores), key=lambda x: x[1], reverse=reverse_sort)
        # self.population = [x[0] for x in lista]
        # self.fitnesses = [x[1] for x in lista]
        executor = get_reusable_executor()
        fitness_list=executor.map(self.fitness_lunar_lander,self.population)
        # fitness_list = [self.fitness_lunar_lander(ind) for ind in self.population]
        lista = sorted(zip(self.population, fitness_list), key=lambda x: x[1], reverse=reverse_sort)
        self.population, self.fitnesses = executor.map(list, zip(*lista))
    
    def select(self, T: int) -> list[list]:
        """Return a copy of an indivudual by tournament selection. Population already ordered by fitness"""
        choices=random.sample(copy.copy(self.population), k=T)
        indices=[self.population.index(c) for c in choices]
        return self.population[np.argmin(indices)]
    
    # def crossover22(self, parent1: list, parent2: list, pcross: float) -> tuple[list, list]:
    #     """One point crossover. Return two children"""
    #     if random.random() < pcross:
    #         crossover_point = random.randint(1, self.chromosome_length - 1)
    #         child1 = parent1[:crossover_point] + parent2[crossover_point:]
    #         child2 = parent2[:crossover_point] + parent1[crossover_point:]
    #     else:
    #         child1, child2 = parent1[:], parent2[:]

    #     return child1, child2
    
    def crossover(self, parent1: list, parent2: list, pcross: float) -> tuple[list, list]:
        """BLX-alpha crossover. Return two children"""
        alpha: float = 0.5
        if random.random() < pcross:
            child1 = []
            child2 = []
            for gene1, gene2 in zip(parent1, parent2):
                # Se define el intervalo entre los genes de los padres
                x_min = min(gene1, gene2)
                x_max = max(gene1, gene2)
                I = x_max - x_min
                # Se amplía el rango según el factor alpha
                lower_bound = x_min - alpha * I
                upper_bound = x_max + alpha * I
                # Se generan dos genes aleatorios dentro del intervalo extendido
                c1 = random.uniform(lower_bound, upper_bound)
                c2 = random.uniform(lower_bound, upper_bound)
                child1.append(c1)
                child2.append(c2)
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, individual: list, pmut: float) -> list:
        """Mutate an individual, swap elements. Return mutated individual"""
        # def mutate_swap(individual: list, pmut: float) -> list:
        #     """ Mutación por intercambio """
        #     if random.random() < pmut:
        #         index1, index2 = random.choices(range(len(individual)), k=2)
        #         individual[index1], individual[index2] = individual[index2], individual[index1]
        #     return individual

        def mutate_gaussian(individual: list, pmut: float) -> list:
            """ Mutación gaussiana """
            if random.random() < pmut:
                individual = [gen + random.uniform(-1, 1) for gen in individual]
            return individual

        # def mutate_random(individual: list, pmut: float) -> list:
        #     """ Mutación aleatoria """
        #     if random.random() < pmut:
        #         index1, index2 = random.choices(range(len(individual)), k=2)
        #         individual[index1] = random.uniform(-1,1)
        #         individual[index2] = random.uniform(-1,1)
        #     return individual
        
        # mutations = [mutate_swap, mutate_gaussian, mutate_random]
        # operator = random.choice(mutations)
        return mutate_gaussian(individual, pmut)

    def evolve(self, pmut=0.1, pcross=0.7, ngen=100, T=6, trace=50, reverse_sort=False, elitism=False) -> None:
        """Evolution procedure. Initial population already created"""
        for i in range(ngen):
            new_pop = []
            self.sort_pop(reverse_sort)

            if i % trace == 0 or i == ngen-1:
                print(f"Nº gen: {i}, Best fitness: {self.fitnesses[0]}")

            self.max_fitnesses.append(self.fitnesses[0])
            self.min_fitnesses.append(self.fitnesses[-1])
            self.mean_fitnesses.append(sum(self.fitnesses)/len(self.fitnesses))
            self.best_global_individual = (self.fitnesses[0], self.population[0]) if self.fitnesses[0] > self.best_global_individual[0] else self.best_global_individual
            if elitism:
                new_pop.append(self.population[0])
                new_pop.append(self.population[1])
            while len(new_pop) != self.population_size:   
                individual1 = self.select(T)
                individual2 = self.select(T)
                child1, child2 = self.crossover(individual1, individual2, pcross)
                mutated1 = self.mutate(child1, pmut)
                mutated2 = self.mutate(child2, pmut)
                new_pop.append(mutated1)
                new_pop.append(mutated2)
                
            self.population = [*new_pop] # make a copy

            # if i % trace == 0 or i == ngen-1: # en la última gen se ordena
                # self.sort_pop(reverse_sort)
                # print(f"Nº gen: {i}, Best fitness: {self.fitnesses[0]}")