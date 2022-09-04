import numpy as np

print("""
Genetic Algorithm in Python for finding the Solution of 
Linear Diophantine Equation a + 2b + 3c + 4d = 30. 
f(x) = a + 2b + 3c + 4d - 30 is minimised to get the solution of Linear Diophantine Equation. 

The stages are: 
1. Random initialization of Population 
2. Calculating the fitness value 
3. Selecting the fittest chromosomes for Crossover 
4. Performing the Crossover Operation 
5. Picking the chromosome for Mutation 
6. Performing the Mutation Operation 
7. Checking the fitness value of the New Population 
8. Displaying the Possible Solution 
""")

# Random Initialization of Chromosome Population ('n' is the number of chromosomes in the population)
n = 6
chromosome = np.random.randint(1, 9, (n, 4))
print("Random Initial Population:\n", chromosome)

# Calculation of the Objective function for fitness
epoch = 1
# f(x) = 30 - a + 2b + 3c + 4d
objective = abs(30 - chromosome[:, 0] - 2 * chromosome[:, 1] - 3 * chromosome[:, 2] - 4 * chromosome[:, 3])
# Fitness
fitness = 1 / (1 + objective)
print("\nInitial fitness values are: ", objective)

val1, val2, val3, val4 = 0, 0, 0, 0
while np.any(objective) != 0.:
    print("\nGeneration Number:", epoch)
    # Computation of fitness function
    objective = abs(30 - chromosome[:, 0] - 2 * chromosome[:, 1] - 3 * chromosome[:, 2] - 4 * chromosome[:, 3])
    # Selection of the fittest chromosome for recombination
    fitness = 1 / (1 + objective)
    # Calculating the total of fitness function
    total = fitness.sum()
    # Calculating Probability for each chromosome
    prob = fitness / total
    # Selection using Roulette Wheel by Calculating Cumulative Probability
    cum_sum = np.cumsum(prob)
    rand_nums = np.random.random((chromosome.shape[0]))
    # Making a new matrix of chromosome for calculation purpose
    chromosome_2 = np.zeros((chromosome.shape[0], 4))

    for i in range(rand_nums.shape[0]):
        for j in range(chromosome.shape[0]):
            if rand_nums[i] < cum_sum[j] and chromosome_2[i, :].any() != chromosome[j, :].any():
                chromosome_2[i, :] = chromosome[j, :]
                break

    chromosome = chromosome_2

    # Preparing for Crossover operations
    # Crossover Rate pc
    pc = 0.25
    flag = rand_nums < pc
    # Determining the chromosomes for crossover
    cross_chromosome = chromosome[[(i == True) for i in flag]]
    len_cross_chrom = len(cross_chromosome)

    # Calculating crossover point
    cross_values = np.random.randint(1, 3, len_cross_chrom)
    cpy_chromosome = np.zeros(cross_chromosome.shape)

    # Performing Crossover
    for i in range(cross_chromosome.shape[0]):
        cpy_chromosome[i, :] = cross_chromosome[i, :]

    if len_cross_chrom == 1:
        cross_chromosome = cross_chromosome
    else:
        for i in range(len_cross_chrom):
            c_val = cross_values[i]
            if i == len_cross_chrom - 1:
                cross_chromosome[i, c_val:] = cpy_chromosome[0, c_val:]
            else:
                cross_chromosome[i, c_val:] = cpy_chromosome[i + 1, c_val]

    index_chromosome = 0
    index_new_chromosome = 0
    for i in flag:
        if i:
            chromosome[index_chromosome, :] = cross_chromosome[index_new_chromosome, :]
            index_new_chromosome = index_new_chromosome + 1
        index_chromosome = index_chromosome + 1

    print("Updated Population after Crossover:\n", chromosome)

    # Preparing for Mutation Operation
    a, b = chromosome.shape[0], chromosome.shape[1]
    total_gen = int(a * b)

    # Mutation rate = pm
    pm = 0.1
    no_of_mutations = int(np.round(pm * total_gen))

    # Performing the Mutation of the selected chromosome
    gen_num = np.random.randint(1, total_gen - 1, no_of_mutations)
    replacing_num = np.random.randint(1, 9, no_of_mutations)

    for i in range(no_of_mutations):
        a = gen_num[i]
        row = a // 4
        col = a % 4
        chromosome[row, col] = replacing_num[i]

    print("Updated Population after Mutation:\n", chromosome)
    val1 = chromosome[0][0] + 2 * chromosome[0][1] + 3 * chromosome[0][2] + 4 * chromosome[0][3]
    val2 = chromosome[1][0] + 2 * chromosome[1][1] + 3 * chromosome[1][2] + 4 * chromosome[1][3]
    val3 = chromosome[2][0] + 2 * chromosome[2][1] + 3 * chromosome[2][2] + 4 * chromosome[2][3]
    val4 = chromosome[3][0] + 2 * chromosome[3][1] + 3 * chromosome[3][2] + 4 * chromosome[3][3]

    if val1 == 30 or val2 == 30 or val3 == 30 or val4 == 30:
        print("\nSolution... after ", epoch, " generations")
        break
    else:
        epoch = epoch + 1
        continue

# Displaying the Possible Solutions
if val1 == 30:
    print("Total Generations: ", epoch)
    print("One of the Possible Solutions is: ", chromosome[0])
    print("After substitution of above alleles, the Diophantine Equation { a + 2*b + 3*c + 4*d } is satisfied as::\n",
          int(chromosome[0][0]), "+ ( 2 *", int(chromosome[0][1]), ") + ( 3 *", int(chromosome[0][2]), ") + ( 4 *",
          int(chromosome[0][3]), ") = ", val1)
if val2 == 30:
    print("Total Generations : ", epoch)
    print("One of the Possible Solutions is : ", chromosome[1])
    print("After substitution of above alleles, the Diophantine Equation { a + 2*b + 3*c + 4*d } is satisfied as::\n",
          int(chromosome[1][0]), "+ ( 2 *", int(chromosome[1][1]), ") + ( 3 *", int(chromosome[1][2]), ") + ( 4 *",
          int(chromosome[1][3]), ") = ", val2)
if val3 == 30:
    print("Total Generations : ", epoch)
    print("One of the Possible Solutions is : ", chromosome[2])
    print("After substitution of above alleles, the Diophantine Equation { a + 2*b + 3*c + 4*d } is satisfied as::\n",
          int(chromosome[2][0]), "+ ( 2 *", int(chromosome[2][1]), ") + ( 3 *", int(chromosome[2][2]), ") + ( 4 *",
          int(chromosome[2][3]), ") = ", val3)
if val4 == 30:
    print("Total Generations : ", epoch)
    print("One of the Possible Solutions is : ", chromosome[3])
    print("After substitution of above alleles, the Diophantine Equation { a + 2*b + 3*c + 4*d } is satisfied as::\n",
          int(chromosome[3][0]), "+ ( 2 *", int(chromosome[3][1]), ") + ( 3 *", int(chromosome[3][2]), ") + ( 4 *",
          int(chromosome[3][3]), ') = ', val4)
