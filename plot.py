from itertools import islice

def skip_lines_generator(y):
    def skip_lines(x):
        return not x.startswith(y)
    return skip_lines

skip_id = skip_lines_generator('id')
skip_generation = skip_lines_generator('generation')
skip_average = skip_lines_generator('average')

def strip_score(x):
    return x.strip('score: ')

LOG = './20220414/20220414_10000mutations_log.txt'
NAME = '20220414_10000mutations_plot.txt'

GENERATIONS = 100
PER_GENERATION = 100

log = open(LOG, 'r')
log = islice(log, 4, None)

log = filter(skip_id, log)
log = filter(skip_generation, log)
log = filter(skip_average, log)
log = map(strip_score, log)

fp = open(NAME, 'w')

for line in log:
    if line.startswith('- - - - - generation'):
        num_list = filter(str.isdigit, line)
        gen = "".join(num_list)
        # print(int(gen))

        for i in range(PER_GENERATION):
            fp.write(str(gen)+'\t'+str(next(log)))
fp.close()
