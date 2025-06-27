import os 
import csv

# NUM_ITERS =10
# filename = 'output_before.csv'
# command = 'nvidia-smi dmon -s p -c ' + str(NUM_ITERS) + ' --format csv >' + filename
# print(command)
# os.system(command)

# # Get average GPU reading 
# with open('output_before.csv', 'r') as f:
#     reader = list(csv.reader(f, delimiter=','))

# sum=0
# for i in range(2, NUM_ITERS+2):
#     sum = sum + float(reader[i][1])
# print(sum)
# avg_power = sum / NUM_ITERS
# print(avg_power)

#os.remove('output_before.csv')

def run_power(filename):
    command = 'nvidia-smi dmon -s p --format csv >>' + filename
    os.system(command)


def read_power(filename):
    # command = 'nvidia-smi dmon -s p -c ' + str(num_iters) + ' --format csv >>' + filename
    # command = 'nvidia-smi dmon -s p --format csv >>' + filename
    # os.system(command)

    with open(filename, 'r') as f:
        reader = list(csv.reader(f, delimiter=','))[1:]
    
    #print(len(reader))
    
    # filename2 = filename + 'no_headings.csv'
    # with open(filename2, 'a') as f2:
    #     csvwriter = csv.writer(f2)
    #     csvwriter.writerows(reader[1:])

    # with open(filename2, 'r') as f3:
    #     reader = list(csv.reader(f3, delimiter=','))
    
    # return reader
    return len(reader)-1

def get_avg_power(filename):
    # command = 'nvidia-smi dmon -s p -c ' + str(num_iters) + ' --format csv >' + filename
    # # print(command)
    # os.system(command)

    # Get average GPU reading 
    with open(filename, 'r') as f:
        reader = list(csv.reader(f, delimiter=','))
        num_iters = len(reader)
    #print(reader)

    sum=0
    num_skips=0
    for i in range(num_iters):
        # if i%num_iters+2!=0 and i%num_iters+3!=0:
        if reader[i][0] != '#gpu' and reader[i][0] != '#Idx':
            sum = sum + float(reader[i][1])
        else:
            num_skips+=1
    #print(sum)
    avg_power = sum / (num_iters-num_skips)

    return avg_power

# os.remove('output_before.csv')
# run_power('output_before.csv')
#print(read_power('output_before.csv'))
# print(read_power('output_before.csv', 5))
# print("\n")
# print(read_power('output_before.csv', 5))
# print("\n")

print(get_avg_power('output_before.csv'))#, read_power('output_before.csv')))


# LEFT OFF: need to get it running where I do run_power then read and get_avg and then run power again and make sure the average works with the way I'm indexing because rn it works one time but no over and over
# probs gonna want to write a script that calls the functions so that we can keep running power and then call the other functions 