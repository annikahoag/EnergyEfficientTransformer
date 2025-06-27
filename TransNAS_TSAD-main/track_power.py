import os

def run_power(filename):
    command = 'nvidia-smi dmon -s p --format csv >>' + filename
    os.system(command)

print("in track_power")
# os.remove('power_output.csv')
with open('power_output.csv', 'w') as f:
    pass
run_power('power_output.csv')