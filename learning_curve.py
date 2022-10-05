import matplotlib.pyplot as plt

avg = []
max = []
min = []
with open('gen_info.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()

    for line in filecontents:
        # remove linebreak which is the last character of the string
        info = line[:-1].split(' ')
        max.append(info[0])
        avg.append(info[1])
        min.append(info[2])

plt.plot(avg, range(len(avg)), label="average")
plt.plot(max, range(len(max)), label="maximum")
plt.plot(min, range(len(min)), label="minimum")

plt.xlabel('generations')

plt.legend()

plt.show()


