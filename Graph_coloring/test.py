a = "100010010100"
n = 4
step = int(len(a) /n)
for i in range(0, len(a), step):
    print(a[i:i+step])