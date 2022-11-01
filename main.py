arr = []
for i in range(4):
    arr.append(input().split(" "))
print(arr)
sum1 = 0
for i in range(4):
    sum1 += float(arr[i][i])
sum2 = 0
for i in range(4):
    sum2 += float(arr[i][3-i])
print(sum1, sum2)