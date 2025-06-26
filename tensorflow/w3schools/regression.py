xArray = [50,60,70,80,90,100,110,120,130,140,150]
yArray = [7,8,8,9,9,9,10,11,14,14,15]
# linear regression
# Calculate Sums
xSum=0
ySum=0
xxSum=0
xySum=0
count = len(xArray)
for i in range(count):
  xSum += xArray[i]
  ySum += yArray[i]
  xxSum += xArray[i] * xArray[i]
  xySum += xArray[i] * yArray[i]

# Calculate slope and intercept
slope = (count * xySum - xSum * ySum) / (count * xxSum - xSum * xSum)
intercept = (ySum / count) - (slope * xSum) / count

# Generate values
xValues = []
yValues = []
for x in range(50,151):
  xValues.append(x)
  yValues.append(x * slope + intercept)
