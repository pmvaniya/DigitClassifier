def condense(canvas, t):
    n = int(len(canvas) ** 0.5)
    r = int(n/t)
    
    for i in range(len(canvas)):
        canvas[i] /= 100

    temp = []
    for i in range(n):
        temp.append(canvas[:n])
        canvas = canvas[n:]
    canvas = temp

    temp = []
    for i in range(len(canvas)):
        temp.append([])
        for j in range(t):
            temp[i].append(sum(canvas[i][:r]))
            canvas[i] = canvas[i][r:]
    canvas = temp

    temp = []
    for i in range(0,len(canvas),r):
        for j in range(i+1, i+r):
            for k in range(len(canvas[0])):
                canvas[i][k] += canvas[j][k]
        temp.append(canvas[i])
    canvas = temp

    temp = []
    for i in canvas:
        temp += i
    canvas = temp

    for i in range(len(canvas)):
        canvas[i] /= (t * t)

    return canvas