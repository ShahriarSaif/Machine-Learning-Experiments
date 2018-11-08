import random as rnd
#perceptron to learn to classify
#even and odd binary bit patterns
#even = 0, odd = 1

def dec_to_bin(a, n):
    pat = []
    invrtd = []
    while a != 0:
        r = a % 2
        a = int(a / 2)
        pat.append(r)
    i = len(pat)-1
    while i >= 0:
        invrtd.append(pat[i])
        i -= 1
    if len(invrtd) < n:
        m = n - len(invrtd)
        for i in range(m):
            invrtd.insert(0, 0)
    invrtd.append(1)
    return invrtd

def vec_add(a, b):
    r = []
    n = len(a)
    for i in range(n):
        r.append(a[i] + b[i])
    return r
    
def vec_sc_mul(a, k):
    r = [] 
    n = len(a)
    for i in range(n):
        r.append(a[i] * k)
    return r

def vec_dot(a, b):
    sum = 0
    n = len(a)
    for i in range(n):
        sum += a[i] * b[i]
    return sum

def perceptron(x, w):
    a = vec_dot(x, w)
    if a >= 0:
        return 1
    return 0

def train(tr, w, alpha):
    m = len(tr)
    for i in range(m):
        delta = tr[i][1] - perceptron(tr[i][0], w)
        w = vec_add(w, vec_sc_mul(vec_sc_mul(tr[i][0], delta), alpha))
    return w

def define_pattern(nf, m):
    pat = []
    for i in range(m):
        x = dec_to_bin(i, nf)
        y = 0
        if i % 2 == 0:
            y = 0
        else:
            y = 1
        pat.append((x, y))
    return pat

nf = int(input('enter the number of features(#binary bits to be used): ')) 
m = int((2 ** nf) / 2)
tr = define_pattern(nf, m)
w = []
for i in range(nf + 1):
    w.append(rnd.random())
w = train(tr, w, 0.3)
x = int(input('enter a number: '))
x = dec_to_bin(x, nf)
y = perceptron(x,w)
if y == 1:
    print('the number is: odd')
else:
    print('the number is: even')
print(w)