import encoding

def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1.dot(X) + b1
    A1 = encoding.ReLu(Z1)
    Z2 = W2.dot(A1) + b2 
    A2 = encoding.softmax(Z2)
    return Z1, A1, Z2 , A2