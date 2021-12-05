import numpy as np 


class BatchNorm:
    def __init__(self, input_dims, alpha=1e-4, train = True):
        self.g = np.random.randn(1, input_dims)
        self.b = np.zeros((1, input_dims))

        self.db = np.zeros(self.b.shape)
        self.dg = np.zeros(self.g.shape)

        self.mu = np.zeros((1, input_dims))
        self.s2 = np.zeros((1, input_dims))
        self.train = train
        self.alpha = alpha

        self.params = {}

    def forward(self, z):
        if self.train:
            mu = np.sum(z, axis=0, keepdims=True)/z.shape[0]
            s2 = np.sum((z - mu)*(z - mu), axis=0, keepdims=True)/z.shape[0]

            z_norm = (z - mu)/np.sqrt(s2 + 1e-8)


            self.mu = 0.97 * self.mu + 0.03* mu
            self.s2 = 0.97 * self.s2 + 0.03* s2

            Z_bn = z_norm * self.g + self.b

            self.params = {"Z_bn": Z_bn, "Z_norm":z_norm, "Z": z,  "mu":mu, "s2":s2}
            return Z_bn



        z_norm = (z - self.mu)/np.sqrt(self.s2 + 1e-8)
        return z_norm * self.g + self.b

    def backward(self, dZ_bn):
        mu = self.params["mu"]
        s2 = self.params["s2"]
        Z = self.params["Z"]
        Z_norm = self.params["Z_norm"]

        e = 1e-8

        M = Z.shape[0]

        zu = Z - mu
        dZ_norm = dZ_bn * self.g
        dZ = (np.sum(dZ_norm * Z_norm, axis=0, keepdims=True)/M * np.sum(Z_norm, axis=0, keepdims=True) + M * dZ_norm - np.sum(dZ_norm, axis=0, keepdims=True) - Z_norm * np.sum(dZ_norm * Z_norm, axis=0, keepdims=True)) / (M * np.sqrt(s2 + e)) 

        dg = np.sum(dZ_bn * Z_norm, axis=0, keepdims=True)/M
        db = np.sum(dZ_bn, axis=0, keepdims=True)/M

        self.dg = dg
        self.db = db

        return dZ, db, dg

    def step(self):
        self.b -= self.alpha * self.db
        self.g -= self.alpha * self.dg

if __name__ == "__main__":
    bn = BatchNorm(10)
    bn.train = True
    z = np.random.randn(100, 10)

    bn.forward(z)
    dZ_norm, dB, dg = bn.backward(z)

    print(dZ_norm.shape, dB.shape, dg.shape)
