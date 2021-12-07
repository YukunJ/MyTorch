import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here
        self.r_preact = None
        self.z_preact = None
        self.n_preact = None
        self.r = None
        self.z = None
        self.n = None
        
        
    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        
        # r gate
        r_preact = x @ self.Wrx.T + self.bir + h @ self.Wrh.T + self.bhr
        r = self.r_act(r_preact)
        self.r_preact = r_preact
        self.r = r
        
        # z gate
        z_preact = x @ self.Wzx.T + self.biz + h @ self.Wzh.T + self.bhz
        z = self.z_act(z_preact)
        self.z_preact = z
        self.z = z
        
        # n gate
        n_preact = x @ self.Wnx.T + self.bin + r * (h @ self.Wnh.T + self.bhn)
        n = self.h_act(n_preact)
        self.n_preact = n_preact
        self.n = n
        
        # final hidden output at current timestamp
        h_t = (1 - z) * n + z * h
        

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        # return h_t
        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.h to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        self.x = self.x.reshape(-1, 1)
        self.hidden = self.hidden.reshape(-1, 1)
        dx = np.zeros((1, self.d))
        dh = np.zeros((1, self.h))
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
        # first convert ht = (1-zt) * nt + zt * h_{t-1}
        dh += delta * self.z
        dn = delta * (1 - self.z)
        dz = delta * (self.hidden.flatten() - self.n)
        
        # next convert nt = tanh of W_in x + b_in + r * (w_hn h_{t-1}) + r * b_hn
        dn_preact = dn * self.h_act.derivative().reshape(1, -1)
        self.dWnx += np.dot(dn_preact.T, self.x.T)
        dx += np.dot(dn_preact, self.Wnx)
        self.dbin = (self.dbin + dn_preact).flatten()
        dr = dn_preact * (self.hidden.T @ self.Wnh.T + self.bhn)
        self.dbhn = (self.dbhn + dn_preact * self.r).flatten()
        self.dWnh += np.dot((dn_preact * self.r).T, self.hidden.T)
        dh += np.dot(dn_preact * self.r, self.Wnh)
        
        # next convert zt = sigmoid of W_iz x + b_iz + W_hz h_{t-1} + b_hz
        dz_preact = dz * self.z_act.derivative().reshape(1, -1)
        self.dWzx += np.dot(dz_preact.T, self.x.T)
        dx += np.dot(dz_preact, self.Wzx)
        self.dbiz = (self.dbiz + dz_preact).flatten()
        self.dWzh += np.dot(dz_preact.T, self.hidden.T)
        dh += np.dot(dz_preact, self.Wzh)
        self.dbhz = (self.dbhz + dz_preact).flatten()
        
        # finally convert rt = sigmoid of W_ir x + b_ir + W_hr h_{t-1} + b_hr
        dr_preact = dr * self.r_act.derivative().reshape(1, -1)
        self.dWrx += np.dot(dr_preact.T, self.x.T)
        dx += np.dot(dr_preact, self.Wrx)
        self.dbir = (self.dbir + dr_preact).flatten()
        self.dWrh += np.dot(dr_preact.T, self.hidden.T)
        dh += np.dot(dr_preact, self.Wrh)
        self.dbhr = (self.dbhr + dr_preact).flatten()
        
        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        # return dx, dh
        return dx, dh
