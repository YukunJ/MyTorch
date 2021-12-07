import numpy as np

class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        # self.b (np.array) : (self.out_channel)
        batch_size, in_channel, input_size = x.shape
        out_channel = self.out_channel
        output_size = (input_size-self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, out_channel, output_size))

        output_idx = 0
        for start in range(0, input_size-self.kernel_size+1, self.stride):
            slice_x = x[:, :, start:start+self.kernel_size]
            out[:, :, output_idx] = np.tensordot(slice_x, self.W, axes=([1, 2], [1, 2])) + self.b[None, :]
            output_idx += 1
        # for backward record
        self.x = x
        self.out = out
        return out
                
        

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        dx = np.zeros(self.x.shape)
        out_channel, output_size = delta.shape[1], delta.shape[2]
        batch_size, in_channel, input_size = self.x.shape

        if self.stride == 1:
            delta_upsampled = delta
        else:
            up = input_size - self.kernel_size + 1
            delta_upsampled = np.zeros((batch_size, out_channel, up))
            for b in range(batch_size):
                for out_channel_idx in range(out_channel):
                    for x in range(output_size):
                        delta_upsampled[b, out_channel_idx, x*self.stride] = delta[b, out_channel_idx, x]
        
        # zero padding
        delta_upsampled_padded = np.zeros((batch_size, out_channel, delta_upsampled.shape[2]+2*(self.kernel_size-1)))
        for b in range(batch_size):
            for out_channel_idx in range(out_channel):
                delta_upsampled_padded[b, out_channel_idx, self.kernel_size-1:self.kernel_size-1+delta_upsampled.shape[2]] = delta_upsampled[b, out_channel_idx, :]

        for b in range(batch_size):
            for x in range(output_size):
                m = x * self.stride
                for out_channel_idx in range(out_channel):
                    # db calculation
                    self.db[out_channel_idx] += delta[b, out_channel_idx, x]
                    for in_channel_idx in range(in_channel):
                        for x_prime in range(self.kernel_size):
                            # dW calculation
                            self.dW[out_channel_idx, in_channel_idx, x_prime] += delta[b, out_channel_idx,x] * self.x[b, in_channel_idx, m+x_prime]
            for in_channel_idx in range(in_channel):
                for in_idx in range(input_size):
                    dx[b, in_channel_idx,in_idx] = np.tensordot(self.W[:,in_channel_idx,::-1], delta_upsampled_padded[b,:, in_idx:in_idx+self.kernel_size], axes=([0, 1],[0, 1]))
        return dx

class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        batch_size, in_channel, input_width, input_height = x.shape
        out_channel = self.out_channel
        kernel_size = self.kernel_size
        stride = self.stride
        output_width = (input_width-kernel_size) // stride + 1
        output_height = (input_height-kernel_size) // stride + 1
        out = np.zeros((batch_size, out_channel, output_width, output_height))
        width_idx, height_idx = 0, 0
        for start_width in range(0, input_width-kernel_size+1, stride):
            height_idx = 0
            for start_height in range(0, input_height-kernel_size+1, stride):
                slice_x = x[:, :, start_width:start_width+kernel_size, start_height:start_height+kernel_size]
                out[:, :, width_idx, height_idx] = np.tensordot(slice_x, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b[None, :]
                height_idx += 1
            width_idx += 1
        self.x = x
        self.out = out
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        # W = dW = (out_channel, in_channel, kernel_size, kernel_size)
        # b = dW = (out_channel)
        # x = dx = (batch_size, in_channel, input_width, input_height)
        # delta = (batch_size, out_channel, output_width, output_height)
        batch_size, in_channel, input_width, input_height = self.x.shape
        out_channel = self.out_channel
        kernel_size = self.kernel_size
        stride = self.stride
        output_width = (input_width-kernel_size) // stride + 1
        output_height = (input_height-kernel_size) // stride + 1
        dx = np.zeros(self.x.shape)
        # create dilated output graident
        
        up_width = input_width - kernel_size + 1
        up_height = input_height - kernel_size + 1
        if stride == 1:
            delta_dilated = delta
        else:
            delta_dilated = np.zeros((batch_size, out_channel, up_width, up_height))
            for b in range(batch_size):
                for out_channel_idx in range(out_channel):
                    for x_width in range(output_width):
                        for x_height in range(output_height):
                            delta_dilated[b, out_channel_idx, x_width*stride, x_height*stride] = delta[b, out_channel_idx, x_width, x_height]
        # pad zero
        delta_dilated_padded = np.pad(delta_dilated,((0,0), (0,0), (kernel_size-1, kernel_size-1), (kernel_size-1, kernel_size-1)), mode='constant', constant_values=0)
        
        # dx: delta_dilated_padded convolve with filped filter
        flipped_W = np.rot90(self.W, k=2, axes=(2,3))
        for start_width in range(input_width):
            for start_height in range(input_height):
                dx[:, :, start_width, start_height] = np.tensordot(delta_dilated_padded[:,:,start_width:start_width+kernel_size, start_height:start_height+kernel_size], flipped_W, axes=([1,2,3],[0,2,3]))
                
        # dW: delta_dilated convolve with X
        for kernel_width_idx in range(kernel_size):
            for kernel_height_idx in range(kernel_size):
                self.dW[:, :, kernel_width_idx, kernel_height_idx] = np.tensordot(delta_dilated, self.x[:,:,kernel_width_idx:kernel_width_idx+up_width, kernel_height_idx:kernel_height_idx+up_height], axes = ([0,2,3],[0,2,3]))
        
        # db: aggregate
        self.db = np.sum(delta, axis=(0, 2, 3))
        return dx
        


class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Much like Conv2D, but take two attributes into consideration: padding and dilation.
        Make sure you have read the relative part in writeup and understand what we need to do here.
        HINT: the only difference are the padded input and dilated kernel.
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # After doing the dilation， the kernel size will be: (refer to writeup if you don't know)
        self.kernel_dilated = (kernel_size - 1) * (dilation -1) + kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        # TODO: padding x with self.padding parameter (HINT: use np.pad())
        batch_size, in_channel, input_width, input_height = x.shape
        out_channel = self.out_channel
        padding = self.padding
        dilation = self.dilation
        kernel_size = self.kernel_size
        kernel_dilated = self.kernel_dilated
        stride = self.stride
        padded_x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
        self.x = x # original x without padding
        self.padded_x = padded_x
        input_width_padded = input_width + 2 * padding
        input_height_padded = input_height + 2 * padding
        output_width = (input_width_padded - kernel_dilated) // stride + 1
        output_height = (input_height_padded - kernel_dilated) // stride + 1
        out = np.zeros((batch_size, out_channel, output_width, output_height))
        #       HINT: for loop to get self.W_dilated (self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated)
        for out_channel_idx in range(out_channel):
            for in_channel_idx in range(in_channel):
                for width_idx in range(kernel_size):
                    for height_idx in range(kernel_size):
                        self.W_dilated[out_channel_idx, in_channel_idx, width_idx * (dilation), height_idx * (dilation)] = self.W[out_channel_idx, in_channel_idx, width_idx, height_idx]
        # TODO: regular forward, just like Conv2d().forward()
        width_idx, height_idx = 0, 0
        for start_width in range(0, input_width_padded-kernel_dilated+1, stride):
            height_idx = 0
            for start_height in range(0, input_height_padded-kernel_dilated+1, stride):
                slice_x = padded_x[:, :, start_width:start_width+kernel_dilated, start_height:start_height+kernel_dilated]
                out[:, :, width_idx, height_idx] = np.tensordot(slice_x, self.W_dilated, axes=([1,2,3], [1,2,3])) + self.b[None, :]
                height_idx += 1
            width_idx += 1
        self.out = out
        return out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        #       for whole process while we only need original part of input and kernel for backpropagation.
        #       Please refer to writeup for more details.

        # for dx, use dilated kernel,
        # for dW, use padded input
        batch_size, in_channel, input_width, input_height = self.x.shape
        dx = np.zeros((batch_size, in_channel, input_width, input_height))
        dx_padded = np.zeros(self.padded_x.shape)
        out_channel = self.out_channel
        padding = self.padding
        dilation = self.dilation
        kernel_size = self.kernel_size
        kernel_dilated = self.kernel_dilated
        stride = self.stride
        input_width_padded = input_width + 2 * padding
        input_height_padded = input_height + 2 * padding
        output_width = (input_width_padded - kernel_dilated) // stride + 1
        output_height = (input_height_padded - kernel_dilated) // stride + 1
        x = self.x
        padded_x = self.padded_x
        out = self.out
        
        # create dilated output graident
        # just assume what we have is x_padded and W_dilated
        # compute the derivative w.r.t. x_padded and W_dilated, then transform them back
        up_width = input_width_padded - kernel_dilated + 1
        up_height = input_height_padded - kernel_dilated + 1
        if stride == 1:
            delta_dilated = delta
        else:
            delta_dilated = np.zeros((batch_size, out_channel, up_width, up_height))
            for b in range(batch_size):
                for out_channel_idx in range(out_channel):
                    for x_width in range(output_width):
                        for x_height in range(output_height):
                            delta_dilated[b, out_channel_idx, x_width*stride, x_height*stride] = delta[b, out_channel_idx, x_width, x_height]
        # pad zero
        delta_dilated_padded = np.pad(delta_dilated, ((0,0), (0, 0), (kernel_dilated-1, kernel_dilated-1), (kernel_dilated-1, kernel_dilated-1)), mode='constant', constant_values=0)
        # flip map
        flipped_W_dilated = np.rot90(self.W_dilated, k=2, axes=(2,3))
        # dx_padded
        for start_width in range(input_width_padded):
            for start_height in range(input_height_padded):
                dx_padded[:, :, start_width, start_height] = np.tensordot(delta_dilated_padded[:,:, start_width:start_width+kernel_dilated, start_height:start_height+kernel_dilated], flipped_W_dilated, axes=([1,2,3], [0,2,3]))
        # dx
        dx = dx_padded[:, :, padding:-padding, padding:-padding]
        # dW_dilated
        dW_dilated = np.zeros(self.W_dilated.shape)
        for kernel_width_idx in range(kernel_dilated):
            for kernel_height_idx in range(kernel_dilated):
                dW_dilated[:,:, kernel_width_idx, kernel_height_idx] = np.tensordot(delta_dilated, padded_x[:,:, kernel_width_idx:kernel_width_idx+up_width, kernel_height_idx:kernel_height_idx+up_height], axes=([0,2,3], [0,2,3]))
        # mapping back to dW
        for out_channel_idx in range(out_channel):
            for in_channel_idx in range(in_channel):
                for width_idx in range(kernel_size):
                    for height_idx in range(kernel_size):
                        self.dW[out_channel_idx, in_channel_idx, width_idx, height_idx] = dW_dilated[out_channel_idx, in_channel_idx, width_idx * dilation, height_idx * dilation]
        self.db = np.sum(delta, axis=(0, 2, 3))
        
        return dx

class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(self.b, self.c*self.w)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape(self.b, self.c, self.w)

