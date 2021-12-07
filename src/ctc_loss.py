import numpy as np
from ctc import *

class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        # Don't Need Modify
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """
        # -------------------------------------------->
        # Don't Need Modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        #####  Attention:
        #####  and then the mean over the batch is taken

        # -------------------------------------------->
        # Don't Need Modify
        B, _ = target.shape
        totalLoss = np.zeros(B)
        # <---------------------------------------------
        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------
            ctc = CTC(self.BLANK)
            tar = target[b, 0:target_lengths[b]]
            log = logits[0:input_lengths[b], b]
            extSymbols, skipConnect = ctc.targetWithBlank(tar)
            alpha = ctc.forwardProb(log, extSymbols, skipConnect)
            beta = ctc.backwardProb(log, extSymbols, skipConnect)
            gamma = ctc.postProb(alpha, beta)
            T, S = gamma.shape
            loss = 0.0
            #exp_log = np.exp(log)
            #log = exp_log / np.sum(exp_log, axis=1)[:, None]
            for t in range(T):
                for r in range(S):
                    loss -= gamma[t, r] * np.log(log[t, extSymbols[r]])
            # -------------------------------------------->
            totalLoss[b] = loss
            # Your Code goes here
            
            # <---------------------------------------------
        
        return float(np.mean(totalLoss))

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: (seqlength, batch_size, len(Symbols))
            derivative of divergence wrt the input symbols at each time.

        """
        # -------------------------------------------->
        # Don't Need Modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        # <---------------------------------------------

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            ctc = CTC(self.BLANK)
            tar = self.target[b, 0:self.target_lengths[b]]
            log = self.logits[0:self.input_lengths[b], b]
            extSymbols, skipConnect = ctc.targetWithBlank(tar)
            alpha = ctc.forwardProb(log, extSymbols, skipConnect)
            beta = ctc.backwardProb(log, extSymbols, skipConnect)
            gamma = ctc.postProb(alpha, beta)
            T, S = gamma.shape
            for t in range(T):
                for r in range(S):
                    dY[t, b, extSymbols[r]] -= 1/log[t, extSymbols[r]] * gamma[t, r]
            # Your Code goes here
            
            # <---------------------------------------------

        return dY
