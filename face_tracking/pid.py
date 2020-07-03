import time



class PID:
    def __init__(self, kP=1, kI=0, kD=0):
        # initialize gains
        self.kP = kP
        self.kI = kI
        self.kD = kD

    def initialize(self):
        #intialize the current and previous time
        self.currTime = time.time()
        self.prevTime = self.currTime

        #initialize the previous error:
        self.prevError = 0

        #initialize the term result variables
        self.cP = 0
        self.cI = 0
        self.cD = 0
        
    def update(self, error, sleep = 0.010):
        #pause for a bit
        time.sleep(sleep)
        
        #grab the current time and calucalte delta time
        self.currTime = time.time()
        deltaTime = self.currTime - self.prevTime
        
        #delta error
        deltaError = error - self.prevError
        
        #proportional term
        self.cP = error

        #integral term
        #SHOULDNT THIS BE = ONLY? NOT +=
        self.cI += error * deltaTime
        
        #derivative terma nd prevent divide by zero
        self.cD = (deltaError / deltaTime) if deltaTime > 0 else 0
        
        #Save previous time and erorr for the next update
        self.prevtime = self.currTime
        self.prevError = error
        
        #sum the terms and return
        return sum( [self.kP * self.cP,
                     self.kI * self.cI,
                     self.kD * self.cD
                    ])