
class Buffer(object):
    '''
    A buffer is a queue with a max length constraint.
    (If an element is added to an already full buffer, the first item of the queue is dropped)
    '''
    def __init__(self, windowLength):
        self.rollingWindowLength = windowLength  # 0 means disabled
        self.lasts = []

    def addLast(self, last):
        '''
        Append and item to the queue and delete the first element of it if needed
        :param last: item to add
        '''
        if self.rollingWindowLength == 0:
            return
        self.lasts.append(last)
        self.lasts = self.lasts[-self.rollingWindowLength:]
