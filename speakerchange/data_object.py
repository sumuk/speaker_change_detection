from collections import defaultdict

class segmentobj():
    def __init__(self,start_time=0,end_time=0,obj=None):
        self.start_time = start_time
        self.end_time = end_time
        self.obj = obj
    def __repr__(self):
        return str(self.start_time)+","+str(self.end_time)+','+str(self.obj)
    def __radd__(x,y):
        return str(y)+" "+str(x.start_time)+","+str(x.end_time)+','+str(x.obj)

class seg_object():
    def __init__(self,timing=None):
        if timing is None:
            self.timing = defaultdict(segmentobj)
    def add_seg(self,st,ed,obj):
        seg_no = len(self.timing)
        self.timing[seg_no+1] = segmentobj(st,ed,obj)
    def __repr__(self):
        lis = []
        for i,j in self.timing.items():
            lis.append(str(i)+j)
        return "\n".join(lis)