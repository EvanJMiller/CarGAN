
class CarAnnotation:

    def __init__(self, x1, x2, y1, y2, class_num, fname):
        self.box = (x1, x2, y1, y2)
        self.class_num = class_num
        self.fname = fname



