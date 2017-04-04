# -*- coding: utf-8 -*-

def create(name, base, **kargs):
    '''
    Create a class during execution
    '''
    dict_inst = {}
    dict_cls = {}
    for obj_name, obj in kargs.iteritems():
        if isinstance(obj, type):
            dict_inst[obj_name] = obj
        else:
            dict_cls[obj_name] = obj


    # A DeprecationWarning is raised when the object inherits from the 
    # class "object" which leave the option of passing arguments, but
    # raise a warning stating that it will eventually stop permitting
    # this option. Usually this happens when the base class does not
    # override the __init__ method from object.
    def initType(self, *args, **kargs):
        """Replace the __init__ function of the new type, in order to
        add attributes that were defined with **kargs to the instance.
        """
        for obj_name, obj in dict_inst.iteritems():
            setattr(self, obj_name, obj())
        if base.__init__ is not object.__init__:
            base.__init__(self, *args, **kargs)

    objtype = type(str(name), (base,), dict_cls)
    objtype.__init__ = initType
    return  objtype


if __name__ == "__main__":
    a = create("liste",list,nb=0)
    print(type(a))
