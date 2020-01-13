import ctypes

lib = ctypes.cdll.LoadLibrary('./plugin/libRoccoR.so')

class RoccoR(object):
    
    def __init__(self, filename):
        
        lib.RoccoR_new.argtypes = [ctypes.c_char_p]
        lib.RoccoR_new.restype  = ctypes.c_void_p
        
        lib.RoccoR_kScaleDT.argtypes = [ctypes.c_void_p, \
                                        ctypes.c_int, ctypes.c_double, ctypes.c_double, \
                                        ctypes.c_double, ctypes.c_int, ctypes.c_int]
        lib.RoccoR_kScaleDT.restype  = ctypes.c_double

        lib.RoccoR_kSpreadMC.argtypes = [ctypes.c_void_p, \
                                         ctypes.c_int, ctypes.c_double, ctypes.c_double, \
                                         ctypes.c_double, ctypes.c_double, ctypes.c_int, \
                                         ctypes.c_int]
        lib.RoccoR_kSpreadMC.restype  = ctypes.c_double

        lib.RoccoR_kSmearMC.argtypes = [ctypes.c_void_p, \
                                         ctypes.c_int, ctypes.c_double, ctypes.c_double, \
                                         ctypes.c_double, ctypes.c_int, ctypes.c_double, \
                                         ctypes.c_int, ctypes.c_int]
        lib.RoccoR_kSmearMC.restype = ctypes.c_double

        self.obj = lib.RoccoR_new(filename)
        
    def kScaleDT(self, Q, pt, eta, phi, s = 0, m = 0):

        return lib.RoccoR_kScaleDT(self.obj, \
                                   Q, pt, eta, phi, s, m)

    def kSpreadMC(self, Q, pt, eta, phi, gt, s = 0, m = 0):

        return lib.RoccoR_kSpreadMC(self.obj, \
                                    Q, pt, eta, phi, gt, s, m)

    def kSmearMC(self, Q, pt, eta, phi, n, u, s = 0, m = 0):

        return lib.RoccoR_kSmearMC(self.obj, \
                                   Q, pt, eta, phi, n, u, s, m)

            
