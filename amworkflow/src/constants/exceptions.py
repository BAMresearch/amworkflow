import numpy as np
class DimensionViolationException(Exception):
    pass

class GmshUseBeforeInitializedException(Exception):
    def __init__(self, message = "Gmsh must be initialized first!"):
        self.message = message
        super().__init__(message)

class DimensionInconsistencyException(Exception):
    def __init__(self, arr_a, arr_b):
        self.dim_a = np.array(arr_a).shape()[0]
        self.dim_b = np.array(arr_b).shape()[0]
        self.message = f"Dimensions are inconsistent. Got Dim A {self.dim_a}, Dim B {self.dim_b}."
        super().__init__(self.message)
        
class NoDataInDatabaseException(Exception):
    def __init__(self, item: str):
        self.message = f"{item} not found."
        super().__init__(self.message)
        
class InvalidGeometryException(Exception):
    def __init__(self, item: str):
        self.message = f"{item} is not valid, check input values."
        super().__init__(self.message)
        
class InvalidFileFormatException(Exception):
    def __init__(self, item: str):
        self.message = f"Invalid file format {item} imported."
        super().__init__(self.message)

class InsufficientDataException(Exception):
    def __init__(self):
        self.message = "Insufficient Data, exit."
        super().__init__(self.message)