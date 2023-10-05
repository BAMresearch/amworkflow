class geometry():
    def __int__(self, parameters) -> None:
        """general geometry class


        """

        self.p = parameters

    def geometry_spawn(self):
        """define geometry using occ fct or own

            to be overwritten by user
        """
        self.shape = None

        pass
    def create(self):
        """ create step and stl file from the geometry
        """

        pass