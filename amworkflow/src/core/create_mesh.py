from src.core.create_geometry import CreateWall

class CreateMesh(CreateWall):
    def __init__(self, yaml_file_dir: str, yaml_file_name: str):
        super().__init__(yaml_file_dir, yaml_file_name)
        