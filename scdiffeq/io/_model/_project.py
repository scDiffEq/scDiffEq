
import ABCParse
import pathlib


class Project(ABCParse.ABCParse):
    def __init__(self, path=pathlib.Path("./").absolute(), *args, **kwargs):

        self.__parse__(locals())

        for k, v in self._VERSION_PATHS.items():
            setattr(self, k, v)

    @property
    def _PROJECT_PATH(self):
        if isinstance(self._path, pathlib.Path):
            return self._path
        elif isinstance(self._path, str):
            return pathlib.Path(self._path)
        else:
            raise TypeError("arg: `path` must be of type: [pathlib.Path, str]")

    @property
    def _VERSION_PATHS(self):
        version_paths = sorted(list(self._PROJECT_PATH.glob("version_*")))
        return {path.name: path for path in version_paths}
    
    def __repr__(self):
        return """scDiffEq"""