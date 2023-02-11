from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler
from iopath.common.file_io import PathManager as PathManagerBase

__all__ = ["path_mgr"]


path_mgr = PathManagerBase()
path_mgr.register_handler(HTTPURLHandler())
path_mgr.register_handler(OneDrivePathHandler())
