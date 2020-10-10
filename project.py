import os
import sys
from dataclasses import dataclass


@dataclass
class Project:
    windows_fmu_prefix = os.path.join("resources", "windows")
    linux_fmu_prefix = os.path.join("resources", "linux")

    @classmethod
    def get_fmu_prefix(cls):
        if "win" in sys.platform:
            return cls.windows_fmu_prefix
        else:
            return cls.linux_fmu_prefix

    @classmethod
    def get_fmu(cls, fmu_name):
        return os.path.join(cls.get_fmu_prefix(), fmu_name)
